# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import glob
import torch
import numpy as np
from scipy.stats import norm
import multiprocessing
import nltk
import einops
from einops import rearrange
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from core.tokenizer.mmtokenizer import _MMSentencePieceTokenizer
from core.datasets import indexed_dataset

def get_size_in_bytes(arr):
    return arr.nbytes

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class CodecManipulator(object):
    r"""
    **mm tokenizer v0.1**
    see codeclm/hf/mm_tokenizer_v0.1_hf/id2vocab.json

    text tokens: 
        llama tokenizer 0~31999
    
    special tokens: "32000": "<EOD>", "32001": "<SOA>", "32002": "<EOA>", "32003": "<SOI>", "32004": "<EOI>", "32005": "<SOV>", "32006": "<EOV>", "32007": "<s_local>", "32008": "<e_local>", "32009": "<s_global>", "32010": "<e_global>", "32011": "<semantic>", "32012": "<acoustic>", "32013": "<low_level>", "32014": "<dac_16k>", "32015": "<dac_44k>", "32016": "<xcodec>", "32017": "<placeholder>", "32018": "<semantic_mert>", "32019": "<semantic_hubert>", "32020": "<visual>", "32021": "<semanticodec>"

    mm tokens:
        dac_16k: 4 codebook, 1024 vocab, 32022 - 36117
        dac_44k: 9 codebook, 1024 vocab, 36118 - 45333
        xcodec: 12 codebook, 1024 vocab, 45334 - 57621
        semantic mert: 1024, 57622 - 58645
        semantic hubert: 512, 58646 - 59157
        visual: 64000, not included in v0.1
        semanticodec 100tps 16384: semantic=16384, 59158 - 75541, acoustic=8192, 75542 - 83733
    """
    def __init__(self, codec_type, quantizer_begin=None, n_quantizer=None, teacher_forcing=False, data_feature="codec"):
        self.codec_type = codec_type
        self.mm_v0_2_cfg = {
            "dac16k": {"codebook_size": 1024, "num_codebooks": 4, "global_offset": 32022, "sep": ["<dac_16k>"], "fps": 50},
            "dac44k": {"codebook_size": 1024, "num_codebooks": 9, "global_offset": 36118, "sep": ["<dac_44k>"]},
            "xcodec": {"codebook_size": 1024, "num_codebooks": 12, "global_offset": 45334, "sep": ["<xcodec>"], "fps": 50},
            "mert": {"codebook_size": 1024, "global_offset": 57622, "sep": ["<semantic_mert>"]},
            "hubert": {"codebook_size": 512, "global_offset": 58646, "sep": ["<semantic_hubert>"]},
            "semantic/s": {"codebook_size": 16384, "num_codebooks": 1, "global_offset": 59158, "sep": ["<semanticodec>", "<semantic>"]},
            "semantic/a": {"codebook_size": 8192, "num_codebooks": 1, "global_offset": 75542, "sep": ["<semanticodec>", "<acoustic>"]},
            "semanticodec": {"codebook_size": [16384, 8192], "num_codebooks": 2, "global_offset": 59158, "sep": ["<semanticodec>"], "fps": 50},
            "special_tokens": {
                '<EOD>': 32000, '<SOA>': 32001, '<EOA>': 32002, '<SOI>': 32003, '<EOI>': 32004, '<SOV>': 32005, '<EOV>': 32006, '<s_local>': 32007, '<e_local>': 32008, '<s_global>': 32009, '<e_global>': 32010, '<semantic>': 32011, '<acoustic>': 32012, '<stage_1>': 32013, '<dac_16k>': 32014, '<dac_44k>': 32015, '<xcodec>': 32016, '<stage_2>': 32017, '<semantic_mert>': 32018, '<semantic_hubert>': 32019, '<visual>': 32020, '<semanticodec>': 32021
            },
            "metadata": {
                "len": 83734,
                "text_range": [0, 31999],
                "special_range": [32000, 32021],
                "mm_range": [32022, 83733]
            },
            "codec_range": {
                "dac16k": [32022, 36117],
                "dac44k": [36118, 45333],
                "xcodec": [45334, 57621],
                # "hifi16k": [53526, 57621],
                "mert": [57622, 58645],
                "hubert": [58646, 59157],
                "semantic/s": [59158, 75541],
                "semantic/a": [75542, 83733],
                "semanticodec": [59158, 83733]
            }
        }
        self.sep = self.mm_v0_2_cfg[self.codec_type]["sep"]
        self.sep_ids = [self.mm_v0_2_cfg["special_tokens"][s] for s in self.sep]
        self.codebook_size = self.mm_v0_2_cfg[self.codec_type]["codebook_size"]
        self.num_codebooks = self.mm_v0_2_cfg[self.codec_type]["num_codebooks"]
        self.global_offset = self.mm_v0_2_cfg[self.codec_type]["global_offset"]
        self.fps = self.mm_v0_2_cfg[self.codec_type]["fps"] if "fps" in self.mm_v0_2_cfg[self.codec_type] else None

        self.quantizer_begin = quantizer_begin if quantizer_begin is not None else 0
        self.n_quantizer = n_quantizer if n_quantizer is not None else self.num_codebooks  
        self.teacher_forcing = teacher_forcing 
        self.data_feature = data_feature

    def tokenizer_sanity_check(self, tokenizer, version="v0.1"):
        tokenizer = _MMSentencePieceTokenizer(tokenizer)
        print(f"asserting tokenizer version {version}")
        ver = version.replace(".", "_")
        cfg = getattr(self, f"mm_{ver}_cfg")
        # check len
        assert len(tokenizer.tokenizer) == cfg["metadata"]["len"], f"len(tokenizer)={len(tokenizer.tokenizer)}, cfg_len={cfg['metadata']['len']}"
        # check special tokens
        for special_token, idx in cfg["special_tokens"].items():
            assert tokenizer.tokenizer.PieceToId(special_token) == idx, f"special_token={special_token}, idx={idx}, PieceToId={tokenizer.tokenizer.PieceToId(special_token)}"
        # check mm tokens
        mm_start, mm_end = cfg["metadata"]["mm_range"]
        for i in range(mm_start, mm_end+1):
            piece = tokenizer.tokenizer.IdToPiece(i)
            _piece = piece.replace("<", "").replace(">", "")
            mm_type, code = _piece.split("/")
            assert mm_type in cfg, f"mm_type={mm_type}"
            global_offset = cfg[mm_type]["global_offset"]
            num_codebooks = cfg[mm_type]["num_codebooks"]
            codebook_size = cfg[mm_type]["codebook_size"]

    def offset_tok_ids(self, x, global_offset=0, codebook_size=2048, num_codebooks=4):
        """
        x: (K, T)
        """
        if isinstance(codebook_size, int):
            assert x.max() < codebook_size, f"max(x)={x.max()}, codebook_size={codebook_size}"
        elif isinstance(codebook_size, list):
            for i, cs in enumerate(codebook_size):
                assert x[i].max() < cs, f"max(x)={x[i].max()}, codebook_size={cs}, layer_id={i}"
        else:
            raise ValueError(f"codebook_size={codebook_size}")
        assert x.min() >= 0, f"min(x)={x.min()}"
        assert x.shape[0] == num_codebooks or x.shape[0] == self.n_quantizer, \
            f"x.shape[0]={x.shape[0]}, num_codebooks={num_codebooks}, n_quantizer={self.n_quantizer}"

        _x = x.copy()
        _x = _x.astype(np.uint32)
        cum_offset = 0
        quantizer_begin = self.quantizer_begin
        quantizer_end = quantizer_begin+self.n_quantizer
        for k in range(self.quantizer_begin, quantizer_end): # k: quantizer_begin to quantizer_end - 1
            if isinstance(codebook_size, int):
                _x[k] += global_offset + k * codebook_size
            elif isinstance(codebook_size, list):
                _x[k] += global_offset + cum_offset
                cum_offset += codebook_size[k]
            else:
                raise ValueError(f"codebook_size={codebook_size}")
        return _x[quantizer_begin:quantizer_end]

    def unoffset_tok_ids(self, x, global_offset=0, codebook_size=2048, num_codebooks=4):
        """
        x: (K, T)
        """
        if isinstance(codebook_size, int):
            assert x.max() < global_offset + codebook_size * num_codebooks, f"max(x)={x.max()}, codebook_size={codebook_size}"
        elif isinstance(codebook_size, list):
            assert x.max() < global_offset + sum(codebook_size), f"max(x)={x.max()}, codebook_size={codebook_size}"
        assert x.min() >= global_offset, f"min(x)={x.min()}, global_offset={global_offset}"
        assert x.shape[0] == num_codebooks or x.shape[0] == self.n_quantizer, \
            f"x.shape[0]={x.shape[0]}, num_codebooks={num_codebooks}, n_quantizer={self.n_quantizer}"
        
        _x = x.copy()
        _x = _x.astype(np.uint32)
        cum_offset = 0
        quantizer_begin = self.quantizer_begin
        quantizer_end = quantizer_begin+self.n_quantizer
        for k in range(quantizer_begin, quantizer_end):
            if isinstance(codebook_size, int):
                _x[k-quantizer_begin] -= global_offset + k * codebook_size
            elif isinstance(codebook_size, list):
                _x[k-quantizer_begin] -= global_offset + cum_offset
                cum_offset += codebook_size[k]
            else:
                raise ValueError(f"codebook_size={codebook_size}")
        return _x

    def flatten(self, x):
        if len(x.shape) > 2:
            x = x.squeeze()
        assert x.shape[0] == self.num_codebooks or x.shape[0] == self.n_quantizer, \
            f"x.shape[0]={x.shape[0]}, num_codebooks={self.num_codebooks}, n_quantizer={self.n_quantizer}"
        return einops.rearrange(x, 'K T -> (T K)')

    def unflatten(self, x, n_quantizer=None):
        x = x.squeeze()
        assert len(x.shape) == 1
        assert x.shape[0] % self.num_codebooks == 0 or x.shape[0] % self.n_quantizer == 0, \
            f"x.shape[0]={x.shape[0]}, num_codebooks={self.num_codebooks}, n_quantizer={self.n_quantizer}"
        if n_quantizer!=self.num_codebooks:
            return einops.rearrange(x, '(T K) -> K T', K=n_quantizer)
        return einops.rearrange(x, '(T K) -> K T', K=self.num_codebooks)
    
    def get_codec_type_from_range(self, ids):
        ids_range = [ids.min(), ids.max()]
        codec_range = self.mm_v0_2_cfg["codec_range"]
        for codec_type, r in codec_range.items():
            if ids_range[0] >= r[0] and ids_range[1] <= r[1]:
                return codec_type
        raise ValueError(f"ids_range={ids_range}, codec_range={codec_range}")

    def npy2ids(self, npy):
        if isinstance(npy, str):
            data = np.load(npy)
        elif isinstance(npy, np.ndarray):
            data = npy
        else:
            raise ValueError(f"not supported type: {type(npy)}")

        assert len(data.shape)==2,  f'data shape: {data.shape} is not (n_codebook, seq_len)'
        data = self.offset_tok_ids(
            data, 
            global_offset=self.global_offset, 
            codebook_size=self.codebook_size, 
            num_codebooks=self.num_codebooks, 
        )
        data = self.flatten(data)
        codec_range = self.get_codec_type_from_range(data)
        assert codec_range == self.codec_type, f"get_codec_type_from_range(data)={codec_range}, self.codec_type={self.codec_type}"
        data = data.tolist()
        return data
    
    def ids2npy(self, token_ids):
        # make sure token_ids starts with codebook 0
        if isinstance(self.codebook_size, int):
            codebook_0_range = (self.global_offset + self.quantizer_begin*self.codebook_size, self.global_offset + (self.quantizer_begin+1)*self.codebook_size)
        elif isinstance(self.codebook_size, list):
            codebook_0_range = (self.global_offset, self.global_offset + self.codebook_size[0])
        assert token_ids[0] >= codebook_0_range[0] \
            and token_ids[0] < codebook_0_range[1], f"token_ids[0]={token_ids[self.quantizer_begin]}, codebook_0_range={codebook_0_range}"
        data = np.array(token_ids)
        data = self.unflatten(data, n_quantizer=self.n_quantizer)
        data = self.unoffset_tok_ids(
            data, 
            global_offset=self.global_offset, 
            codebook_size=self.codebook_size, 
            num_codebooks=self.num_codebooks, 
        )
        return data

    def npy_to_json_str(self, npy_path):
        data = self.npy2ids(npy_path)
        return json.dumps({"text": data, "src": npy_path, "codec": self.codec_type})
    
    def sep(self):
        return ''.join(self.sep)
    
    def sep_ids(self):
        return self.sep_ids

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        if 'audio_prompt_len' in self.args and self.args.audio_prompt_len > 0:
            if self.args.audio_prompt_len >= 30:
                Encoder.x_values = np.linspace(20, 40, 100)
            elif self.args.audio_prompt_len > 0:    
                Encoder.x_values = np.linspace(1, 5, 100)
            else:
                raise ValueError(f"not support {self.args.audio_prompt_len} in length")
            # Define the Gaussian probability density function with mean=30 and variance=3
            pdf_values = norm.pdf(Encoder.x_values, loc=30, scale=np.sqrt(3))
            # norm PDF，make it sum to 1
            pdf_normalized = pdf_values / pdf_values.sum()
            Encoder.cdf_values = np.cumsum(pdf_normalized)

        Encoder.tokenizer = _MMSentencePieceTokenizer(self.args.tokenizer_model, vocab_extra_ids=self.args.vocab_extra_ids)
        Encoder.codectool = CodecManipulator(self.args.codec_type, self.args.quantizer_begin, self.args.n_quantizer, self.args.teacher_forcing, self.args.data_feature)
        print(f'Initial codecmanipulator with codec_type={self.args.codec_type}, quantizer_begin={self.args.quantizer_begin}, n_quantizer={self.args.n_quantizer}')
        print(f'Using teacher_forcing: {self.args.teacher_forcing}')
        print(f'Processing data_feature: {self.args.data_feature}')
        # TODO: finish sanity check
        # self.codectool.tokenizer_sanity_check(Encoder.tokenizer, version="v0.1")
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            if key == 'codec' and "text" in self.args.json_keys:
                print("[Warning] codec will be merged after text...")
                continue
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            if key == 'codec' and "text" in self.args.json_keys:
                print("[Warning] codec will be merged after text...")
                continue
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)

    def parse_line(self, line):
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print("one line is not a valid json, skipping...")
            return None
        return data

    def encode_no_tokenizer(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            if key == 'codec' and "text" in self.args.json_keys:
                print("[Warning] codec will be merged after text...")
                continue
            doc_ids = data[key]
            assert isinstance(doc_ids, list), "input token_ids must be a list"
            sentence_lens = []
            if len(doc_ids) > 0 and self.args.append_eod:
                sentence_lens.append(len(doc_ids))
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)
    
    def encode_mix_text_and_codec(self, json_line):      
        data = json.loads(json_line)
        # assert text and codec are in the json
        assert 'text' in data and 'codec' in data, "`text` and `codec` must be in the json key"

        ids = {}
        lens = {}

        # tokenize the text
        instruction = self.args.instruction
        text = instruction + '\n' + data['text']

        if self.args.instruction_dropout_rate > 0.0:
            # randomly drop the instruction
            if np.random.rand() < self.args.instruction_dropout_rate:
                text = data['text']

        if self.args.to_lower:
            text = text.lower()
            
        text_ids = Encoder.tokenizer.tokenize(text)

        # read codec npy
        codec_path = data['codec']
        # codec_path = map_path(codec_path)
        codec_ids = [Encoder.tokenizer.soa] + Encoder.codectool.sep_ids + Encoder.codectool.npy2ids(codec_path) + [Encoder.tokenizer.eoa]
        
        if self.args.order == "textfirst":
            doc_ids = text_ids + codec_ids
        else:
            doc_ids = codec_ids + text_ids

        sentence_lens = []
        sentence_lens.append(len(doc_ids))
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids.append(Encoder.tokenizer.eod)
            sentence_lens[-1] += 1
        
        key = "text" # hardcode key
        ids[key] = doc_ids
        lens[key] = sentence_lens

        return ids, lens, len(text) + 4 * len(codec_ids)
        
    def encode_token_level_interleave(self, json_line):
        data = json.loads(json_line)
        # assert text and codec are in the json
        # assert 'text' in data and 'codec' in data, "`text` and `codec` must be in the json key"

        ids = {}
        lens = {}

        # tokenize the text
        instruction = self.args.instruction
        text = instruction + '\n' + data['text']

        if self.args.instruction_dropout_rate > 0.0:
            # randomly drop the instruction
            if np.random.rand() < self.args.instruction_dropout_rate:
                text = data['text']

        if self.args.to_lower:
            text = text.lower()
            
        text_ids = Encoder.tokenizer.tokenize(text)

        # read codec npy
        # codec_path = data['codec']
        raw_codec_vocals = np.load(data['vocals_codec'])
        raw_codec_instrumental = np.load(data['instrumental_codec'])
        if raw_codec_vocals.shape != raw_codec_instrumental.shape:
            if raw_codec_vocals.shape[-1] - raw_codec_instrumental.shape[-1] <= 10:
                min_len = min(raw_codec_vocals.shape[-1], raw_codec_instrumental.shape[-1])
                raw_codec_vocals = raw_codec_vocals[:, :min_len]
                raw_codec_instrumental = raw_codec_instrumental[:, :min_len]
            else:
                raise AssertionError(f'mismatch shape of vocal codec and instrumental codec {data["id"]}')
        
        vocals_ids = Encoder.codectool.npy2ids(raw_codec_vocals)
        instrumental_ids = Encoder.codectool.npy2ids(raw_codec_instrumental)
        ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')

        codec_ids = [Encoder.tokenizer.soa] + Encoder.codectool.sep_ids + list(ids_segment_interleaved) + [Encoder.tokenizer.eoa]
        
        if self.args.order == "textfirst":
            doc_ids = text_ids + codec_ids
        else:
            doc_ids = codec_ids + text_ids

        sentence_lens = []
        sentence_lens.append(len(doc_ids))
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids.append(Encoder.tokenizer.eod)
            sentence_lens[-1] += 1
        
        key = "text" # hardcode key
        ids[key] = doc_ids
        lens[key] = sentence_lens

        return ids, lens, len(text) + 4 * len(codec_ids)


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()


    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Processing", input_file_name)
        
        fin = open(input_file_name, 'r', encoding='utf-8')

        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = _MMSentencePieceTokenizer(self.args.tokenizer_model, vocab_extra_ids=self.args.vocab_extra_ids)

        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        # encoded_docs = pool.imap(encoder.encode, fin, 32)
        # encoded_docs = pool.imap(encoder.encode_no_tokenizer, fin, 32)
        if self.args.use_token_level_interleave:
            encoded_docs = pool.imap(encoder.encode_token_level_interleave, fin, 32)
        else:
            encoded_docs = pool.imap(encoder.encode_mix_text_and_codec, fin,  32)
        # # DEBUG：
        # encoded_docs = []
        # encoder.initializer()
        # lines = fin.readlines()
        # for i in range(10):
        #     encoded_docs.append(encoder.encode_mix_text_and_codec(lines[i], quantizer_begin, n_quantizer))
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            if key == 'codec' and "text" in self.args.json_keys:
                print("[Warning] codec will be merged after text...")
                continue
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        builders[key].finalize(output_idx_files[key])


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'Llama2Tokenizer',
                                'NullTokenizer', 'MMSentencePieceTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--codec-type', type=str, required=True,
                       choices=['dac16k', 'dac44k', 'xcodec', 'mert', 'hubert', 'semantic/s', 'semantic/a', 'semanticodec'],)
    group.add_argument('--instruction', type=str, default="Generate audio from the given text condition.",
                          help='A text instruction to prepend to before text and codec. e.g. `Generate audio from the given text condition.`')
    group.add_argument('--order', type=str, required=True,
                          choices=['textfirst', 'audiofirst'],
                       help='For text2audio, should enable textfirst, for audio2text, should disable textfirst.')
    group.add_argument('--use-token-level-interleave', action='store_true',
                       help='use token level interleave')
    group.add_argument('--instruction-dropout-rate', type=float, default=0.0,
                       help='Dropout rate for the instruction.')
    group.add_argument('--tokenizer-model', type=str, required=True,
                       help='YTTM tokenizer model.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--to-lower', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    group.add_argument('--quantizer-begin', type=int, default=None,
                       help='index of quantizer begining')
    group.add_argument('--n-quantizer', type=int, default=None,
                       help='number of quantizers to extract codes')
    group.add_argument('--teacher-forcing', action='store_true', default=False,
                       help='whether to use teacher forcing in stage 2')
    group.add_argument('--data-feature', type=str, default='codec',
                       help='which feature to process')     
    group.add_argument('--audio-prompt-len', 
                       type=int,
                       default=-1,
                       help='length of audio prompt in-context learning')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True

def main():
    args = get_args()
    print(f"[INFO] instruction dropout rate is set to: {args.instruction_dropout_rate}")

    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    in_ss_out_names = []
    if args.partitions == 1:
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            'partition': args.input,
            'sentence_split': sentence_split_file,
            'output_prefix': args.output_prefix}
        in_ss_out_names.append(file_names)
    else:
        in_file_names = glob.glob(args.input)

        # Count total number of lines across .jsonl files
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += (fc + 1)
            partition_size = math.ceil(total_sample_count / args.partitions)

        # create .jsonl parition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # check to see if paritions were already created
        partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)

        # check to see if paritions with split sentences already created
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

        if not partitions_present and not split_sentences_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            if args.keep_sequential_samples: line_count = 0
            for in_file_name in in_file_names:
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8')

                for line in fin:
                    partitioned_input_files[index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1)%args.partitions

                fin.close()

            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers//args.partitions)

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    # split sentences in partition files
    if args.split_sentences and not split_sentences_present:
        processes = []
        for name in in_ss_out_names:
            p = multiprocessing.Process(target=partition.split_sentences,
                                        args=((name['partition'], name['sentence_split']),))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if args.partitions == 1:
            return


    # encode partition files in parallel
    processes = []
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    for name in in_ss_out_names:
        p = multiprocessing.Process(target=partition.process_json_file,
                                    args=((name[input_key], name['output_prefix']),))
        p.start()
        processes.append(p)
        # partition.process_json_file((name[input_key], name['output_prefix']), quantizer_begin=args.quantizer_begin, n_quantizer=args.n_quantizer)

    for p in processes:
        p.join()

    if args.partitions == 1:
        return

    # merge bin/idx partitions
    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer = _MMSentencePieceTokenizer(args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids)

    for key in args.json_keys:
        if key == 'codec' and "text" in args.json_keys:
            print("[Warning] codec will be merged after text...")
            continue

        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            parition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                             key, level)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':

    main()

