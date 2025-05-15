
"""
# you can run the following command to make DB2TOKCNT readable
autopep8 --in-place --aggressive --aggressive finetune/scripts/parse_mixture.py

This script is used to parse the mixture of the pretraining data
input: path to the yaml file
output: a megatron style data mixture string
"""

import os
import sys
import argparse
import yaml
import re


EXAMPLE_LOG_STRING = """Zarr-based strategies will not be registered because of missing packages
Counting tokens in  ./mmap/example.bin

  0%|          | 0/597667 [00:00<?, ?it/s]
 14%|█▍        | 83737/597667 [00:00<00:00, 837344.85it/s]
 30%|██▉       | 178908/597667 [00:00<00:00, 904601.65it/s]
 45%|████▌     | 269369/597667 [00:00<00:00, 883202.85it/s]
 60%|█████▉    | 357748/597667 [00:00<00:00, 841687.26it/s]
 74%|███████▍  | 442183/597667 [00:00<00:00, 837274.61it/s]
 88%|████████▊ | 528884/597667 [00:00<00:00, 847062.60it/s]
100%|██████████| 597667/597667 [00:00<00:00, 850432.85it/s]
Total number of tokens:  806001459
"""

global DB2TOKCNT
DB2TOKCNT = {}

def get_count_logs_paths(logs_dir, pattern='count.*.log'):
    return [
        os.path.join(
            logs_dir,
            f) for f in os.listdir(logs_dir) if re.match(
            pattern,
            f)]


def get_tokcnt_from_log(log_path, by_billions=True):
    """
    input: path to the log file
    output: Tuple of (path, token_count)
    """
    print(f"[INFO] Checking token count log from {log_path}")
    match_path_pattern = r'Counting tokens in\s+(.*)'
    match_tokcnt_pattern = r'Total number of tokens:\s+(\d+)'

    with open(log_path, 'r') as f:
        log = f.read()
        path = re.search(match_path_pattern, log).group(1)
        tokcnt = int(re.search(match_tokcnt_pattern, log).group(1))
        if by_billions:
            tokcnt = tokcnt / 1e9
            # into string x.xxxB
            tokcnt = f"{tokcnt:.3f}B"
        return (path, tokcnt)


def get_tokcnts_from_logs(logs_dir, by_billions=True):

    logs = get_count_logs_paths(logs_dir)
    for log in logs:
        db, tokcnt = get_tokcnt_from_log(log, by_billions)
        DB2TOKCNT[db] = tokcnt


def parse_args():
    parser = argparse.ArgumentParser(
        description="parse the mixture of the pretraining data")
    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        required=True,
        help="path to the yaml file")
    parser.add_argument(
        "--reload-db2tokcnt",
        "-r",
        action="store_true",
        help="DB2TOKCNT is currently hardcoded, reload it from the TOKEN_COUNT_LOG_DIR"
    )
    parser.add_argument(
        "--by-billions",
        "-b",
        action="store_true",
        help="output the tokcnt by billions")
    return parser.parse_args()


def load_yaml(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def parse_mixture_from_cfg_deprecated(cfg):
    keys = list(cfg.keys())
    # find keys ends with _ROUND
    rounds = [k for k in keys if k.endswith("_ROUND")]

    def repeat_str(s, n):
        return "".join([s for _ in range(n)])

    total_tokcnt = 0
    mixture_str = ""
    for r in rounds:
        repeat_times = float(r.replace("_ROUND", ""))
        mmap_paths = sorted(set(cfg[r]))
        for mmap_path in mmap_paths:
            mmap_path_without_ext = os.path.splitext(mmap_path)[0]
            if repeat_times >= 1:
                mixture_str += repeat_str(
                    f"1 {mmap_path_without_ext} ", int(repeat_times))
            else:
                # weight is less than 1
                mixture_str += f"{repeat_times} {mmap_path_without_ext} "
            tokcnt = DB2TOKCNT[mmap_path]
            if isinstance(tokcnt, str):
                assert tokcnt.endswith("B"), f"invalid tokcnt: {tokcnt}"
                tokcnt = float(tokcnt.replace("B", "")) * 10**9
                total_tokcnt += tokcnt * repeat_times
            else:
                assert isinstance(tokcnt, int), f"invalid tokcnt: {tokcnt}"
                total_tokcnt += tokcnt * repeat_times

    # total iter count
    total_iter = total_tokcnt / (cfg["GLOBAL_BATCH_SIZE"] * cfg["SEQ_LEN"])

    # into string x.xxxB
    total_tokcnt /= 1e9
    total_tokcnt = f"{total_tokcnt:.3f}B"

    return mixture_str, total_tokcnt, total_iter


def parse_mixture_from_cfg(cfg):
    keys = list(cfg.keys())
    # find keys ends with _ROUND
    rounds = [k for k in keys if k.endswith("_ROUND")]

    def repeat_str(s, n):
        return "".join([s for _ in range(n)])

    total_tokcnt = 0
    mixture_str = ""
    for r in rounds:
        repeat_times = float(r.replace("_ROUND", ""))
        mmap_paths = sorted(set(cfg[r]))
        for mmap_path in mmap_paths:
            mmap_path_without_ext = os.path.splitext(mmap_path)[0]
            tokcnt = DB2TOKCNT[mmap_path]
            if isinstance(tokcnt, str):
                assert tokcnt.endswith("B"), f"invalid tokcnt: {tokcnt}"
                tokcnt = float(tokcnt.replace("B", "")) * 10**9
                total_tokcnt += tokcnt * repeat_times
            else:
                assert isinstance(tokcnt, int), f"invalid tokcnt: {tokcnt}"
                total_tokcnt += tokcnt * repeat_times

            mixture_str += f"{int(tokcnt * repeat_times)} {mmap_path_without_ext} "

    # total iter count
    total_iter = total_tokcnt / (cfg["GLOBAL_BATCH_SIZE"] * cfg["SEQ_LEN"])

    # into string x.xxxB
    total_tokcnt /= 1e9
    total_tokcnt = f"{total_tokcnt:.3f}B"

    return mixture_str, total_tokcnt, total_iter


if __name__ == "__main__":
    
    args = parse_args()

    cfg = load_yaml(args.cfg)
    print(f"[INFO] Loaded cfg from {args.cfg}")

    TOKEN_COUNT_LOG_DIR = cfg["TOKEN_COUNT_LOG_DIR"]
    print(f"[INFO] TOKEN_COUNT_LOG_DIR: {TOKEN_COUNT_LOG_DIR}")

    get_tokcnts_from_logs(TOKEN_COUNT_LOG_DIR,
            by_billions=args.by_billions)
    print(f"[INFO] DB2TOKCNT reloaded from the logs in {TOKEN_COUNT_LOG_DIR}\n")

    mixture_str, total_tokcnt, total_iter = parse_mixture_from_cfg(cfg)
    print(f"[CRITICAL] DATA_PATH **(copy to the training script)**:\n{mixture_str}\n")
    print(f"[CRITICAL] TRAIN_ITERS **(copy to the training script)**:\n{total_iter}\n")
    print(f"[INFO] Total token count: {total_tokcnt}")
