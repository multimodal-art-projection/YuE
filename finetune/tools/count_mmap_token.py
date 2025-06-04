import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from core.datasets.indexed_dataset import MMapIndexedDataset
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmap_path", type=str, required=True, help="Path to the .bin mmap file")
    return parser.parse_args()


args = get_args()

slice_path = args.mmap_path
if slice_path.endswith(".bin"):
    slice_path = slice_path[:-4]

dataset = MMapIndexedDataset(slice_path)


def count_ids(dataset):
    count = 0
    for doc_ids in tqdm(dataset):
        count += doc_ids.shape[0]
    return count

print("Counting tokens in ", args.mmap_path)
total_cnt = count_ids(dataset)
print("Total number of tokens: ", total_cnt)
