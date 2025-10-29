# src/parallel/word_count_mpi.py
from __future__ import annotations

import sys, os



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from collections import Counter
from mpi4py import MPI
from utils.dataSetGenrator import DatasetGenerator

from utils.logger import log_header, log_info, log_success, log_warning
from utils.text_preprocess import clean_and_tokenize

def chunk_indices(n: int, size: int, rank: int) -> tuple[int, int]:
    """Return [start, end) indices for this rank given n items (balanced split)."""
    base = n // size
    rem = n % size
    start = rank * base + min(rank, rem)
    length = base + (1 if rank < rem else 0)
    return start, start + length

def main(dataset_dir: str, top_k: int = 10) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        #log_header(f"MPI Word Count — Dataset: {dataset_dir}  (P={size})")
        if not os.path.isdir(dataset_dir):
            #log_warning(f"Dataset folder not found: {dataset_dir}")
            file_list = []
        else:
            file_list = sorted(
                [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(".txt")]
            )
            #log_info(f"Found {len(file_list)} .txt files")
    else:
        file_list = None

    # Broadcast file list to all ranks
    file_list = comm.bcast(file_list, root=0)
    n_files = len(file_list)

    # Early exit if nothing to do
    if n_files == 0:
        if rank == 0:
            #("No files to process. Exiting.")
            print("No files found")
        return

    # Compute slice for this rank
    s, e = chunk_indices(n_files, size, rank)
    my_files = file_list[s:e]

    # Local compute
    local_counter = Counter()
    local_token_count = 0

    t0 = MPI.Wtime()
    for fp in my_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = clean_and_tokenize(text)
            local_counter.update(tokens)
            local_token_count += len(tokens)
        except Exception as ex:
            # Keep going, just warn locally (avoid noisy cross-rank logs)
            if rank == 0:
                #log_warning(f"Skipping {fp}: {ex}")
                print("Exception: {}".format(ex))

    local_elapsed = MPI.Wtime() - t0

    # Gather counters at root
    all_counters = comm.gather(local_counter, root=0)
    # Reduce totals
    total_tokens = comm.reduce(local_token_count, op=MPI.SUM, root=0)
    total_files  = comm.reduce(len(my_files),     op=MPI.SUM, root=0)
    max_local_t  = comm.reduce(local_elapsed,     op=MPI.MAX, root=0)  # rough bound on wall time
    sum_local_t  = comm.reduce(local_elapsed,     op=MPI.SUM, root=0)  # for average diagnostics

    if rank == 0:
        # Merge all counters
        global_counter = Counter()
        for c in all_counters:
            global_counter.update(c)

        #log_success(f"Parallel word count complete — Files: {total_files}, Tokens: {total_tokens}")
        #log_info(f"Wall-time (approx, max over ranks): {max_local_t:.4f}s | Avg per-rank time: {sum_local_t/size:.4f}s")

        # Print top-K
        #log_info(f"Top {top_k} frequent words:")
        for w, cnt in global_counter.most_common(top_k):
            print(f"  {w}: {cnt}")

if __name__ == "__main__":
    DatasetGenerator.create_datasets()
    parser = argparse.ArgumentParser(description="MPI Word Count (Parallel)")
    parser.add_argument("--dataset", type=str, default="datasets/test/medium",
                        help="Path to folder containing .txt files")
    parser.add_argument("--topk", type=int, default=10, help="How many top words to print")
    args = parser.parse_args()
    main(args.dataset, top_k=args.topk)
