"""
Aggregate a metric from './results.json' over multiple runs.

Usage:
  python collect_results.py --key "Final/Test/mae" --row model --column dataset

Output:
  Table of mean ± std for the key, grouped by row × column values.
  Skips entries with empty/missing 'eval'.

Example:
model\dataset                       QM9
GNN                   1.1500e-02 ± 5.0e-04
"""


import json
import numpy as np
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Metric to extract, e.g. 'Final/Test/mae'")
    parser.add_argument("--row", required=True, help="Field to group by as rows")
    parser.add_argument("--column", required=True, help="Field to group by as columns")
    return parser.parse_args()

def main():
    args = parse_args()

    with open("./results.json", "r") as f:
        data = json.load(f)

    table = defaultdict(lambda: defaultdict(list))

    for exp in data:
        eval_runs = exp.get("eval")
        if not eval_runs:
            continue  # skip if eval is missing or empty

        for run in eval_runs:
            row_val = run.get(args.row)
            col_val = run.get(args.column)
            key_val = run.get(args.key)
            if row_val is not None and col_val is not None and key_val is not None:
                table[row_val][col_val].append(key_val)

    row_keys = sorted(table.keys())
    col_keys = sorted({col for row in table.values() for col in row.keys()})

    header = f"{args.row}\\{args.column}".ljust(20) + "".join(f"{col:>30}" for col in col_keys)
    print(header)

    for row in row_keys:
        row_str = f"{str(row):<20}"
        for col in col_keys:
            vals = table[row].get(col, [])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                row_str += f"{mean:.4e} ± {std:.1e}".rjust(30)
            else:
                row_str += "N/A".rjust(30)
        print(row_str)

if __name__ == "__main__":
    main()
