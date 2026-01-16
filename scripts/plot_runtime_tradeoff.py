from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str, help="Path to runtime_tradeoff.csv")
    ap.add_argument("--out", type=str, default="runtime_tradeoff.png")
    args = ap.parse_args()

    xs, ys = [], []
    with Path(args.csv_path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["sec_per_gsm"]))
            ys.append(float(row["n_entities"]))

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Seconds per GSM")
    plt.ylabel("Total entities extracted+linked")
    plt.title("Runtime vs yield (entities) across retrieval/rerank settings")
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote plot: {args.out}")


if __name__ == "__main__":
    main()
