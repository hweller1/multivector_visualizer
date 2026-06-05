#!/usr/bin/env python3
import json
import argparse
import os

ENGINE_ORDER = ["hnsw", "colbert", "plaid", "warp", "tachiom"]
COLORS = {"hnsw": "#1f77b4", "colbert": "#ff7f0e", "plaid": "#2ca02c", "warp": "#d62728", "tachiom": "#9467bd"}

def load(path):
    with open(path) as f:
        return json.load(f)

def pareto_plot(results, out):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for r in results:
            ax.scatter(r["p99_ms"], r["recall_at_10"], label=r["engine"], color=COLORS.get(r["engine"], "gray"))
        ax.set_xscale("log")
        ax.set_xlabel("p99 latency (ms)")
        ax.set_ylabel("Recall@10")
        ax.set_title("Recall vs Latency Pareto")
        ax.legend()
        fig.savefig(out)
        plt.close(fig)
    except ImportError:
        print(f"matplotlib not available, skipping {out}")

def recall_k_plot(results, out):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for engine in ENGINE_ORDER:
            r_list = [r for r in results if r["engine"] == engine]
            if r_list:
                r = r_list[0]
                ks = [1, 10, 100]
                recalls = [r.get("recall_at_1", 0), r.get("recall_at_10", 0), r.get("recall_at_100", 0)]
                ax.plot(ks, recalls, label=engine, color=COLORS.get(engine, "gray"), marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("Recall@k")
        ax.set_title("Recall@k by Engine")
        ax.legend()
        fig.savefig(out)
        plt.close(fig)
    except ImportError:
        print(f"matplotlib not available, skipping {out}")

def latency_cdf_plot(results, out):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for r in results:
            percentiles = [r.get("p50_ms", 0), r.get("p95_ms", 0), r.get("p99_ms", 0)]
            probs = [0.5, 0.95, 0.99]
            ax.plot(percentiles, probs, label=r["engine"], color=COLORS.get(r["engine"], "gray"), marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("Latency CDF by Engine")
        ax.legend()
        fig.savefig(out)
        plt.close(fig)
    except ImportError:
        print(f"matplotlib not available, skipping {out}")

def qps_bar_plot(results, out):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        engines = [r["engine"] for r in results]
        qps = [r.get("qps", 0) for r in results]
        ax.bar(engines, qps, color=[COLORS.get(e, "gray") for e in engines])
        ax.set_xlabel("Engine")
        ax.set_ylabel("QPS")
        ax.set_title("Throughput by Engine")
        fig.savefig(out)
        plt.close(fig)
    except ImportError:
        print(f"matplotlib not available, skipping {out}")

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", default="output/bench_results.json")
    args = parser.parse_args()
    os.makedirs("output", exist_ok=True)
    results = load(args.input) if os.path.exists(args.input) else []
    pareto_plot(results, "output/pareto.png")
    recall_k_plot(results, "output/recall_k.png")
    latency_cdf_plot(results, "output/latency_cdf.png")
    qps_bar_plot(results, "output/qps_bar.png")
    print("Plots saved to output/")

if __name__ == "__main__":
    main()
