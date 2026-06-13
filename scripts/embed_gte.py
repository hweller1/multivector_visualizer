#!/usr/bin/env python3
"""
Embed MS MARCO passages with GTE-ModernColBERT-v1 (local MPS inference).

Output layout under --out-dir (same binary format as the Jina path):
  offsets.bin  — u64[N] byte offsets into data.bin
  lengths.bin  — u32[N] token counts per passage (after pooling)
  data.bin     — f32[Σtokens × 128] token embeddings
  meta.json

Resume-safe: derives progress from existing file sizes. Rerun to continue.

Usage:
  .venv/bin/python3 scripts/embed_gte.py data/msmarco/collection.tsv \\
      --limit 5000000 --pool-factor 2
"""
import argparse
import json
import os
import struct
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("collection_tsv")
    p.add_argument("--out-dir", default="data/msmarco/gte")
    p.add_argument("--limit", type=int, default=5_000_000)
    p.add_argument("--batch-size", type=int, default=256,
                   help="passages per encode call (internal pylate batch_size=64)")
    p.add_argument("--pool-factor", type=int, default=2)
    p.add_argument("--device", default="mps")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    offsets_path = os.path.join(args.out_dir, "offsets.bin")
    data_path    = os.path.join(args.out_dir, "data.bin")
    lengths_path = os.path.join(args.out_dir, "lengths.bin")

    done        = os.path.getsize(offsets_path) // 8 if os.path.exists(offsets_path) else 0
    data_offset = os.path.getsize(data_path)         if os.path.exists(data_path)    else 0

    print(f"  GTE-ModernColBERT-v1 | device={args.device} | pool_factor={args.pool_factor}")
    print(f"  Target: {args.limit:,} | Resume from: {done:,}")

    try:
        from pylate import models as pylate_models
    except ImportError:
        print("Error: pylate not installed. Run: .venv/bin/pip install pylate")
        sys.exit(1)

    print("  Loading model…", flush=True)
    model = pylate_models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device=args.device,
        document_length=300,
    )
    print("  Model ready.\n")

    data_f = open(data_path,    "ab")
    off_f  = open(offsets_path, "ab")
    len_f  = open(lengths_path, "ab")

    batch   = []
    count   = done
    t_start = time.time()

    with open(args.collection_tsv, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < done:
                continue
            if count >= args.limit:
                break
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                batch.append(parts[1])

            if len(batch) >= args.batch_size:
                data_offset = _flush(model, batch, data_f, off_f, len_f,
                                     data_offset, args.pool_factor)
                count += len(batch)
                batch = []
                _print_progress(count, args.limit, t_start)

    if batch:
        data_offset = _flush(model, batch, data_f, off_f, len_f,
                             data_offset, args.pool_factor)
        count += len(batch)

    data_f.close()
    off_f.close()
    len_f.close()

    meta = {
        "count":       count,
        "dim":         128,
        "model":       "lightonai/GTE-ModernColBERT-v1",
        "pool_factor": args.pool_factor,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n  Done. {count:,} passages in {elapsed/3600:.1f}h → {args.out_dir}/")


def _flush(model, texts, data_f, off_f, len_f, data_offset, pool_factor):
    matrices = model.encode(
        sentences=texts,
        batch_size=64,
        is_query=False,
        pool_factor=pool_factor,
        show_progress_bar=False,
    )
    for mat in matrices:
        n_tokens = mat.shape[0]
        off_f.write(struct.pack("<Q", data_offset))
        len_f.write(struct.pack("<I", n_tokens))
        raw = mat.astype("float32").tobytes()
        data_f.write(raw)
        data_offset += len(raw)
    return data_offset


def _print_progress(count, total, t_start):
    elapsed = time.time() - t_start
    rate    = count / elapsed if elapsed > 0 else 0
    eta_h   = (total - count) / rate / 3600 if rate > 0 else float("inf")
    pct     = count / total * 100
    print(f"\r  {pct:5.1f}%  {count:,}/{total:,}  {rate:,.0f} p/s  ETA {eta_h:.1f}h",
          end="", flush=True)


if __name__ == "__main__":
    main()
