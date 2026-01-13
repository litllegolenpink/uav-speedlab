# scripts/summarize_perf.py
import pandas as pd
import argparse

p = argparse.ArgumentParser()
p.add_argument("--csv", type=str, default="perf_log_with_model.csv")
p.add_argument("--skip", type=int, default=5, help="skip warmup frames")
args = p.parse_args()

df = pd.read_csv(args.csv)
df2 = df[df["frame"] >= args.skip].copy()

def q(x, p):
    return float(x.quantile(p))

print(f"rows={len(df)}  used={len(df2)}  skip={args.skip}")

for col in ["infer_ms", "cpu_percent", "ram_mb"]:
    s = df2[col]
    print(
        f"{col}: mean={s.mean():.2f}  p50={q(s,0.5):.2f}  p95={q(s,0.95):.2f}  max={s.max():.2f}"
    )
