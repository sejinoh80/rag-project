import pandas as pd
df = pd.read_csv("summary.csv")

start = df["launch_time"].min()
end   = df["finish_time"].max()
total_time = end - start

qps = len(df) / total_time
avg_prompt_tps = df["prompt_tokens"].sum() / total_time
avg_gen_tps = df["generation_tokens"].sum() / total_time

out = pd.DataFrame([{
    "time_sec": total_time,
    "requests": len(df),
    "QPS": qps,
    "avg_prompt_throughput_tokens_per_s": avg_prompt_tps,
    "avg_generation_throughput_tokens_per_s": avg_gen_tps,
    "avg_TTFT_s": df["ttft"].mean(),
}])
out.to_csv("perf_summary.csv", index=False)
print(out)