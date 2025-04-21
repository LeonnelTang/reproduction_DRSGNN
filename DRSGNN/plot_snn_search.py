import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# import seaborn as sns
import numpy as np

log_file = Path("./DRSGNN/tmpdir/snn/snn_search.log")

# 提取日志中每条记录的参数和结果
pattern = re.compile(
    r"learning_rate=([\d.]+), T=(\d+), max_test_accuracy=([\d.]+)"
)

results = []

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            lr, T, acc = match.groups()
            results.append({
                "learning_rate": float(lr),
                "T": int(T),
                "accuracy": float(acc)
            })

# 构建DataFrame
df = pd.DataFrame(results)

# 可视化：不同T下，准确率随learning_rate变化的曲线图
plt.figure(figsize=(12, 6))
for T_val in sorted(df["T"].unique()):
    subset = df[df["T"] == T_val]
    grouped = subset.groupby("learning_rate")["accuracy"].mean().reset_index()
    plt.plot(grouped["learning_rate"], grouped["accuracy"], label=f"T={T_val}")

plt.xlabel("Learning Rate")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Learning Rate (by T)")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("acc_vs_lr_by_T.png")


# 可视化：不同learning_rate下，准确率随T变化的曲线图
plt.figure(figsize=(10, 6))
for lr_val in sorted(df["learning_rate"].unique()):
    subset = df[df["learning_rate"] == lr_val]
    grouped = subset.groupby("T")["accuracy"].mean().reset_index()
    grouped["log2_T"] = np.log2(grouped["T"])
    plt.plot(grouped["log2_T"], grouped["accuracy"], label=f"LR={lr_val}")

plt.xlabel("log2(T)")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs log2(T) (by Learning Rate)")
plt.xticks(ticks=[5, 6, 7, 8, 9])  # 设置横坐标为整数刻度
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
