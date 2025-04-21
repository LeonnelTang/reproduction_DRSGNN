import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

log_path = Path("DRSGNN/training.log")  # 修改为你的日志路径

# 正则表达式提取 epoch 精度信息
pattern = re.compile(
    r"Epoch (\d+):.*?max_train_accuracy=([\d.]+), loss = ([\d.]+), max_val_accuracy=([\d.]+), train_times=\d+"
)

# 提取第一轮 run 日志内容
first_run_lines = []
started = False
with open(log_path, "r") as f:
    for line in f:
        if "Epoch 0:" in line:
            if started:
                break  # 已经读完第一段
            started = True
        if started:
            first_run_lines.append(line)

# 解析 epoch 精度/损失数据
epochs = []
train_accs = []
val_accs = []
losses = []

for line in first_run_lines:
    match = pattern.search(line)
    if match:
        epoch, train_acc, loss, val_acc = match.groups()
        epochs.append(int(epoch))
        train_accs.append(float(train_acc))
        losses.append(float(loss))
        val_accs.append(float(val_acc))

# 构建 DataFrame
df1 = pd.DataFrame({
    "Epoch": epochs,
    "Train Accuracy": train_accs,
    "Validation Accuracy": val_accs,
    "Loss": losses
})

# 🎯 绘图：准确率 vs Epoch
plt.figure(figsize=(10, 6))
plt.plot(df1["Epoch"], df1["Train Accuracy"], label="Train Accuracy")
plt.plot(df1["Epoch"], df1["Validation Accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch (Run 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 绘图：损失 vs Epoch
plt.figure(figsize=(10, 6))
plt.plot(df1["Epoch"], df1["Loss"], label="Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch (Run 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()