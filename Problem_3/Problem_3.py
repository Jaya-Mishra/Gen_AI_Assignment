# 3.	Problem Statement

# Write a Python program to draw (visualize) the architecture of a Neural Network used to classify fraudulent and non-fraudulent credit card transactions.

# Assume the fraud detection dataset contains the following input features:

# TransactionAmount
# TransactionTime
# MerchantCategory
# CustomerAge
# AccountBalance
# NumberOfTransactionsToday
# Fraud (0 = Genuine, 1 = Fraud)


import matplotlib.pyplot as plt
 
layers = [
    ("Input Layer", 6),
    ("Hidden Layer 1", 16),
    ("Hidden Layer 2", 8),
    ("Hidden Layer 3", 4),
    ("Output Layer", 1)
]
 
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Neural Network Architecture for Fraud Detection", fontsize=10)
ax.axis("off")
 
x_space = 2
y_space = 0.5
 
for i, (layer_name, num_neurons) in enumerate(layers):
    x = i * x_space
    y_start = -(num_neurons - 1) * y_space / 2
 
    for j in range(num_neurons):
        y = y_start + j * y_space
        circle = plt.Circle((x, y), 0.12, fill=False)
        ax.add_patch(circle)
 
    ax.text(x, y_start - 1.2, layer_name, ha="center", fontsize=10)
 
for i in range(len(layers) - 1):
    x1 = i * x_space
    x2 = (i + 1) * x_space
 
    n1 = layers[i][1]
    n2 = layers[i + 1][1]
 
    y1_start = -(n1 - 1) * y_space / 2
    y2_start = -(n2 - 1) * y_space / 2
 
    for j in range(n1):
        for k in range(n2):
            y1 = y1_start + j * y_space
            y2 = y2_start + k * y_space
            ax.plot([x1 + 0.12, x2 - 0.12], [y1, y2], linewidth=0.5)
 
input_features = [
    "TransactionAmount",
    "TransactionTime",
    "MerchantCategory",
    "CustomerAge",
    "AccountBalance",
    "NoOfTransactionsToday"
]
 
x_input = 0
y_start = -(len(input_features) - 1) * y_space / 2
 
for i, feature in enumerate(input_features):
    y = y_start + i * y_space
    ax.text(x_input - 1.3, y, feature, ha="right", fontsize=9)
 
ax.text((len(layers) - 1) * x_space + 1.2, 0, "Fraud\n(0 = Genuine\n1 = Fraud)",
        ha="left", fontsize=10)
 
plt.show()
