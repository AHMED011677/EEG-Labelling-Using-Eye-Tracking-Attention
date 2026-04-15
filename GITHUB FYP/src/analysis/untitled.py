import numpy as np

data = np.load(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\labelled_dataset.npz")

X = data["X"]
y = data["y"]

print("X shape:", X.shape)
print("y:", y)