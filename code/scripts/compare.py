import numpy as np

remurs_X = np.load("remurs_X.npy")
remurs_y = np.load("remurs_y.npy")

old_remurs_X = np.load("old_remurs_X.npy")
old_remurs_y = np.load("old_remurs_y.npy")


print("sums (just to verify)")
print("remurs")
print(f"X: {np.sum(remurs_X)}")
print(f"y: {np.sum(remurs_y)}")
print("old_remurs")
print(f"X: {np.sum(old_remurs_X)}")
print(f"y: {np.sum(old_remurs_y)}")

print()

print("But are they the same?")
print(f"X: {bool(np.prod(remurs_X == old_remurs_X))}")
print(f"y: {bool(np.prod(remurs_y == old_remurs_y))}")
