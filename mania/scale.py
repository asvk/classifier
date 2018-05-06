from scipy.misc import imread
import numpy as np

dataset = "/Users/aleksey/dataset/platesmania2/positive.txt"

with open(dataset) as f:
    train_dataset = [s.strip() for s in f]

res = []

for el in train_dataset:
    img = imread(el, mode='RGB')
    shape = img.shape
    h = shape[0]
    w = shape[1]
    ratio = h * 1. / w
    res.append(ratio)

mean = np.mean(res)
std = np.std(res)

print(mean, std)