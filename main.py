import matplotlib.pyplot as plt
import random
import numpy as np

from model import Adaboost

pos_n = random.randint(5, 20)
neg_n = random.randint(5, 20)
data = []
label = []

# Dimension 2
for n in range(pos_n):
    data.append([random.uniform(0, 60), random.uniform(0, 60)])
    label.append(1) # pos
for n in range(neg_n):
    data.append([random.uniform(40, 100), random.uniform(40, 100)])
    label.append(-1) # neg

data = np.array(data)
print(data.shape)
print("pos: %d, neg: %d" %(pos_n, neg_n))

model = Adaboost()
model.train(data, label)

# input a random vector with dimension 2
res = model.predict([80, 80])
print("Predicted label: ", res)