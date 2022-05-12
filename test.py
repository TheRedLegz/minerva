import pandas as pd
import numpy as np

from minisom import MiniSom

original = pd.read_csv('Mall_Customers.csv')

train = original[['Gender', 'Age', 'Annual Income (k$)']][:180]
train = train.to_numpy()


for i in range(len(train)):
    if train[i][0] == 'Male':
        train[i][0] = 0
    elif train[i][0] == 'Female':
        train[i][0] = 1

    for j in range(len(train[i])):
        train[i][j] = int(train[i][j])

print(train)
test = original[180:]
test = [data for data in test]

n_neurons = 10
m_neurons = 10
som = MiniSom(n_neurons, m_neurons, 3)

som.train(train, 1000)