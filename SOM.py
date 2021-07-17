
from nltk.grammar import standard_nonterm_parser
import numpy as np
from pprint import pprint
from math import sqrt
import lsa
from matplotlib import pyplot as plt

np.random.seed(1)


def most_common(lst, n):
  if len(lst) == 0:
      return -1
      
  counts = np.zeros(shape=n, dtype=np.int)

  for i in range(len(lst)):
    counts[lst[i]] += 1

  return np.argmax(counts)

def find_bmu(matrix, input, matrix_size):
    (row, col) = matrix_size

    num_features = len(input)

    bmu = (0,0)
    nearest = 10000000

    for i in range(row):
        for j in range(col):

            distance = euc_distance(input, matrix[i][j], num_features)

            if distance < nearest:
                nearest = distance
                bmu = (i, j)

    return bmu

def euc_distance(input, cell, num_features):
    sum = 0

    for n in range(num_features):
        sum = sum + pow(input[n] - cell[n], 2)
    
    return sqrt(sum)

def man_distance(bmu_row, bmu_col, row, col):
    return np.abs(bmu_row-row) + np.abs(bmu_col-col)

def SOM(data, learn_rate, matrix_size):

    # 1 initialize some constants

    (steps, num_features) = data.shape
    (row, col) = matrix_size
    range_max = row + col # change this

    # 2 create the initial matrix

    matrix = np.random.random_sample(size=(row,col,num_features))

    # 3 run main logic

    steps_max = steps # change this

    indices = np.zeros(len(data))


    for s in range(steps_max):
        
        # update
        percent = 1.0 - ((s * 1.0) / steps_max)
        curr_range = (int)(percent * range_max)
        curr_rate = percent * learn_rate

        # get a unique input from data

        rand = None

        while True:
            rand = np.random.randint(len(data))

            if(indices[rand] != 1):
                break

            indices[rand] = 1

        input = data[rand]

        # get the bmu
        (bmu_row, bmu_col) = find_bmu(matrix, input, matrix_size)

        for i in range(row):
            for j in range(col):

                cell = matrix[i][j]

                if man_distance(bmu_row, bmu_col, i, j) < curr_range:
                    
                    matrix[i][j] = cell  + curr_rate * (input - cell)

    return matrix



matrix_size = (10, 10)
(row, col) = matrix_size

texts = lsa.preprocessing()
res = lsa.topicModeling(texts)

result = SOM(res, .5, matrix_size)


print(result)

print("Constructing U-Matrix from SOM")

u_matrix = np.zeros(shape=matrix_size, dtype=np.float64)

for i in range(row):
    for j in range(col):
        
        v = result[i][j]

        sum_dists = 0.0
        ct = 0
        
        if i-1 >= 0:  
            sum_dists += euc_distance(v, result[i-1][j], len(v)); ct += 1

        if i+1 <= row-1:
            sum_dists += euc_distance(v, result[i+1][j], len(v)); ct += 1

        if j-1 >= 0:  
            sum_dists += euc_distance(v, result[i][j-1], len(v)); ct += 1

        if j+1 <= col-1: 
            sum_dists += euc_distance(v, result[i][j+1], len(v)); ct += 1

        
        u_matrix[i][j] = sum_dists / ct


plt.imshow(u_matrix, cmap='gray') 
plt.show()

print("Associating each data label to one map node ")

mapping = np.empty(shape=matrix_size, dtype=object)

for i in range(row):
    for j in range(col):
        mapping[i][j] = []

for t in range(len(res)):
    (m_row, m_col) = find_bmu(result, res[t], matrix_size)
    
    rand = np.random.randint(0, 2)
    mapping[m_row][m_col].append(rand)


label_map = np.zeros(shape=matrix_size, dtype=np.int)

for i in range(row):
    for j in range(col):
        label_map[i][j] = most_common(mapping[i][j], 3)

plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4))
plt.colorbar()
plt.show()





