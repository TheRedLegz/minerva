import numpy as np
import pandas as pd
from pprint import pprint
from math import sqrt, exp
from matplotlib import pyplot as plt

np.random.seed(101)


def get_topic_words(matrix, unique, size):
    (row, col) = size
    res = []

    for i in range(row):
        for j in range(col):
            obj = {}

            obj['words'] = []
            obj['weight'] = []

            indx = np.argsort(matrix[i, j, :])[-10:]

            for k in indx:
                obj['words'].append(unique[k][0])
                obj['weight'].append(matrix[i, j, k])

            res.append(obj)
    
    return res


def get_word_cluster(matrix, unique, size):
    (row, col) = size

    deets = []

    for idx in range(len(unique)):
        deet = {}
        sum = 0
        deet['weights'] = []

        for i in range(row):
            for j in range(col):
                weight = matrix[i, j, idx]
                sum += weight
                deet["weights"].append(weight)
        deet['sum'] = sum
        deets.append(deet)
            
    for idx in range(len(unique)):
        deets[idx]['distances'] = []
        sum = deets[idx]['sum']

        for cell in range(len(deets[idx]['weights'])):
            value = deets[idx]['weights'][cell]

            deets[idx]['distances'].append(value/sum)

    return deets



def get_tw_matrix(matrix, unique, size):
    (row, col) = size

    res = []

    for i in range(row):
        for j in range(col):
            sum = np.sum(matrix[i, j])
            distances = []

            for k in range(len(unique)):
                distances.append(matrix[i, j, k]/sum)
            
            res.append(distances)

    return res


def find_topics(SOM_matrix, data_matrix, data_grams, labels, matrix_size):
    topics = []

    for word in data_grams:
        topic_location = find_bmu(
            SOM_matrix, data_matrix[labels.index(word)], matrix_size)
        topics.append((topic_location, word))

    return topics


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

    bmu = (0, 0)
    nearest = 10000000

    for i in range(row):
        for j in range(col):

            distance = euc_distance(input, matrix[i][j], num_features)

            if distance < nearest:
                nearest = distance
                bmu = (i, j)

    return bmu

# NOTE: Contemplate whether preprocessed before or after


def tweet_find_cluster(SOM_model, matrix_size, preprocessed_tweet, unique):
    """
    Takes in the SOM model, tweet, and unique keywords and returns:

    result_matrix = The distances of the input tweet to all the clusters on the map

    bmu = The indices of the best matching cluster in the map
    """
    (row, col) = matrix_size
    u_keywords = [unique for (unique, _) in unique]
    idf_list = [idf for (_, idf) in unique]
    tweet_keyword_list = preprocessed_tweet

    num_features = len(unique)
    tweet_tfidf_values = np.zeros(num_features, dtype=float)

    for keyword in tweet_keyword_list:
        if keyword in u_keywords:
            tweet_tfidf_values[u_keywords.index(keyword)] += 1

    for i, tf in enumerate(tweet_tfidf_values):
        tweet_tfidf_values[i] = (tf / len(tweet_keyword_list)) * idf_list[i]

    bmu = (0, 0)
    nearest = 10000000
    result_matrix = np.zeros(matrix_size, dtype=float)
    total_distance = 0

    for i in range(row):
        for j in range(col):
            distance = euc_distance(
                tweet_tfidf_values, SOM_model[i][j], num_features)
            result_matrix[i][j] = distance
            total_distance += distance
            if distance < nearest:
                nearest = distance
                bmu = (i, j)

    for i in range(row):
        for j in range(col):
            result_matrix[i][j] /= total_distance
    return (result_matrix, bmu)


def euc_distance(input, cell, num_features):
    sum = 0

    for n in range(num_features):
        sum = sum + pow(input[n] - cell[n], 2)

    return sqrt(sum)


def man_distance(bmu_row, bmu_col, row, col):
    return np.abs(bmu_row-row) + np.abs(bmu_col-col)


def SOM(data, learn_rate, matrix_size, steps_max=3000):

    # 1 Initialize some constants

    (steps, num_features) = data.shape
    (row, col) = matrix_size
    range_max = row + col

    # 2 Create the initial matrix

    matrix = np.random.random_sample(size=(row, col, num_features))

    # 3 Run main logic

    indices = np.zeros(len(data))

    for s in range(steps_max):
        if s % (steps_max/10) == 0:
            print(str(np.round((s/steps_max) * 100.00)) + " percent")

        # Update
        percent = 1.0 - ((s * 1.0) / steps_max)

        # curr_range for manhattan distance updating
        curr_range = (int)(percent * range_max)

        # curr_range for pythagorean updating
        # curr_range = range_max * exp(-(s / steps_max))
        curr_rate = percent * learn_rate

        # get a unique input from data

        rand = np.random.randint(len(data))

        input = data[rand]

        # get the bmu
        (bmu_row, bmu_col) = find_bmu(matrix, input, matrix_size)

        for i in range(row):
            for j in range(col):

                cell = matrix[i][j]

                # Weight Updating (w/ man_distance)
                if man_distance(bmu_row, bmu_col, i, j) < curr_range:
                    matrix[i][j] = cell + curr_rate * (input-cell)

                    # [1, 0.4, 0.5, ..., 9n]
                    # [0.3, 0.6, 1, ..., 9n]

                # Weight Updating (w/o Mexican Hat)
                # Formula: cell+curr_rate*(EXP(-((POWER(bmui-i,2)+POWER(bmuj-j,2))/(num_features???*POWER(curr_range,2)))))*(input- cell)
                # matrix[i][j] = cell  + curr_rate * (exp(-(((bmu_row - i)**2 + (bmu_col - j)**2)/(num_features*(curr_range**2))))) * (input - cell)

    return matrix

def get_SOM_model(SOM_matrix, matrix_data, labels):
    """Main function to get the SOM model"""
    matrix_size = SOM_matrix.shape
    (row, col, _) = matrix_size
    mapping = np.empty(shape=(row, col), dtype=object)

    for i in range(row):
        for j in range(col):
            mapping[i][j] = []

    for t in range(len(matrix_data)):
        (m_row, m_col) = find_bmu(SOM_matrix, matrix_data[t], (row, col))
        mapping[m_row][m_col].append(labels[t])

    return mapping
