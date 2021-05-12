import inline as inline
import numpy as np
import scipy as sc
import sklearn
import unittest; t = unittest.TestCase()
from pprint import pprint
import os
import data
import matplotlib
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats
import timeit
from statistics import mean
from scipy.spatial import distance



def pydistance(x1: 'Vector', x2: 'Vector') -> float:
    '''
        Calculates the square eucledian distance between two data points x1, x2

        Args:
            x1, x2 (vector-like): Two vectors (ndim=1) for which we want to calculate the distance
                `len(x1) == len(x2)` will always be True

        Returns:
            float: The square eucleadian distance between the two vectors
    '''
    return sum([(x1d - x2d) ** 2 for x1d, x2d in zip(x1, x2)])


def pynearest(u: list, X: list, Y: list, distance: callable = pydistance) -> int:
    '''
        Applies the nearest neighbour to the input `u`
        with training set `X` and labels `Y`. The
        distance metric can be specified using the
        `distance` argument.

        Args:
            u (list): The input vector for which we want a prediction
            X (list): A 2 dimensional list containing the trainnig set
            Y (list): A list containing the labels for each vector in the training set
            distance (callable): The distance metric. By default the `pydistance` function

        Returns:
            int: The label of the closest datapoint to u in X
    '''
    xbest = None
    ybest = None
    dbest = float('inf')
    for x, y in zip(X, Y):
        d = distance(u, x)
        if d < dbest:
            ybest = y
            xbest = x
            dbest = d
    return ybest


def pybatch(U, X, Y, nearest=pynearest, distance=pydistance):
    '''
        Applies the nearest neighbor algorithm, to all the datapoints
        `u` $\in$ `U`, with `X` the training set and `Y` the labels.
        Both the distance metric and the method of finding the
        neearest neighbor can be specified.

        Args:
            U (list): List of vectors for which a prediction is desired.
            X (list): A 2 dimensional list containing the trainnig set
            Y (list): A list containing the labels for each vector in the training set
            nearest (callable): The method by which the nearest neighbor search happens.
            distance (callable): The distance metric. By default the `pydistance` function

        Returns:
            list: A list of predicted labels for each `u` $\in$ `U`
    '''
    return [nearest(u, X, Y, distance=distance) for u in U]



def plot():

    # Values for the number of dimensions d to test
    dlist = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    # Measure the computation time for each choice of number of dimensions d
    tlist = []
    for d in dlist:
        U, X, Y = data.toy(100, 100, d)
        # get the average of three runs
        delta = mean(timeit.repeat(lambda: pybatch(U, X, Y), number=1, repeat=3))
        tlist.append(delta)

    # Plot the results in a graph
    fig = plt.figure(figsize=(5, 3))
    plt.plot(dlist, tlist, '-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('d')
    plt.ylabel('time')
    plt.grid(True)
    plt.show()

def npdistance(x1: 'vector-like', x2: 'vector-like') -> float:
    '''
        Calculates the square eucledian distance between two data points x1, x2
        using `numpy` vectorized operations

        Args:
            x1, x2 (vector-like): Two vectors (ndim=1) for which we want to calculate the distance
                `len(x1) == len(x2)` will always be True

        Returns:
            float: The distance between the two vectors x1, x2
    '''
    x1_array = np.array(x1)
    x2_array = np.array(x2)
    X_array = x1_array-x2_array
    X_array_dist = np.sum(X_array**2)
    return X_array_dist

def plot2():
    dlist = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    pydistance_list = []
    for d in dlist:
        U, X, Y = data.toy(100, 100, d)
        delta = mean(timeit.repeat(lambda: pybatch(U, X, Y), number=1, repeat=3))
        pydistance_list.append(delta)

    npdistance_list = []
    for d in dlist:
        U, X, Y = data.toy(100, 100, d)
        delta = mean(timeit.repeat(lambda: pybatch(U, X, Y, distance=npdistance), number=1, repeat=3))
        npdistance_list.append(delta)

    fig = plt.figure(figsize=(5, 3))
    plt.plot(dlist, pydistance_list, '-o', color='red', label='pynearest with pydistance')
    plt.plot(dlist, npdistance_list, '-*', color='green', label='npdistance with npdistance')
    plt.xscale('log');
    plt.yscale('log');
    plt.xlabel('d');
    plt.ylabel('time');
    plt.grid(True)
    plt.show()
    # YOUR CODE HERE


def npnearest(u: np.ndarray, X: np.ndarray, Y: np.ndarray, *args, **kwargs):
    '''
        Finds x1 so that x1 is in X and u and x1 have a minimal distance (according to the
        provided distance function) compared to all other data points in X. Returns the label of x1

        Args:
            u (np.ndarray): The vector (ndim=1) we want to classify
            X (np.ndarray): A matrix (ndim=2) with training data points (vectors)
            Y (np.ndarray): A vector containing the label of each data point in X
            args, kwargs  : Ignored. Only for compatibility with pybatch

        Returns:
            int: The label of the data point which is closest to `u`
    '''
    distance = ((u - X) ** 2).sum(axis=1)
    min_value = distance.min()
    min_value_indexp = np.where(distance == min_value)
    min_value_index = min_value_indexp[0]
    result = Y[min_value_index]
    return int(result)

def plot3():
    # YOUR CODE HERE
    Nlist = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    pydistance_list = []
    for N in Nlist:
        U, X, Y = data.toy(100, N, 100)
        delta = mean(timeit.repeat(lambda: pybatch(U, X, Y, distance=npdistance), number=1, repeat=3))
        pydistance_list.append(delta)

    npdistance_list = []
    for N in Nlist:
        U, X, Y = data.toy(100, N, 100)
        delta = mean(
            timeit.repeat(lambda: pybatch(U, X, Y, nearest=npnearest, distance=npdistance), number=1, repeat=3))
        npdistance_list.append(delta)

    fig = plt.figure(figsize=(5, 3))
    plt.plot(Nlist, pydistance_list, '-o', color='red', label='pynearest with npdistance')
    plt.plot(Nlist, npdistance_list, '-*', color='green', label='npnearest with npdistance')
    plt.xscale('log');
    plt.yscale('log');
    plt.xlabel('d');
    plt.ylabel('time');
    plt.grid(True)
    plt.show()
    # YOUR CODE HERE

def npbatch(U, X, Y, *args, **kwargs):
    '''
        This function has the same functionality as the `pybatch` function.
        HOWEVER, the distance function is fixed (scipy.spatial.distance.cdist).
        It does not use any of the functions defined by us previously.

        Args:
            U (np.ndarray): A matrix (ndim=2) containing multiple vectors which we want to classify
            X (np.ndarray): A matrix (ndim=2) that represents the training data
            Y (np.ndarray): A vector (ndim=1) containing the labels for each data point in X

            All other arguments are ignored. *args, **kwargs are only there for compatibility
            with the `pybatch` function

        Returns:
            np.ndarray: A vector (ndim=1) with the predicted label for each vector $u \in U$
    '''
    dist = distance.cdist(U, X, 'euclidean')
    min_value_index = np.argmin(dist, axis=1)
    result = Y[min_value_index]
    return result

def plot4():
    # YOUR CODE HERE
    Mlist = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    pydistance_list = []
    for M in Mlist:
        U, X, Y = data.toy(M, 100, 100)
        delta = mean(timeit.repeat(lambda: pybatch(U, X, Y), number=1, repeat=3))
        pydistance_list.append(delta)

    npdistance_list = []
    for M in Mlist:
        U, X, Y = data.toy(M, 100, 100)
        delta = mean(
            timeit.repeat(lambda: npbatch(U, X, Y), number=1, repeat=3))
        npdistance_list.append(delta)

    fig = plt.figure(figsize=(5, 3))
    plt.plot(Mlist, pydistance_list, '-o', color='red', label='pybatch')
    plt.plot(Mlist, npdistance_list, '-*', color='green', label='npbatch')
    plt.xscale('log');
    plt.yscale('log');
    plt.xlabel('d');
    plt.ylabel('time');
    plt.grid(True)
    plt.show()

def plot_first_digits():
    '''
        Loads the digit dataset and plots the first 16 digits in one image
        You are encouraged to implement this functions without
        the use of any for-loops
    '''
    X, Y = data.digits()
    A = X[0:4, 0:4]
    fig, ax = plt.subplots()
    ax.imshow(A, origin='lower')
    ax.set_aspect('equal')


def train_test_split(x: np.ndarray, y: np.ndarray):
    '''
    Splits the data into train and test sets
    The first 1000 samples belong to the training set the rest to the test set

    Args:
        x (np.ndarray): A matrix (ndim=2) containing the data
        y (np.ndarray): A vector (ndim=1) containing the label for each datapoint

    Returns:
        tuple: A tuple containing 4 elements. The training data, the test data, the training labels
            and the test labels
    '''
    [x_train, x_test] = np.split(x, [1000], axis=0)
    [y_train, y_test] = np.split(y, [1000], axis=0)
    return x_train, x_test, y_train, y_test


def predict(x_train, x_test, y_train):
    '''
    For each x in x_test this function finds the nearest neighbour in x_train and
    returns that label

    This function is a wrapper of the `npbatch` function

    Args:
        x_train (np.ndarray): A matrix (ndim=2) containing all the training data
        x_test (np.ndarray): A matrix (ndim=2) containing all the test data for which we want a prediction
        y_train (np.ndarray): A vector (ndim=1) containing the label of each datapoint in the training set

    Returns:
        np.ndarray: A vector with the prediction for each datapoint/vector in x_test
    '''
    y_test_predict = npbatch(x_test, x_train, y_train)
    return y_test_predict

def evaluate(x_train, x_test, y_train, y_test) -> float:
    '''
    Evaluates the accuracy of our nearest neighbor classifier
    by calculating the ratio of test samples for which
    our classification method disagrees with the ground truth

    Args:
        x_train (np.ndarray): A matrix (ndim=2) containing the training data for the classifier
        x_test (np.ndarray): A matrix (ndim=2) containing the test data for which the classifier
            will make a prediction
        y_train (np.ndarray): The labels for the training data
        y_test (np.ndarray): The labels for the test data
    Returns:
        float: The ratio in [0-1] of the test samples for which our
            nearest neighbor classifier disagrees with the provided labels
    '''
    predictions = predict(x_train, x_test, y_train)
    disagree = np.equal(predictions, y_test)
    true_value = np.sum(disagree)
    length = len(disagree)
    false_value = length - true_value
    return false_value/length

if __name__ == '__main__':
    plot_first_digits()


