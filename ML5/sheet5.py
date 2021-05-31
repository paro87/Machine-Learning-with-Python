import math

import utils
import numpy as np
import itertools
import unittest

from typing import Union
from minified import max_allowed_loops, no_imports

t = unittest.TestCase()

def softplus(z):
    return np.log(1 + np.exp(z))

@no_imports
def softplus_robust(x: Union[float, np.float32, np.float64]) -> Union[float, np.float32, np.float64]:
    """
    Numerically stable implementation of softplus function. Will never
    overflow to infinity if input is finite

    Args:
        x (Union[float, np.float32, np.float64]): The number of which we
        want to calculate the softplus value
    Returns:
        Union[float, np.float32, np.float64]: softplus(x)
    """
    # YOUR CODE HERE
    if x>0:
        return x+np.log(1 + np.exp(-x))
    return np.log(1 + np.exp(x))
    # YOUR CODE HERE




@max_allowed_loops(0)
@no_imports
def softplus_robust_vec(X: "vector-like"):
    """
    Vectorized version of the numericaly robust softplus function

    Args:
        X (vector-like): A vector (ndim=1) of values on which we want to apply the softplus function.
            It is not always a np.ndarray

    Returns:
        np.ndarray: A vector (ndim=1) where the ret[i] == softplus_robust(X[i])
    """
    # these are wrong!!!
    # return np.array([softplus_robust(x) for x in X])
    # return np.array(list(map(softplus_robust, X)))
    # return np.vectorize(softplus_robust)(X)
    # etc...
    # YOUR CODE HERE
    d = np.log(1 + np.exp(-np.abs(X))) + np.maximum(X, 0)
    print(d)
    return d
    # YOUR CODE HERE


def generate_all_observations() -> np.ndarray:
    """
    All x in { -1,1 }^10 (vectors with 10 elements where each element
    containts either -1 or 1)

    Returns:
        np.ndarray : All valid obvervations
    """
    return np.array(tuple(itertools.product([-1.0, 1.0], repeat=10)))


def calc_logZ(w: np.ndarray) -> float:
    """
    Calculates the log of the partition function Z

    Args:
        w (np.ndarray): A ten element vector (shape=(10,)) of parameters
    Returns:
        float: The log of the partition function Z
    """

    print()
    Z = np.sum(np.exp(generate_all_observations() @ w))
    return np.log(Z)


@no_imports
@max_allowed_loops(0)
def calc_logZ_robust(w):


    # YOUR CODE HERE
    T = generate_all_observations() @ w
    maximum = np.max(T)
    Z = np.sum(np.exp(generate_all_observations() @ w-maximum))
    return maximum + np.log(Z)
    # YOUR CODE HERE


@no_imports
@max_allowed_loops(0)
def important_indexes(w: np.ndarray, tol: float = 0.001) -> np.ndarray:
    """
    Calculates the indexes of important binary vectors for the
    parameter vector `w`.

    Args:
        w (np.ndarray): The parameter vector of the partition function
        tol (float): The probability threshold

    Returns:
        (np.ndarray): The indexes where the probability is greater or equal
        to `tol`
    """

    logZ = calc_logZ_robust(w)


    # YOUR CODE HERE
    T = generate_all_observations() @ w
    y = T - logZ
    z = np.exp(y)
    tpl = np.where(z >= tol)
    arr = np.asarray(tpl[0])
    return arr.astype(np.int32)
    # YOUR CODE HERE


@no_imports
@max_allowed_loops(0)
def logp_robust(X: np.ndarray, m: np.ndarray, S: np.ndarray):
    """
    Numerically robust implementation of `logp` function

    Returns:
        (float): The logp probability density
    """
    # YOUR CODE HERE
    N = X.shape[0]
    d = X.shape[1]

    F1 = N*d*np.log(2*np.pi)
    F2 = N*np.log(np.linalg.det(S))
    D = X - m
    A = np.linalg.solve(S, np.transpose(D))
    trace = np.trace(np.dot(D,A))
    F3 =-0.5*(F1+F2+trace)
    return F3
    # YOUR CODE HERE

def logp(X, m, S):
    # Find the number of dimensions from the data vector
    d = X.shape[1]
    print(X)
    # Invert the covariance matrix
    Sinv = np.linalg.inv(S)

    # Compute the quadratic terms for all data points
    Q = -0.5 * (np.dot(X - m, Sinv) * (X - m)).sum(axis=1)

    # Raise them quadratic terms to the exponential
    Q = np.exp(Q)

    # Divide by the terms in the denominator
    P = Q / np.sqrt((2 * np.pi) ** d * np.linalg.det(S))
    print(P)
    print(P.size)
    # Take the product of the probability of each data points
    Pprod = np.prod(P)

    # Return the log-probability
    return np.log(Pprod)

if __name__ == '__main__':
    # Verify your function
    X = utils.softplus_inputs
    y_scalar = [softplus_robust(x) for x in X]

    for x, y in zip(X, y_scalar):
        print("softplus(%11.4f) = %11.4f" % (x, y))

    # the elements can be any of the three
    t.assertIsInstance(y_scalar[0], (float, np.float32, np.float64))
    t.assertAlmostEqual(softplus_robust(100000000), 100000000)

