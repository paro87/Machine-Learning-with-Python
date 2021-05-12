from typing import Dict, List, Tuple, Optional
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from utils import train_test_idxs

t = TestCase()
from minified import max_allowed_loops, no_imports

from IPython.display import Markdown as md


from ML3.minified import max_allowed_loops, no_imports


@no_imports
@max_allowed_loops(1)
def read_from_file(file: str = "data.csv") -> Dict[str, List[Tuple[float, float]]]:
    """
        Opens a csv file and parses it line by line. Each line consists of a label and two
        data dimensions. The function returns a dictionary where each key is a label and
        the value is a list of all the datapoints that have that label. Each datapoint
        is represented by a pair (2-element tuple) of floats.

        Args:
            file (str, optional): The path to the file to open and parse. Defaults to
            "data.csv".

        Returns:
            Dict[str, List[Tuple[float, float]]]: The parsed contents of the csv file
    """
    D = {}
    lst = []
    with open('./'+file, 'r') as f:
        for line in f:
            lst = line.split(',')
            key = lst[0]
            value = (float(lst[1]), float(lst[2].strip()))
            D.setdefault(key, []).append(value)
    return D


@no_imports
@max_allowed_loops(1)
def stack_data(D: Dict[str, List[Tuple[float, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
        Convert a dictionary dataset into a two arrays of data and labels. The dictionary
        keys represent the labels and the value mapped to each key is a list that
        contains all the datapoints belonging to that label. The output are two arrays
        the first is the datapoints in a single 2d array and a vector of intergers
        with the coresponding label for each datapoint. The order of the datapoints is
        preserved according to the order in the dictionary and the lists.

        The labels are converted from a string to a unique int.

        The datapoints are entered in the same order as the keys in the `D`. First
        all the datapoints of the first key are entered then the second and so on.
        Within one label order also remains.

        Args:
            D (Dict[str, List[Tuple[float, float]]]): The dictionary that should be stacked

        Returns:
            Tuple[np.ndarray, np.ndarray]: The two output arrays. The first is a
            float-matrix containing all the datapoints. The second is an int-vector
            containing the labels for each datapoint
    """
    X = []
    y = np.array([])
    label = 0
    for key, value in D.items():
        key_array = np.array([1] * len(value))*label
        y = np.hstack((y, key_array))
        label += 1
        X.extend(value)

    X = np.array(X)
    y = y.astype(np.int64)
    return X, y


@no_imports
@max_allowed_loops(1)
def get_clusters(X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    """
        Receives a labeled dataset and splits the datapoints according to label

        Args:
            X (np.ndarray): The dataset
            y (np.ndarray): The label for each point in the dataset

        Returns:
            List[np.ndarray]: A list of arrays where the elements of each array
            are datapoints belonging to the label at that index.

        Example:
        >>> get_clusters(
                np.array([[0.8, 0.7], [0, 0.4], [0.3, 0.1]]),
                np.array([0,1,0])
            )
        >>> [array([[0.8, 0.7],[0.3, 0.1]]),
             array([[0. , 0.4]])]
    """
    pointList = []
    maxValue = np.max(y)
    for i in range(maxValue+1):
        result = np.where(y == i)
        new_X = np.array(X[result[0]])
        pointList.append(new_X)
    return pointList


@no_imports
@max_allowed_loops(0)
def split(X: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split the data into train and test sets. The training and test set are clustered by
    label using `get_clusters`. The size of the training set is 80% of the whole
    dataset

    Args:
        X (np.ndarray): The dataset (2d)
        y (np.ndarray): The labels (1d)

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: The clustered training and
        testset
    """
    (X_test_indices, X_train_indices) = train_test_idxs(len(X), 0.8)
    (y_test_indices, y_train_indices) = train_test_idxs(len(y), 0.8)

    X_train = X[X_train_indices]
    y_train = y[y_train_indices]
    X_test = X[X_test_indices]
    y_test = y[y_test_indices]

    tr_ratio = 0.8
    limit = int(tr_ratio * len(X))
    # X_train = X[:limit]
    # y_train = y[:limit]
    # X_test = X[limit:]
    # y_test = y[limit:]

    tr_clusters = get_clusters(X_train, y_train)
    te_clusters = get_clusters(X_test, y_test)

    return tr_clusters, te_clusters


@no_imports
@max_allowed_loops(1)
def calc_means(clusters: List[np.ndarray]) -> np.ndarray:
    """
    For a collections of clusters calculate the mean for each cluster

    Args:
        clusters (List[np.ndarray]): A list of 2d arrays

    Returns:
        List[np.ndarray]: A matrix where each row represents a mean of a cluster

    Example:
        >>> tiny_clusters = [
            np.array([[0.2, 0.3], [0.1, 0.2]]),
            np.array([[0.8, 0.9], [0.7, 0.5], [0.6, 0.7]]),
        ]
        >>> calc_means(tiny_clusters)
        [array([0.15, 0.25]), array([0.7,0.7])]
    """
    meanList = []
    for cluster in clusters:
        cluster_mean = np.mean(cluster, axis = 0)
        meanList.append(cluster_mean)
    result = np.array(meanList)
    return result


@no_imports
def plot_scatter_and_mean(clusters: List[np.ndarray], letters: List[str], means: Optional[List[np.ndarray]] = None,
) -> None:
    """
    Create a scatter plot visulizing each cluster and its mean

    Args:
        clusters (List[np.ndarray]): A list containing arrrays representing
        each cluster
        letters (List[str]): The "name" of each cluster
        means (Optional[List[np.ndarray]]): The mean of each cluster. If not
        provided the mean of each cluster in `clusters` should be calculated and
        used

    """
    assert len(letters) == len(clusters)

    title = "Scatter plot of the clusters"
    plt.figure(figsize=(8,8))
    plt.title(title, fontsize=20)
    mean_list = calc_means(clusters)
    mean_index = 0
    for cluster in clusters:
        x_vals = cluster[:,0]
        y_vals = cluster[:,1]
        x_mean = np.float16(mean_list[mean_index, 0])
        y_mean = np.float16(mean_list[mean_index, 1])
        plt.plot(x_vals, y_vals, 'o', color='b', ms=5, alpha=0.6, label=f'Cluster: {letters[mean_index]}')
        plt.plot(x_mean, y_mean, 'x', color='r', ms=7, label=f'mean: {x_mean}, {y_mean}', zorder=3)
        mean_index += 1

    plt.legend(loc='best')
    plt.show()


@no_imports
def plot_projection(clusters: List[np.ndarray], letters: List[str], means: np.ndarray, axis: int = 0):
    """
    Plot a histogram of the dimension provided in `axis`

    Args:
        clusters (List[np.ndarray]): The clusters from which to create the historgram
        letters (List[str]): The string representation of each class
        means (np.ndarray): The mean of each class
        axis (int): The axis from which to create the historgram. Defaults to 0.
    """
    title = f"Projection to axis {axis} histogram plot"
    plt.figure(figsize=(14, 5))
    plt.title(title, fontsize=20)
    mean_index = 0
    for cluster in clusters:
        vals = cluster[:, axis]
        plt.hist(vals, bins=30, rwidth=0.8, alpha=0.6, label=f'Cluster: {letters[mean_index]}', density=True)  # num of bins, block width percentage
        plt.axvline(x=vals.mean(), ls='--', c='r')  # plot dashed mean line
        mean_index += 1

    plt.legend(loc='best')
    plt.show()


@no_imports
@max_allowed_loops(1)
def within_cluster_cov(clusters: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the within class covariance for a collection of clusters

    Args:
        clusters (List[np.ndarray]): A list of clusters each consisting of
        an array of datapoints

    Returns:
        np.ndarray: The within cluster covariance

    Example:
        >>> within_cluster_cov(
            [array([[0.2, 0.3], [0.1, 0.2]]), array([[0.8, 0.9], [0.7, 0.5], [0.6, 0.7]])]
        )
        >>> array([[0.025, 0.025],
                   [0.025, 0.085]])
    """
    d = clusters[0].shape[1]
    S_w = np.zeros((d, d))
    mean_list = calc_means(clusters)
    mean_index = 0
    for cluster in clusters:
        mean = mean_list[mean_index]
        res = cluster - mean
        resT = res.T
        S_w += np.dot(resT, res)
        mean_index += 1
    return S_w


@no_imports
@max_allowed_loops(0)
def calc_mean_of_means(clusters: List[np.ndarray]) -> np.ndarray:
    """
    Given a collection of datapoints divided in clusters, calculate the
    mean of all cluster means.
    Args:
        clusters (List[np.ndarray]): A list of clusters represented in arrays

    Returns:
        np.ndarray: A single datapoint that represents the mean of all the
        cluster means

    Example:
        >>> calc_mean_of_means(
                [np.array([[0.222, 0.333], [0.1, 0.2]]), np.array([[0.8, 0.9], [0.7, 0.5], [0.6, 0.7]])]
            )
        >>> array([0.4305 , 0.48325])
    """
    mean_list = calc_means(clusters)
    return np.mean(mean_list, axis=0)


@no_imports
@max_allowed_loops(1)
def between_cluster_cov(clusters: List[np.ndarray], cluster_means: List[np.ndarray], mean_of_means: np.ndarray) -> np.ndarray:
    """
    Calculate the covariance between clusters.

    Args:
        clusters (List[np.ndarray]): A list of datapoints divided by cluster
        cluster_means (List[np.ndarray]): A list of vectors representing the mean
        of each cluster
        mean_of_means (np.ndarray): A vector, the mean of all datapoints

    Returns:
        np.ndarray: Covariance between clusters

    Example:
        >>> tiny_clusters = [
            np.array([[0.2, 0.3], [0.1, 0.2]]),
            np.array([[0.8, 0.9], [0.7, 0.5], [0.6, 0.7]]),
        ]
        >>> tiny_means = [np.array([0.15, 0.25]), np.array([0.7, 0.7])]
        >>> tiny_mean_of_means = np.array([0.425, 0.475])
        >>> between_cluster_cov(tiny_clusters, tiny_means, tiny_mean_of_means)
        array([[0.378125, 0.309375],
               [0.309375, 0.253125]])

    """
    d = clusters[0].shape[1]
    S_b = np.zeros((d, d))
    mean_index = 0
    for cluster in clusters:
        Nk = len(cluster)
        mean_diff = cluster_means[mean_index] - mean_of_means
        mean_diff = np.array([mean_diff])
        mean_diff_transpose = mean_diff.T
        dot1 = np.dot(Nk, mean_diff_transpose)
        dot2 = np.dot(dot1, mean_diff)
        S_b +=dot2
        mean_index += 1
    return S_b


@no_imports
@max_allowed_loops(0)
def rotation_matrix(S_w: np.ndarray, S_b: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Calculate the transformation matrix given the within- and between cluster
    covariance matrices.

    Args:
        S_w (np.ndarray): The within cluster covariance
        S_b (np.ndarray): The between cluster covariance

    Returns:
        np.ndarray: The transformation matrix
        int: The axis along with the transformed data achieves maximal variance

    Example:
        >>> tiny_S_w = np.array([[0.025, 0.025], [0.025, 0.085]])
        >>> tiny_S_b = np.array([[0.378125, 0.309375], [0.309375, 0.253125]])
        >>> rotation_matrix(tiny_S_w, tiny_S_b)
        (array([[ 0.99752952, -0.63323779],
                [-0.07024856,  0.7739573 ]]), 0)
    """
    A = np.linalg.solve(S_w, S_b)
    w, v = np.linalg.eig(A)
    max_axis = np.argmax(w)
    tup = tuple((v, max_axis))
    return tup


@no_imports
@max_allowed_loops(1)
def rotate_clusters(W_rot: np.ndarray, clusters: List[np.ndarray]) -> List[np.ndarray]:
    """
    Rotate all the datapoints in all the clusters

    Args:
        W_rot (np.ndarray): The rotation matrix
        clusters (List[np.ndarray]): The list of datapoints divided in clusters that
        will be rotated

    Returns:
        List[np.ndarray]: The rotated datapoints divided by cluster
    """
    result = []
    for cluster in clusters:
        cluster_rotation = cluster.dot(W_rot)
        result.append(cluster_rotation)
    return result

if __name__ == '__main__':
    letters = "MNU"
    D = read_from_file(file="data.csv")
    X, y = stack_data(D)
    output = split(X, y)
    tr_clusters, te_clusters = output
    means = calc_means(tr_clusters)
    mean_of_means = calc_mean_of_means(tr_clusters)
    S_w = within_cluster_cov(tr_clusters)
    S_b = between_cluster_cov(tr_clusters, means, mean_of_means)
    output = rotation_matrix(S_w, S_b)
    W_rot, max_axis = output
    rad = np.deg2rad(30)
    c, s = np.cos(rad), np.sin(rad)
    rot30 = np.array([[c, -s], [s, c]])


    tiny_clusters = [
        np.array([[0.2, 0.3], [0.1, 0.2]]),
        np.array([[0.8, 0.9], [0.7, 0.5], [0.6, 0.7]]),
    ]
    tiny_rotated_result = rotate_clusters(rot30, tiny_clusters)
    print(tiny_rotated_result)
    tiny_rotated_expected = [
        np.array([[0.32320508, 0.15980762], [0.18660254, 0.12320508]]),
        np.array(
            [[1.14282032, 0.37942286], [0.85621778, 0.0830127], [0.86961524, 0.30621778]]
        ),
    ]
    for r, e in zip(tiny_rotated_result, tiny_rotated_expected):
        np.testing.assert_allclose(r, e)

    rot_tr_clusters = rotate_clusters(W_rot, tr_clusters)
    t.assertIsInstance(rot_tr_clusters, List)
    for norm, rotated in zip(tr_clusters, rot_tr_clusters):
        t.assertIsInstance(rotated, np.ndarray)
        t.assertEqual(norm.shape, rotated.shape)


