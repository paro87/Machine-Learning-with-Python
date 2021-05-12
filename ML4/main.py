import utils
from typing import Dict, List
import matplotlib.pyplot as plt
from minified import max_allowed_loops, no_imports

# Use unittest asserts
import unittest
import random
import numpy as np

t = unittest.TestCase()

@max_allowed_loops(1)
@no_imports
def simulate(transitions: Dict[str, str]) -> List[str]:
    """
    Simulates a markov chain defined by the above transitions.
    This function always sets the random seed to `123`. All simulations start with
    initial state `A`. It always simulates 2000 steps including the initial state.
    Args:
        transitions (Dict[str, str]): A dictionary with eight keys [A-H]. For each key a string is
        mapped as its value. Each of those strings can only contain the letters [A-H] each
        letter can only appear once. `'A': 'BE'` means that from state `A` we can reach
        the states `B` and `E` and no other state.
    Returns:
        List[str]: A list of states (a string containing one of the letters [A-H])
        that were visited during the simulation.
    """
    initial_state = 'A'
    random.seed(123)
    visited_states = []
    for i in range(2000):
        visited_states.append(initial_state)
        value = transitions.get(initial_state)
        initial_state = value[random.randint(0, len(value)-1)]
    return visited_states

@max_allowed_loops(1)
@no_imports
def compute_histogram(valid_states: List[str], state_sequence: List[str]) -> List[float]:
    """
    Returns a list of percentages relating as to how many times each state
    has been visited according to the `state_sequence` list

    Args:
        valid_states (List[str]): A list of all valid states
        state_sequence (List[str]): A sequence of states for which we
            want to calculate the frequencies
    Returns:
        List[float]: A list of length 8. Contains the percentage `[0-1]` of occurrences of each state
        in the `state_sequence`.
    """
    occurrence_percentages = []
    for state in valid_states:
        occurrence_number = state_sequence.count(state)
        occurrence_percentage = occurrence_number/len(state_sequence)
        occurrence_percentages.append(occurrence_percentage)
    return occurrence_percentages

@max_allowed_loops(0)
@no_imports
def plot_histogram(valid_states: List[str], frequencies: List[float]) -> None:
    """
    Plots a bar graph of a provided histogram.

    Args:
        valid_states (List[str]): The list of states
        frequencies (List[float]): The frequency of each state
    """
    ticks = range(len(frequencies))
    plt.bar(ticks, frequencies, align='center')
    plt.xticks(ticks, valid_states)
    plt.show()

@max_allowed_loops(0)
@no_imports
def modify_transitions(transitions: Dict[str, str]) -> Dict[str, str]:
    """
    Creates a modified transition dictionary without modifying the provided one.

    This function creates a new transition dictionary such that from state `F` the only
    possible following state is `E`.

    Args:
        transitions (dict): A dictionary that describes the possible transitions from each state
    Returns:
        dict: A modified transition dict where from state `F` only state `E` can follow
    """
    keys = list(transitions.keys())
    values = list(transitions.values())
    index_F = keys.index('F')
    values[index_F] = 'E'
    zip_iterator = zip(keys, values)
    new_transitions = dict(zip_iterator)
    return new_transitions

def state_string_to_index(state: str) -> int:
    """
    Converts the state string into a numerical index, where:
    'A' -> 0
    'B' -> 1
    ...
    'H' -> 7
    'I' -> 8
    ...

    Args:
        state (str): A state string in with len(state) == 1
    Returns:
        int: The index of the state
    """
    return ord(state) - 65

@max_allowed_loops(3)
@no_imports
def to_matrix(transition: Dict[str, str]) -> np.ndarray:
    """
    Converts a transition dictionary into a transition matrix. The first row
    represents the probability of moving from the first state to every state.

    If the state dict is irreflexive (we cannot go from one state to the same
    state) the sum of the diagonal is 0.

    The sum of each row should be 1.

    All the elements in the matrix are values in [0-1].

    Args:
        transition (Dict[str, str]): A dictionary describing the possible
            transitions from each state.

    Returns:
        np.ndarray: The transition matrix (ndim=2) that represents the same
        (uniform) transitions as the transition dict
    """
    n = len(transition)
    mat = np.zeros((n, n))
    visited_states = simulate(transition)
    initial_state = 'A'
    for i in range(1, len(visited_states)):
        final_state = visited_states[i]
        mat[state_string_to_index(initial_state)][state_string_to_index(final_state)] += 1
        initial_state = final_state
    row_sum = np.sum(mat, axis=1)
    row_sum_as_column = row_sum.reshape(-1,1)
    return mat/row_sum_as_column

@max_allowed_loops(3)
@no_imports
def build_transition_matrix(transition: Dict[str, str]) -> np.ndarray:
    """
    Builds a transition matrix from a transition dictionary, similarly to
    `to_matrix` function. However, this function does not create a uniform
    distribution among the following states.

    If the the next valid states are two then the distribution is uniform.

    If the the next valid states are three, then moving vertically should have a
    50% chance and moving left twice as much as moving right.

    Like in the `to_matrix` function the sum of each row should be 1.

    Args:
        transition (Dict[str,str]) A dictionary describing the possible
            transitions from each state.
    Returns:
        np.ndarray: A transition matrix
    """
    # YOUR CODE HERE
    n = len(transition)
    mat = np.zeros((n, n))
    for key, values in transition.items():
        row_index = state_string_to_index(key)
        for value in values:
            column_index = state_string_to_index(value)
            if len(values) == 2:
                mat[row_index][column_index] = 0.5
            elif len(values) == 3:
                location = row_index - column_index
                chance = 0.5
                if location == 1:
                    chance = 0.5 * 2 / 3
                elif location == -1:
                    chance = 0.5 / 3
                mat[row_index][column_index] = chance
    return mat

@max_allowed_loops(1)
@no_imports
def simulate_1000(transition: Dict[str, str]) -> np.ndarray:
    """
    Simulates 1000 particles for 500 time steps, in order to approximate
    the stationary distribution

    Args:
        transition (Dict[str, str]): A transition dict, that will be
        converted into a transition matrix using the
        `build_transition_matrix` function
    Returns:
        np.ndarray: The estimated stationary distribution vector (ndim=1)

    """
    num_steps = 5000
    new_T = build_transition_matrix(T)
    pad_shape = ((0, 0), (1, 0))
    P = np.pad(new_T, pad_shape, mode='constant', constant_values=0)
    X = utils.getinitialstate()
    for i in range(num_steps):
        X = utils.mcstep(X, P, i)
    return X.mean(axis=0)


if __name__ == '__main__':
    # List of states
    S = list("ABCDEFGH")

    # Dictionary of transitions
    T = {
        "A": "BE",
        "B": "AFC",
        "C": "BGD",
        "D": "CH",
        "E": "AF",
        "F": "EBG",
        "G": "FCH",
        "H": "GD",
    }

    stationary_distribution = simulate_1000(T)
    print(stationary_distribution.shape)
    t.assertIsInstance(stationary_distribution, np.ndarray)
    t.assertEqual(stationary_distribution.shape, (8,))
    np.testing.assert_allclose(np.sum(stationary_distribution), 1)


