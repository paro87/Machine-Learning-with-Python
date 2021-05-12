
import unittest; t = unittest.TestCase()
def decision(x: tuple) -> str:
    '''
        This function implements the decision tree represented in the above image. As input the function
        receives a tuple with three values that represent some information about a patient.
        Args:
            x (tuple): Input tuple containing exactly three values. The first element represents
            a patient is a smoker this value will be 'yes'. All other values represent that
            the patient is not a smoker. The second element represents the age of a patient
            in years as an integer. The last element represents the diet of a patient.
            If a patient has a good diet this string will be 'good'. All other
                values represent that the patient has a poor diet.
        Returns:
            string: A string that has either the value 'more' or 'less'.
            No other return value is valid.

    '''
    if x[0] == 'yes':
        if x[1] < 29.5:
            result = 'less'
        else:
            result = 'more'
    else:
        if x[2] == 'good':
            result = 'less'
        else:
            result = 'more'
    return result

def parse_line_test(line: str) -> tuple:
    '''
        Takes a line from the file, including a newline, and parses it into a patient tuple

        Args:
            line (str): A line from the `health-test.txt` file
        Returns:
            tuple: A tuple representing a patient
    '''
    lst = [element for element in line.split(',')]
    lst = (lst[0], int(lst[1]), lst[2].strip())

    tup = tuple(lst)
    return tup

def gettest() -> list:
    '''
        Opens the `health-test.txt` file and parses it
        into a list of patient tuples. You are encouraged to use
        the `parse_line_test` function but it is not necessary to do so.

        Returns:
            list: A list of patient tuples
    '''
    with open('./health-test.txt', 'r') as f:
        data = []
        data.extend(parse_line_test(x) for x in f)
    return data

def evaluate_testset(dataset: list) -> float:
    '''
        Calculates the percentage of datapoints for which the
        decision function evaluates to `'more'` for a given dataset

        Args:
            dataset (list): A list of patient tuples

        Returns:
            float: The percentage of data points which are evaluated to `'more'`
    '''
    count = sum([decision(x) == 'more' for x in dataset])
    return count/len(dataset)

def parse_line_train(line: str) -> tuple:
    '''
        This function works similarly to the `parse_line_test` function.
        It parses a line of the `health-train.txt` file into a tuple that
        contains a patient tuple and a label.

        Args:
            line (str): A line from the `health-train.txt`

        Returns:
            tuple: A tuple that contains a patient tuple and a label as a string
    '''
    lst = [element for element in line.split(',')]
    tpl = (lst[0], int(lst[1]), lst[2]), (lst[3].strip())
    return tpl

def gettrain() -> list:
    '''
        Opens the `health-train.txt` file and parses it into
        a list of patient tuples accompanied by their respective label.

        Returns:
            list: A list of tuples comprised of a patient tuple and a label
    '''
    with open('./health-train.txt', 'r') as f:
        data = []
        data.extend(parse_line_train(x) for x in f)
    return data

def distance(a: tuple, b: tuple) -> float:
    '''
        Calculates the distance between two data points (patient tuples)
        Args:
            a, b (tuple): Two patient tuples for which we want to calculate the distance
        Returns:
            float: The distance between a, b according to the above formula
    '''
    distance = (a[0] != b[0]) + ((a[1] - b[1]) / 50.0) ** 2 + (a[2] != b[2])
    return distance

def neighbor(x: tuple, trainset: list) -> str:
    '''
        Returns the label of the nearest data point in trainset to x.
        If x is `('no', 30, 'good')` and the nearest data point in trainset
        is `('no', 31, 'good')` with label `'less'` then `'less'` will be returned

        Args:
            x (tuple): The data point for which we want to find the nearest neighbor
            trainset (list): A list of tuples with patient tuples and a label

        Returns:
            str: The label of the nearest data point in the trainset. Can only be 'more' or 'less'
    '''
    minimum = 1
    lab = ''
    for y in trainset:
        dis = distance(x, y[0])
        if dis < minimum:
            minimum = dis
            lab = y[1]
    return lab

def compare_classifiers(trainset: list, testset: list) -> float:
    '''
        This function compares the two classification methods by finding all the datapoints for which
        the methods disagree.

        Args:
            trainset (list): The training set used in the nearest neighbour classfier.
            testset (list): Contains the elements which will be used to compare the
                decision tree and nearest neighbor classification methods.

        Returns:
            list: A list containing all the data points which yield different results for the two
                classification methods.
            float: The percentage of data points for which the two methods disagree.

    '''
    disagree = []
    count = 0
    for x in testset:
        neigh = neighbor(x, trainset)
        dec = decision(x)
        if neigh != dec:
            print(type(x))
            disagree.append(tuple([x]))
            count += 1
    percentage = count / len(disagree)
    return disagree, percentage

def parse_line_train_num(line: str) -> tuple:
    '''
        Takes a line from the file `health-train.txt`, including a newline,
        and parses it into a numerical patient tuple

        Args:
            line (str): A line from the `health-test.txt` file
        Returns:
            tuple: A numerical patient
    '''
    lst = [element for element in line.split(',')]
    value1 = 1.0 if lst[0] == 'yes' else 0.0
    value2 = float(lst[1])
    value3 = 1.0 if lst[2] == 'good' else 0.0
    tup = (value1, value2, value3), lst[3].strip()
    return tup


def gettrain_num() -> list:
    '''
    Parses the `health-train.txt` file into numerical patient tuples

    Returns:
        list: A list of tuples containing numerical patient tuples and their labels
    '''
    data = list()
    with open('./health-train.txt', 'r') as file:
        data.extend(parse_line_train_num(line) for line in file)
        # data.append(parse_line_train_num(line) for line in file)
    return data


def distance_num(a: tuple, b: tuple) -> float:
    '''
    Calculates the distance between two data points (numerical patient tuples)
    Args:
        a, b (tuple): Two numerical patient tuples for which
            we want to calculate the distance
    Returns:
        float: The distance between a, b according to the above formula
    '''
    distance = (a[0] - b[0]) ** 2 + ((a[1] - b[1]) / 50.0) ** 2 + (a[2] - b[2]) ** 2
    return distance


class NearestMeanClassifier:
    '''
        Represents a NearestMeanClassifier.

        When an instance is trained a dataset is provided and the mean for each class is calculated.
        During prediction the instance compares the datapoint to each class mean (not all datapoints)
        and returns the label of the class mean to which the datapoint is closest to.

        Instance Attributes:
            more (tuple): A tuple representing the mean of every 'more' data-point in the dataset
            less (tuple): A tuple representing the mean of every 'less' data-point in the dataset
    '''
    def __init__(self):
        self.more = None
        self.less = None

    def train(self, dataset: list):
        '''
               Calculates the class means for a given dataset and stores
               them in instance attributes more, less.
               Args:
                   dataset (list): A list of tuples each of them containing a numerical patient tuple and its label
               Returns:
                   self
        '''
        smokerMore =0
        smokerLess = 0
        ageMore=0
        ageLess = 0
        dietMore =0
        dietLess = 0
        countMore = 0
        countLess = 0
        for data in dataset:
            if data[1] == 'more':
                smokerMore+=data[0][0]
                ageMore+=data[0][1]
                dietMore+=data[0][2]
                countMore+=1
            else:
                smokerLess += data[0][0]
                ageLess += data[0][1]
                dietLess += data[0][2]
                countLess +=1
        self.more = (smokerMore/countMore, ageMore/countMore, dietMore/countMore)
        self.less = (smokerLess/countLess, ageLess/countLess, dietLess/countLess)
        # YOUR CODE HERE

        return self

    def predict(self, x: tuple) -> str:
        '''
           Returns a prediction/label for numeric patient tuple x.
            The classifier compares the given data point to the mean
            class tuples of each class and returns the label of the
            class to which x is the closest to (according to our
            distance function).

            Args:
                x (tuple): A numerical patient tuple for which we want a prediction

            Returns:
                str: The predicted label
        '''
        distMore = distance_num(x, self.more)
        distLess = distance_num(x, self.less)
        if distLess < distMore:
            return 'less'
        else:
            return 'more'

    def __str__(self):
        return repr(self)

    def __repr__(self):
        more = tuple(round(m, 3) for m in self.more) if self.more else self.more
        less = tuple(round(l, 3) for l in self.less) if self.less else self.less
        return f'NearestMeanClassfier(more: {more}, less: {less})'


def build_and_train(trainset_num: list) -> NearestMeanClassifier:
    '''
        Instantiates the `NearestMeanClassifier`, trains it on the
        `trainset_num` dataset and returns it.

        Args:
            trainset_num (list): A list of numerical patient tuples with their respective labels

        Returns:
            NearestMeanClassifier: A NearestMeanClassifier trained on `trainset_num`
    '''
    nearestMeanClassifier = NearestMeanClassifier()
    nearestMeanClassifier.train(trainset_num)
    return nearestMeanClassifier

def gettest_num() -> list:
    '''
        Parses the `health-test.txt` file into numerical patient tuples

        Returns:
            list: A list containing numerical patient tuples, loaded from `health-test.txt`
    '''
    data = list()
    with open('./health-test.txt', 'r') as file:
        for line in file:
            lst = [element for element in line.split(',')]
            value1 = 1.0 if lst[0] == 'yes' else 0.0
            value2 = float(lst[1])
            value3 = 1.0 if lst[2].strip() == 'good' else 0.0
            tup = (value1, value2, value3)
            data.append(tup)
    return data


def predict_test() -> list:
    '''
        Classifies the test set using all the methods that were developed in this exercise sheet,
        namely `decision`, `neighbor` and `NearestMeanClassifier`

        Returns:
            list: a list of patient tuples containing all the datapoints that were classfied
                the same by all methods, as well as the predicted labels

        Example:
        # >>> predict_test()
        [(('yes', 22, 'poor'), 'less'),
         (('yes', 21, 'poor'), 'less'),
         (('no', 31, 'good'), 'more')]

        This example only shows how the output should look like. The values in the tuples
        are completely made up
    '''
    dataList1 = gettest()
    dataList2 = gettest_num()
    agreed_samples = list()
    for (data1, data2) in zip(dataList1, dataList2):
        lab1 = decision(data1)
        lab2 = neighbor(data1, gettrain())
        lab3 = build_and_train(gettrain_num()).predict(data2)
        if lab1 == lab2 == lab3:
            agreed_samples.append((data1, lab1))
    return agreed_samples

if __name__ == '__main__':
    same_predictions = predict_test()


