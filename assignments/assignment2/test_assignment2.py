from hashlib import sha1
import pandas as pd
import pytest
import altair
import sys
import numpy as np


def test_1_1(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (12545, 15), "The dimensions of training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (3137, 15), "The dimensions of the test set is incorrect. Are you splitting correctly?"
    assert list(answer1.loc[1285]) == [46, 'Local-gov', 121124, 'Bachelors', 13, 'Married-civ-spouse', 'Protective-serv', 'Husband',
                                       'White', 'Male', 15024, 0, 40, 'United-States', '>50K'], "Make sure you are setting your random state to 123."
    assert list(answer2.loc[4077]) == [44, 'Private', 193882, 'Bachelors', 13, 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White', 'Male', 0,
                                       0, 40, 'United-States', '>50K'], "Make sure you are setting your random state to 123"
    return("Success")

def test_1_2_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [x.lower() for x in sorted(answer)]
    assert sha1(str(answer).encode('utf8')).hexdigest() == '9b2c609d4b4ff7198bafcf7cb922e7624ddd8782', "Your answer is incorrect. Are you analysing the dataframe correctly?"
    return("Success")

def test_1_2_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [x.lower() for x in sorted(answer)]
    assert sha1(str(answer).encode('utf8')).hexdigest() == 'fa4f9a4ffa58804eb3108765beeeaf2497beb8c2', "Your answer is incorrect. Are you analysing the dataframe correctly?"
    return("Success")

def test_1_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (11, 15), "The dimensions of your solution is incorrect. Are you using the describe function?"
    assert 'std' in list(answer.index), "Your solution is missing some values. Are you using the describe function?"
    assert 'education_num' in list(answer.columns), "Your solution is missing some columns. Are you using the correct dataframe?"
    assert list(answer.iloc[7]) == [31.0, np.nan, 117789.0, np.nan, 9.0, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, 40.0, np.nan, np.nan], "Your solution is missing some values. Are you using the describe function?"
    return("Success")

def test_1_6(answer1,answer2,answer3,answer4):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (12545, 14), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (3137, 14), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert answer3.shape == (12545,), "The dimensions of the training set is incorrect. Are you splitting correctly? Are you using single brackets?"
    assert answer4.shape == (3137,), "The dimensions of the test set is incorrect. Are you splitting correctly? Are you using single brackets?"
    assert 'income' not in list(answer1.columns), "Make sure the target variable is not part of your X dataset."
    return("Success")


def test_2_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'median' in str(list(answer)[0]), "Make sure your using the simple imputer with the median strategy."
    assert str(list(answer)[1]) == 'StandardScaler()', "Make sure you are using the standard scaler for scaling of the data."
    return("Success")

def test_2_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'int' in str(list(answer)[1]), "Make sure you are specifying the data type to be int for the encoding."
    assert 'ignore' in str(list(answer)[1]), "Make sure you are using the 'ignore' method to handle unknown cases."
    assert 'most_frequent' in str(list(answer)[0]), "Make sure your using the simple imputer with the most_frequent strategy."
    return("Success")

def test_2_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert len(list(answer.transformers)) == 2, "Make sure you are including both pipelines."
    assert str(list(answer.transformers)).count('most_frequent') == 1, "Make sure you are including both pipelines."
    assert str(list(answer.transformers)).count('int') == 1, "Make sure you are including both pipelines."
    return("Success")