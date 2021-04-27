from hashlib import sha1
import numpy as np
import pandas as pd
import re

def test1_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert isinstance(answer, list), "Make sure your answer is of type list."
    answer = [x.lower() for x in answer]
    assert len(answer) == 10, "The length of your answer is incorrect"
    assert sha1(str(answer).encode('utf8')).hexdigest() == 'd5880a54665f6427216fda521c677355c339ea90', "Your answer is incorrect. Are you predicting based on the column `SupportiveRUS`"
    return('success') 

def test1_2(answer):
    assert sha1(str(round(answer)).encode('utf8')).hexdigest() == "356a192b7913b04c54574d18c28d46e6395428ab", "Your answer is incorrect. Are you counting the cases correctly?"
    return("Success")


def test1_3(answer1, answer2):
    assert not answer1 is None, "The 'X' variable does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "The 'y' variable does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (10, 3), "The size of 'X' is incorrect. Are you dropping the target column?"
    assert 'target' not in answer1.columns, "Make sure you are not selecting the target column as part of 'X'"
    assert answer2.shape == (10,), "The size of 'y' is incorrect. Are you only selecting the target column?"
    
def test1_4(answer):
    assert answer.get_params()['criterion'] == 'gini', "Are you initializing a decision tree classifier properly?"
    assert answer.get_params()['splitter'] == 'best', "Are you initializing a decision tree classifier properly?"
    return("Success")

def test1_5(answer):
    assert sha1(str(answer + 1).encode('utf8')).hexdigest() == "936931368287d72a5bda62a8a3e0d2ed6638fa8f", "The score is incorrect. Are you fitting the model correctly?"
    return("Success")

def test1_7(answer):
    assert not answer is None, "The 'X' variable does not exist. Have you passed in the correct variable?"
    assert 'target' not in answer.columns, "Make sure you are not selecting the target column as part of 'X'"
    assert answer.shape == (3, 3), "The size of 'X' is incorrect. Are you dropping the target column?"
    return("Success")
    
def test1_8(answer):
    assert len(answer) == 3, "This output array should have 3 elements in it."
    assert sha1(str(answer.tolist()).encode('utf8')).hexdigest() == "6d45e4baeaae034c0eb1f255f3f6e7c7a48ead02", "The score is incorrect. Are you fitting the model correctly?"
    return("Success")

def test2_1(answer1, answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (1613, 16), "The dimensions of training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (404, 16), "The dimensions of the test set is incorrect. Are you splitting correctly?"
    assert list(answer1.loc[32]) == [0.0358, 0.9590000000000001, 213000, 0.598, 0.0, 
                                     8, 0.358, -5.534, 1, 0.0713, 127.029, 
                                     4.0, 0.424, 1, 'Best Friend', 'Young Thug'], "Make sure you are setting your random state to 77."
    assert list(answer2.loc[42]) == [0.00156, 0.624, 183000, 0.792, 1.54e-06, 10,
                                    0.0772, -5.33, 0, 0.0473, 139.94, 4.0, 0.318,
                                    1, 'Versace Python', 'Riff Raff'], "Make sure you are setting your random state to 77"
    return("Success")

    