from hashlib import sha1
import pandas as pd
import pytest
import altair
import sys
import numpy as np


def test_1_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.lower().encode('utf8')).hexdigest() == "f8031b07549bd8502c648d8ee79292d2dedf6fc7", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_2_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "1469842b4307d36cccb487dc989f21016daadbcc", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_2_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "b8021a7f46f61704c0c650390eccbe68ced52e70", "Your answer is incorrect. Please try again."
    return("Success")


def test_1_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.lower().encode('utf8')).hexdigest() == "4d8235d83eeac8c909d966beac6de30bdaab6012", "Your answer is incorrect. Are you examining the models?"
    return("Success")

def test_1_4_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "959041c33324191578173f21a203e93aa2c2b431", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_4_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "856b62aa687c0aa2b0deb2980b3dd887b3c93ff8", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_4_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "7601e3d5e55dd14143b127b99c174760a573099e", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_5_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "e8dc057d3346e56aed7cf252185dbe1fa6454411", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_5_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "856b62aa687c0aa2b0deb2980b3dd887b3c93ff8", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_5_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "eccfe3795ea096a036ba46a7c1acc2d38f016506", "Your answer is incorrect. Please try again."
    return("Success")

def test_2_1(answer1,answer2,answer3,answer4):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert str(type(answer1)) == "<class 'pandas.core.series.Series'>", "Make sure your passing in a pandas series object."
    assert str(type(answer2)) == "<class 'pandas.core.series.Series'>", "Make sure your passing in a pandas series object."
    assert str(type(answer3)) == "<class 'pandas.core.series.Series'>", "Make sure your passing in a pandas series object."
    assert str(type(answer4)) == "<class 'pandas.core.series.Series'>", "Make sure your passing in a pandas series object."
    assert answer1.shape == (16000,), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (16000,), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert answer3.shape == (4000,), "The dimensions of the training set is incorrect. Are you splitting correctly? Are you using single brackets?"
    assert answer3.shape == (4000,), "The dimensions of the training set is incorrect. Are you splitting correctly? Are you using single brackets?"
    return("Success")

def test_2_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'test_accuracy' in answer.columns, "test_accuracy is not in your dataframe did you call `accuracy` in the scoring argument?"
    assert 'train_accuracy' in answer.columns, "train_accuracy is not in your dataframe. Did you call `accuracy` in the scoring argument and did you specify return_train_score=True?"
    assert 'test_f1' in answer.columns, "test_f1 is not in your dataframe. Did you call `f1` in the scoring argument?"
    assert 'train_f1' in answer.columns, "train_f1 is not in your dataframe. Did you call `f1` in the scoring argument and did you specify return_train_score=True?"
    assert 'test_recall' in answer.columns, "test_recall is not in your dataframe. Did you call `recall` in the scoring argument?"
    assert 'train_recall' in answer.columns, "train_recall is not in your dataframe. Did you call `recall` in the scoring argument and did you specify return_train_score=True?"
    assert 'test_precision' in answer.columns, "test_precision is not in your dataframe. Did you call `precision` in the scoring argument?"
    assert 'train_precision' in answer.columns, "train_precision is not in your dataframe. Did you call `precision` in the scoring argument and did you specify return_train_score=True?"
    return("Success")

def test_3_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (4000, 5), "The dimensions of your dataframe is incorrect."
    assert sorted(list(answer.columns)) == ['neg_label_prob', 'pos_label_prob', 'predicted_y', 'review', 'true_label'], "ncorrect column names. Make sure you are specifying the required columns."
    return("Success")
