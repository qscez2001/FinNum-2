import pandas as pd
import numpy as np
import json
from pprint import pprint
from sklearn.model_selection import train_test_split

data = json.load(open("NTCIR-2020_FinNum_training.json"))
# pprint(data)

def transform_int_to_string(path):
    features = np.genfromtxt(path, delimiter=',', skip_header=1)
    # features = np.genfromtxt('non-sequence/add_keyword_features.csv', delimiter=',', skip_header=1)
    features = features[:,:6].astype(int)
    # features = features[:,:7].astype(int)
    # print(features)
    list_of_str = []
    for row in features:
        new_str = ''
        for i in row:
            new_str = new_str + ' ' + str(i)
        list_of_str.append(new_str)
    # print(list_of_str)
    return np.asarray(list_of_str)

def preprocess():

    full_train = concat()

    # full_train = pd.read_json("NTCIR-2020_FinNum_training.json")
    # print(full_train)

    features = transform_int_to_string('non-sequence/new_features.csv').reshape(-1,1)

    # features = transform_int_to_string('non-sequence/train_data/new_features.csv').reshape(-1,1)

    X = full_train.drop(['relation'], axis=1)
    y = full_train['relation']

    X = X['tweet']

    X = X.to_numpy().reshape(-1,1)
    y = y.to_numpy()
    X = np.concatenate((X, features), axis=1)
    # print(X.shape)
    # print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # a = y.tolist().count(0)
    # b = y.tolist().count(1)
    # c = y_test.tolist().count(0)
    # d = y_test.tolist().count(1)
    # print("null model train= ", a, b, b/(a+b))
    # print("null model test= ", c, d, d/(c+d))

    # print(X_test)
    # print(y_test)

    return X_train, X_test, y_train, y_test
    # return X, y

def concat():

    train = pd.read_json("NumAttach_train.json")
    test = pd.read_json("NumAttach_test.json")

    # print(len(train))
    # print(len(test))
    full = pd.concat([train, test], axis=0, ignore_index=True)
    # print(full)
    return full

def dev_preprocess():

    full_train = pd.read_json("NTCIR-2020_FinNum_dev_v3.json")
    # print(full_train)

    features = transform_int_to_string('non-sequence/dev_data/new_features.csv').reshape(-1,1)

    X = full_train.drop(['relation'], axis=1)
    y = full_train['relation']

    X = X['tweet']
    X = X.to_numpy().reshape(-1,1)
    y = y.to_numpy()
    X = np.concatenate((X, features), axis=1)
    # print(X.shape)
    # print(y.shape)

    # a = y_train.tolist().count(0)
    # b = y_train.tolist().count(1)
    # c = y_test.tolist().count(0)
    # d = y_test.tolist().count(1)
    # print("null model train= ", a, b, b/(a+b))
    # print("null model test= ", c, d, d/(c+d))

    return X, y

def test_preprocess():

    full_train = pd.read_json("NTCIR-2020_FinNum_test(With_ANS).json")
    # print(full_train)

    features = transform_int_to_string('non-sequence/test_data/new_features.csv').reshape(-1,1)

    X = full_train.drop(['relation'], axis=1)
    y = full_train['relation']

    X = X['tweet']
    X = X.to_numpy().reshape(-1,1)
    y = y.to_numpy()
    X = np.concatenate((X, features), axis=1)
    
    # X = full_train['tweet']
    # X = X.to_numpy().reshape(-1,1)
    # X = np.concatenate((X, features), axis=1)

    # all guess 1
    # y = [1] * 2109
    # y = np.asarray(y)

    # a = y_train.tolist().count(0)
    # b = y_train.tolist().count(1)
    # c = y_test.tolist().count(0)
    # d = y_test.tolist().count(1)
    # print("null model train= ", a, b, b/(a+b))
    # print("null model test= ", c, d, d/(c+d))

    return X, y

# transform_int_to_string()
preprocess()
# dev_preprocess()
# test_preprocess()