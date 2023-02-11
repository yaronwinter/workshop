# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:08:06 2020

@author: YaronWinter
"""

import  numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from utils import gen_utils
from utils import config as params

BOW_TYPE = 'bow'

def read_folder(folder_name: str, label: str) -> tuple:
    texts = []
    labels = []
    
    texts_folder = Path(folder_name).rglob('*.txt')
    files = [x for x in texts_folder]
    
    for filename in tqdm(files):
        text = ''
        f = open(filename, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        
        for line in lines:
            line = gen_utils.tokenize_text(line)
            if len(text) > 0:
                text += ' '
            text += line
            
        texts.append(text)
        labels.append(label)
        
    return texts, labels


def folders_to_csv(folder_name: str, csv_name: str, config: dict):
    print('folders_to_csv: dir=' + folder_name + ', csv=' + csv_name)
    texts = []
    labels = []
    
    print('read neg folder')
    t, l = read_folder(folder_name + 'neg/', 0)
    texts += t
    labels += l
    
    print('read pos folder')
    t, l = read_folder(folder_name + 'pos/', 1)
    texts += t
    labels += l
    
    print('sample size: #t=' + str(len(texts)) + ', #l=' + str(len(labels)))
    
    print('Generate a dataframe')
    data = {config[params.TEXT_COL]: texts, config[params.LABEL_COL]: labels}
    sample_df = pd.DataFrame(data)

    print('Shuffle the dataframe')
    sample_df = sample_df.sample(frac = 1)
    sample_df.index = pd.RangeIndex(start=0, stop=len(sample_df), step=1)

    print('Generate the csv')
    sample_df.to_csv(csv_name, index=False)
    print('Done.')
    
    
def generate_sample(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict) -> tuple:
    feature_type = config[params.TRADITIONAL_FEATURE_TYPE]
    max_features = config[params.TRADITIONAL_MAX_FEATURES]
    print('check feature type (' + feature_type + ')')
    if feature_type == params.TRADITIONA_BOW:
        print('Set counter vectorizer')
        cv=CountVectorizer(max_features=max_features ,binary=True)
    elif feature_type == params.TRADITIONAL_TFIDF:
        print('Set tfidf vectorizer')
        cv=TfidfVectorizer(max_features=max_features,use_idf=True)
    else:
        raise ValueError("Illegal feature type: " + feature_type)
    
    print('extract representative lexicon')
    cv_train=cv.fit_transform(train_df[config[params.TEXT_COL]])
    print('\tcv shape: <' + str(cv_train.shape[0]) + ', ' + str(cv_train.shape[1]) + '>')
    
    print('convert cv train into dence matrix and concatente the labels')
    X_train = cv_train.todense()
    y_train = train_df[config[params.LABEL_COL]].values
    print('\tX_train shape: <' + str(X_train.shape[0]) + ', ' + str(X_train.shape[1]) + '>')
    print('\ty_train shape: ' + str(y_train.shape[0]))
    
    print('Transform the test set into matrix')
    cv_test=cv.transform(test_df[config[params.TEXT_COL]])
    X_test = cv_test.todense()
    y_test = test_df[config[params.LABEL_COL]].values
    print('\tX_test shape: <' + str(X_test.shape[0]) + ', ' + str(X_test.shape[1]) + '>')
    print('\ty_test shape: ' + str(y_test.shape[0]))
    print('Done.')
    return X_train, y_train, X_test, y_test

def features_target_split(X: np.ndarray) -> tuple:
    print('X shapes: ' + str(X.shape[0]) + ', ' + str(X.shape[1]))
    y = X[:, -1]
    y = np.asarray(y, dtype=int)
    X = X[:, :-1]
    return X, y

def extract_ngrams(train_df: pd.DataFrame, config: dict, dim: int) -> dict:
    cv = CountVectorizer(max_features=config[params.TRADITIONAL_MAX_FEATURES], ngram_range=(dim,dim))
    cv.fit(train_df[config[params.TEXT_COL]])
    return cv.vocabulary_
    

def ngrams_sample(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict) -> tuple:
    features = {}
    print('extract ngrams')
    for i in range(3):
        features.update(extract_ngrams(train_df, config, i+1))
    print('#ngrams = ' + str(len(features)))
    
    ngrams = {}
    for word in features.keys():
        ngrams[word] = len(ngrams)
        
    tfidf = TfidfVectorizer(vocabulary=ngrams, ngram_range=(1,3))
    tf_train = tfidf.fit_transform(train_df[config[params.TEXT_COL]])
    X_train = tf_train.todense()
    y_train = train_df[config[params.LABEL_COL]].values
    
    tf_test = tfidf.transform(test_df[config[params.TEXT_COL]])
    X_test = tf_test.todense()
    y_test = test_df[config[params.LABEL_COL]].values
    
    print('X_train: ' + str(X_train.shape))
    print('X_test: ' + str(X_test.shape))
    print('y_train: ' + str(y_train.shape))
    print('y_test: ' + str(y_test.shape))
    
    return X_train, y_train, X_test, y_test

def get_sample_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict, unigrams_only: bool) -> tuple:
    print('Find data type')
    if type(train_df) is np.ndarray:
        print('split samples (matrix)')
        X_train, y_train = features_target_split(train_df)
        X_test, y_test = features_target_split(test_df)
    elif unigrams_only:
        print('generate sample from a dataframe')
        X_train, y_train, X_test, y_test = generate_sample(train_df, test_df, config)
    else:
        X_train, y_train, X_test, y_test = ngrams_sample(train_df, test_df, config)
            
    print('Train shapes: ' + str(X_train.shape))
    print('Test shapes: ' + str(X_test.shape))
    X_train = np.asarray(X_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    return X_train, y_train, X_test, y_test


def run_mlp(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict, ug_only: bool):
    print('get matrices')
    X_train, y_train, X_test, y_test = get_sample_matrix(train_df, test_df, config, ug_only)
    
    print('alloc MLP object')
    mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(1024, 512), verbose=True, n_iter_no_change=5)
    
    print('train the model')
    model = mlp.fit(X_train, y_train)
    
    print('test model')
    print('score={:.3f}'.format(model.score(X_test, y_test)))

def run_svm(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict, ug_only: bool):
    print('get matrices')
    X_train, y_train, X_test, y_test = get_sample_matrix(train_df, test_df, config, ug_only)
    
    print('alloc SVM object')
    sv = LinearSVC(random_state=1, verbose=True, penalty='l2', loss='hinge', max_iter=100)

    print('train the model')
    model = sv.fit(X_train, y_train)
    
    print('test model')
    print('score={:.3f}'.format(model.score(X_test, y_test)))

def run_logreg(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict, ug_only: bool):
    print('get matrices')
    X_train, y_train, X_test, y_test = get_sample_matrix(train_df, test_df, config, ug_only)
    
    print('alloc LR object')
    lr = LogisticRegression(random_state=0, solver='liblinear', penalty='l2', verbose=2)
    
    print('train the model')
    model = lr.fit(X_train, y_train)
    
    print('test model')
    print('score={:.5f}'.format(model.score(X_test, y_test)))


def get_words_distribution(corpus_lines: list) -> dict:
    distribution = {}
    for line in corpus_lines:
        words = line.split(' ')
        for word in words:
            if word in distribution:
                distribution[word] += 1
            else:
                distribution[word] = 1
                
    return distribution
