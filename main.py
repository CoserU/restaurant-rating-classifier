import pandas as pd
import numpy as np
from collections import Counter

df_tr = pd.read_csv('F:/Courses/Machine Learning/homework/hw2/reviews_tr.csv', nrows=500000)
df_te = pd.read_csv('F:/Courses/Machine Learning/homework/hw2/reviews_te.csv')
print(df_tr.shape, df_te.shape)


def word_frequency(train_data):
    dic = {}
    w_init = {}
    for index, row in train_data.iterrows():
        for word in row['text'].split():
            dic[word] = dic.get(word, 0) + 1
    for x in dic:
        w_init[x] = 0
    print('w init dimension is:', len(w_init))
    return w_init, dic


def bigram_frequency(train_data):
    w_init = {}
    for index, row in train_data.iterrows():
        words = row['text'].split()
        for i in range(len(words) - 1):
            pair = words[i] + ' ' + words[i + 1]
            if pair not in w_init:
                w_init[pair] = 0
    print('w init dimension is:', len(w_init))
    return w_init


def unigram(text, *args):
    word_count = dict(Counter(text.split()))
    word_count['data_lift'] = 1
    return word_count


def tfidf(text, dic, D):
    word_count = dict(Counter(text.split()))
    for x in word_count:
        if x in dic:
            word_count[x] = word_count[x] * np.log10(D / dic[x])
        else:
            return 'skip'
    word_count['data_lift'] = 1
    return word_count


def bigram(text, *args):
    words = text.split()
    pair_count = {}
    for i in range(len(words) - 1):
        pair = words[i] + ' ' + words[i + 1]
        pair_count[pair] = pair_count.get(pair, 0) + 1
    return pair_count


def need_update(w, tf, label):
    dot = sum(w[k] * tf[k] for k in tf if k in w)
    if (dot > 0 and label == 1) or (dot <= 0 and label == 0):
        return False
    else:
        return True


def update(w, tf, label):
    if label == 1:
        for x in tf:
            w[x] = w[x] + tf[x]
    else:
        for x in tf:
            w[x] = w[x] - tf[x]


def update_2(w, tf, label, w_sum, index, D):
    if label == 1:
        for x in tf:
            w[x] = w[x] + tf[x]
            w_sum[x] = w_sum[x] + tf[x] * (D - index)
    else:
        for x in tf:
            w[x] = w[x] - tf[x]
            w_sum[x] = w_sum[x] - tf[x] * (D - index)


def mean_w(w, denominator):
    for x in w:
        w[x] = w[x] / denominator
    return w


def train_pass(train_data, mode='unigram', extract_func=unigram):
    print('train start')
    D = len(train_data)

    if mode == 'bigram':
        w_init = bigram_frequency(train_data)
    else:
        w_init, dic = word_frequency(train_data)

    train_data.sample(frac=1)
    w_init['data_lift'] = 0
    w = w_init
    for index, row in train_data.iterrows():
        if mode == 'bigram':
            tf = extract_func(row['text'])
        else:
            tf = extract_func(row['text'], dic, D)
        label = row['label']
        if need_update(w, tf, label):
            update(w, tf, label)
        if index % 10000 == 0:
            print(index, 'first pass')

    train_data.sample(frac=1)
    w_sum = {}
    for x in w:
        w_sum[x] = w[x] * D
    for index, row in train_data.iterrows():
        if mode == 'bigram':
            tf = extract_func(row['text'])
        else:
            tf = extract_func(row['text'], dic, D)
        label = row['label']
        if need_update(w, tf, label):
            update_2(w, tf, label, w_sum, index, D)
        if index % 10000 == 0:
            print(index, 'second pass')

    w_final = mean_w(w_sum, D + 1)

    if mode == 'unigram' or mode == 'bigram':
        return w_final
    elif mode == 'tfidf':
        return w_final, dic, D


def evaluate(test_data, mode='unigram', extract_func=unigram, *args):
    print('test start!')
    accurate_count = 0
    all_count = len(test_data)
    w = args[0]
    for index, row in test_data.iterrows():
        tf = extract_func(row['text'], *args[1:])
        if tf == 'skip':
            all_count -= 1
            continue
        label = row['label']
        if not need_update(w, tf, label):
            accurate_count += 1
        if index % 10000 == 0:
            print(index, 'test pass')
    print('test numbers:', all_count)
    return accurate_count / all_count


mode = 'unigram'
# mode = 'tfidf'
# mode = 'bigram'

if mode == 'unigram':
    res = train_pass(df_tr, mode, unigram)
    accuracy = evaluate(df_te, mode, unigram, res)
elif mode == 'tfidf':
    res = train_pass(df_tr, mode, tfidf)
    accuracy = evaluate(df_te, mode, tfidf, *res)
elif mode == 'bigram':
    res = train_pass(df_tr, mode, bigram)
    accuracy = evaluate(df_te, mode, bigram, res)
print(mode, 'accuracy is', accuracy)


if mode == 'unigram' or mode == 'bigram':
    w_final = res
elif mode == 'tfidf':
    w_final = res[0]
w_sort = sorted(w_final, key=w_final.get)
high_words = w_sort[-10:]
low_wods = w_sort[:10]
print(high_words)
print(low_wods)
