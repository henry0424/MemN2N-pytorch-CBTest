"""
Data util codes based on https://github.com/domluna/memn2n
"""

import os
import re
import numpy as np
from tqdm import tqdm
from time import sleep
import gc
import statistics
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_data_CBTest(data_dir, word_type, perc_dict):
    if 'data_CBTest.pkl' in os.listdir(data_dir):
        dict_data = load_obj(os.path.join(data_dir, 'data_CBTest'))
        
        min_val = dict_data['vocab'][1][int(max((1.0 - perc_dict)*len(dict_data['vocab'][1]) - 1.0, 0.0))]

        vocab = set()
        for i in dict_data['vocab'][0].keys():
            if dict_data['vocab'][0][i] >= min_val:
                vocab.add(i)

        for i in dict_data['vocab'][2]:
            vocab.add(i)

        vocab.add('|NotInDictionary|')
        vocab = sorted(vocab)
        return dict_data['train'], dict_data['test'], vocab

    lines_train = []
    sizes_train = []
    lines_test = []
    sizes_test = []
    for i in os.listdir(data_dir):
        dir_ = os.path.join(data_dir, i)
        if 'cbtest_' in dir_ and word_type in dir_:
            if word_type + '_' + 'train' in dir_:
                print("Loading data from:", i)
                f = open(dir_, 'r')
                lines_train =  f.readlines()
                f.close()
                sizes_pos_train, vocab_train = get_data_CBTest(lines_train)

            if word_type + '_' + 'test' in dir_:
                f = open(dir_, 'r')
                lines_test =  f.readlines()
                f.close()
                sizes_pos_test, vocab_test = get_data_CBTest(lines_test)

    for i in vocab_test[0].keys():
        if i in vocab_train[0]:
            vocab_train[0][i] += vocab_test[0][i]
        else:
            vocab_train[0][i] = vocab_test[0][i]

    vals = []
    for i in vocab_train[0].keys():
        vals.append(vocab_train[0][i])
    vals = sorted(vals)

    min_val = vals[int(max((1.0 - perc_dict)*len(vals) - 1.0, 0.0))]

    vocab = set()
    for i in vocab_train[0].keys():
        if vocab_train[0][i] >= min_val:
            vocab.add(i)

    for i in (vocab_train[1] + vocab_test[1]):
        vocab.add(i)

    vocab.add('|NotInDictionary|')
    vocab = sorted(vocab)

    #val = ord(item.lower()[0])
    #if val > 96 and val < 123:
        #vocab.add(item.lower())'''

    if not 'data_CBTest' in os.listdir(data_dir):
        dict_data = {'train': (lines_train, sizes_pos_train), 'test': (lines_test, sizes_pos_test), 'vocab': (vocab_train[0], vals, vocab_train[1] + vocab_test[1])}
        save_obj(dict_data, os.path.join(data_dir, 'data_CBTest'))

    return (lines_train, sizes_pos_train), (lines_test, sizes_pos_test), vocab


def get_data_CBTest(lines):
    max_q_size = 0
    max_op_size = 0
    max_se_size = 0
    examples_pos = []
    vocab = {}
    vocab_q_a = []
    for i in tqdm(range(int(len(lines)/2))):
        flag_21 = True
        tokens = tokenize(lines[i].lower())
        if (len(tokens) > 0):
            #for token in tokens:
                #vocab.add(token)
            index = int(tokens.pop(0))
            if index == 1:
                examples_pos.append(i)
                q = []
                op = []
            if index < 21:
                for token in tokens:
                    if token in vocab:
                        vocab[token.lower()] += 1
                    else:
                        vocab[token.lower()] = 1

                if len(tokens) > max_se_size:
                    max_se_size = len(tokens)
            if index == 21:
                flag_21 = False
                separators = [i for i, e in enumerate(tokens) if e == '|']
                for t in range(separators[0]-2):
                    q.append(tokens[t])
                op.append(tokens[separators[0]-1])
                for s in separators:
                    op.append(tokens[s+1])

                if len(q) > max_q_size:
                    max_q_size = len(q)
                if len(op) > max_op_size:
                    max_op_size = len(op)
                
                vocab_q_a.append(tokens[separators[0]-2])

    if flag_21:
        examples_pos.pop(len(examples_pos)-1)

    return (examples_pos, max_q_size, max_se_size, max_op_size), (vocab, vocab_q_a)


def vectorize_data_CBTest(idx, lines, pos, word_idx, sentence_size, memory_size):
    data = []
    i = pos[idx]
    flag = True
    while flag:
        tokens = tokenize(lines[i].lower())
        if (len(tokens) > 0):
            index = int(tokens.pop(0))
            if index == 1:
                story = []
                q = []
                a = []
                op = []
            if index < 21:
                story.append(tokens)
            if index == 21:
                separators = [i for i, e in enumerate(tokens) if e == '|']
                a.append(tokens[separators[0]-2])
                for t in range(separators[0]-2):
                    q.append(tokens[t])
                
                op.append(tokens[separators[0]-1])
                for s in separators:
                    op.append(tokens[s+1])
                flag = False
        i += 1   

    data.append((story, q, a, op))
    return vectorize_data(data, word_idx, sentence_size, memory_size)

def load_task(data_dir, task_id, only_supporting=False):
    """
    Load the nth task. There are 20 tasks in total.
    Returns a tuple containing the training and testing data for the task.
    """
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = "qa{}_".format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


def tokenize(sent):
    """
    Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    return [x.strip() for x in re.split("(\W+)?", sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    """
    Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    """
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(" ", 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if "\t" in line: # question
            q, a, supporting = line.split("\t")
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a, []))
            story.append("")
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    """
    Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length
    tokens will be discarded.
    """
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.
    If a sentence length < sentence_size, the sentence will be padded with 0's.
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    The answer array is returned as a one-hot encoding.
    """

    S, Q, A, OP = [], [], [], []
    for story, query, answer, options in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w in word_idx else word_idx['|NotInDictionary|'] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else word_idx['|NotInDictionary|'] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            if a in word_idx:
                y[word_idx[a]] = 1
            else:
                y[word_idx['|NotInDictionary|']] = 1

        opt = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        if len(options) > 0:
            for op_ in options:
                if op_ in word_idx:
                    opt[word_idx[op_]] = 1
                else:
                    opt[word_idx['|NotInDictionary|']] = 1

        S.append(ss); Q.append(q); A.append(y); OP.append(opt)
    return np.array(S), np.array(Q), np.array(A), np.array(OP)
