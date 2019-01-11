import os
import random
from itertools import chain
import numpy as np
import torch
import torch.utils.data as data
from data_utils import load_task, vectorize_data, load_data_CBTest, vectorize_data_CBTest, tokenize
from six.moves import range, reduce


class bAbIDataset(data.Dataset):
    def __init__(self, dataset_dir, task_id=1, memory_size=50, train=True):
        self.train = train
        self.task_id = task_id
        self.dataset_dir = dataset_dir

        train_data, test_data = load_task(self.dataset_dir, task_id)
        data = train_data + test_data
        
        self.vocab = set()
        for story, query, answer, _ in data:
            self.vocab = self.vocab | set(list(chain.from_iterable(story))+query+answer)
        self.vocab.add('|NotInDictionary|')
        self.vocab = sorted(self.vocab)
        word_idx = dict((word, i+1) for i, word in enumerate(self.vocab))

        self.max_story_size = max([len(story) for story, _, _, _ in data])
        self.query_size = max([len(query) for _, query, _, _ in data])
        self.sentence_size = max([len(row) for row in \
            chain.from_iterable([story for story, _, _, _ in data])])
        self.memory_size = min(memory_size, self.max_story_size)

        # Add time words/indexes
        for i in range(self.memory_size):
            word_idx["time{}".format(i+1)] = "time{}".format(i+1)

        self.num_vocab = len(word_idx) + 1 # +1 for nil word
        self.sentence_size = max(self.query_size, self.sentence_size) # for the position
        self.sentence_size += 1  # +1 for time words
        self.word_idx = word_idx

        self.mean_story_size = int(np.mean([ len(s) for s, _, _, _ in data ]))

        if train:
            story, query, answer, _ = vectorize_data(train_data, self.word_idx,
                self.sentence_size, self.memory_size)
        else:
            story, query, answer, _ = vectorize_data(test_data, self.word_idx,
                self.sentence_size, self.memory_size)

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        self.data_answer = torch.LongTensor(np.argmax(answer, axis=1))

    def __getitem__(self, idx):
        options = torch.FloatTensor(torch.ones(len(self.word_idx)+1))
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx], options

    def __len__(self):
        return len(self.data_story)

class CBTestDataset(data.Dataset):
    def __init__(self, dataset_dir, word_type='P', memory_size=50, train=True, perc_dict=1.0):
        self.train = train
        self.word_type = word_type
        self.dataset_dir = dataset_dir

        train_data, test_data, self.vocab = load_data_CBTest(self.dataset_dir, self.word_type, perc_dict)

        word_idx = dict((word, i+1) for i, word in enumerate(self.vocab))
        self.idx_words = {'0': '|NullSpace|'}
        self.idx_words = dict((str(i+1), word) for i, word in enumerate(self.vocab))

        self.max_story_size = 20
        self.query_size = max(train_data[1][1], test_data[1][1])
        self.sentence_size = max(train_data[1][2], test_data[1][2])
        self.memory_size = min(memory_size, self.max_story_size)

        # Add time words/indexes
        for i in range(self.memory_size):
            word_idx["time{}".format(i+1)] = "time{}".format(i+1)

        self.num_vocab = len(word_idx) + 1 # +1 for nil word
        self.sentence_size = max(self.query_size, self.sentence_size) # for the position
        self.sentence_size += 1  # +1 for time words
        self.word_idx = word_idx

        self.mean_story_size = 20

        self.train_lines = train_data[0]
        self.train_pos = train_data[1][0]
        self.test_lines = test_data[0]
        self.test_pos = test_data[1][0]

        self.set_train_test(train)


    def set_train_test(self, train):
        if train:
            self.lines = self.train_lines
            self.pos = self.train_pos
        else:
            self.lines = self.test_lines
            self.pos = self.test_pos
        self.len_dataset = len(self.pos)

    def set_lines_pos(self, lines):
        self.lines = lines
        pos = []
        counter = 0
        for line in lines:
            tokens = tokenize(line)
            if len(tokens) > 0 and int(tokens[0]) == 1:
                pos.append(counter)
            counter += 1
        self.pos = pos
        self.len_dataset = len(self.pos)



    def __getitem__(self, idx):
        story, query, answer, options = vectorize_data_CBTest(idx, self.lines, self.pos, self.word_idx,
                                                            self.sentence_size, self.memory_size)
        data_story = torch.LongTensor(story)
        data_query = torch.LongTensor(query)
        data_answer = torch.LongTensor(np.argmax(answer, axis=1))
        data_options = torch.FloatTensor(options)

        '''data_story = torch.LongTensor(story[:self.sentence_size])
        data_query = torch.LongTensor(query[:self.sentence_size])
        data_answer = torch.LongTensor(np.argmax(answer[:self.sentence_size], axis=1))
        data_options = torch.FloatTensor(options[:self.sentence_size])
'''
        return data_story[0], data_query[0], data_answer[0], data_options[0]

    def __len__(self):
        return self.len_dataset
