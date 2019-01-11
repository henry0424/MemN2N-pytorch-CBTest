import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import bAbIDataset, CBTestDataset
from model import MemN2N
import copy

class Eval():
    def __init__(self, config):
        self.eval_data = CBTestDataset(config.dataset_dir, config.word_type, perc_dict=config.perc_dict)
        self.eval_data.set_train_test(train=False)

        settings = {
            "use_cuda": config.cuda,
            "num_vocab": self.eval_data.num_vocab,
            "embedding_dim": 20,
            "sentence_size": self.eval_data.sentence_size,
            "max_hops": config.max_hops
        }

        print("Longest sentence length", self.eval_data.sentence_size)
        print("Longest story length", self.eval_data.max_story_size)
        print("Average story length", self.eval_data.mean_story_size)
        print("Number of vocab", self.eval_data.num_vocab)

        self.mem_n2n = MemN2N(settings)
        self.mem_n2n.load_state_dict(torch.load(config.check_point_path))
        self.mem_n2n.eval()
        print(self.mem_n2n)
            
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()

        self.start_epoch = 0
        self.config = config

    def run_txt(self, file):
        config = self.config
        f = open(file, 'r')
        lines_eval =  f.readlines()
        f.close()
        self.eval_data.set_lines_pos(lines_eval)
        print("Evaluation set size: ", self.eval_data.__len__())
        self.eval_loader = DataLoader(self.eval_data,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=False)

        predicted_answers, real_answers, eval_acc = self.evaluate()
        
        return self.eval_data.pos, self.eval_data.lines, predicted_answers, real_answers, eval_acc, self.eval_data.idx_words
 
    def load(self, directory):
        pass

    def evaluate(self):
        correct = 0
        loader = self.eval_loader

        predicted_answers = []
        real_answers = []
        for step, (story, query, answer, options) in enumerate(loader):
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)
            options = Variable(options)

            if self.config.cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()
                options = options.cuda()

            pred_prob = self.mem_n2n(story, query, options)[1]
            pred = pred_prob.data.max(1)[1] # max func return (max, argmax)
            correct += pred.eq(answer.data).cpu().sum()

            real_answers += list(answer.cpu().numpy())
            predicted_answers += list(pred.cpu().numpy())

        acc = float(correct.item()) / len(loader.dataset)
        return predicted_answers, real_answers, acc
