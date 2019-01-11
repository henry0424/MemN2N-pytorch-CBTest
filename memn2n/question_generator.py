# 107003820 蔡宗廷
# CBTest Question Generator
# question_generator.py
#
#

import argparse
import os
import re
import random

class DataProcess(object):
    def __init__(self):
        self.__version__ = "1.0.1"
        self.path = "."
        self.data = []
        pass

    def read_assign_data(self, path):
        if os.path.exists(path) != True:
            print("Read Data Path Not Exists->", path)
            return
        path = os.path.realpath(path)
        print("Read Data From->", path)
        ws = open(path, 'r', encoding='UTF-8')
        for line in ws.readlines():
            line = line.strip()
            self.data.append(line)
        questions = len(self.data)/22
        # print("len(self.data)", len(self.data), "questions", questions)
        return self.data

    def read_data(self, path, type="train"):
        if os.path.exists(path) != True:
            print("Read Data Path Not Exists->", path)
            return
        path = os.path.realpath(path)
        self.path = path
        print("self.path", self.path)
        files = os.listdir(self.path)
        for f in files:
            if f.find(type) != -1:
                self.read_assign_data(self.path+"/"+f)
        return self.data

    def get_story(self, data, index=0):
        print("get story index:", index)
        story = []
        for i in range(21):
            story.append(data[22*index+i])
        return story

    def get_story_size(self, data):
        story_size = len(data)/22
        return story_size
    
    def remove_ans(self, story, str="AAAAA"):
        last_idx = len(story)-1
        ans_start = story[last_idx].find('\t', 1)
        ans_end = story[last_idx].find('\t', ans_start+1)
        ans = story[last_idx][ans_start:ans_end].strip().rstrip()
        story[last_idx] = story[last_idx].replace("\t"+ans+"\t", " ", 1)
        return story

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--remove", type=int, default=0)
    parser.add_argument("--file", type=str, default="CBTest/data/cbtest_NE_test_2500ex.txt")
    return parser.parse_args()

def valid_question(file):
    _question = DataProcess()
    data = _question.read_assign_data("question.txt")
    max_size = _question.get_story_size(data)
    print(max_size)
    story = _question.get_story(data, 0)
    print(story)


def main(config):
    text_file = open("question.txt", "w+")
    data = _preprocess.read_assign_data(config.file)
    max_size = _preprocess.get_story_size(data)
    print(max_size)

    for i in range(config.number):
        if config.random == 0:
            story = _preprocess.get_story(data, i)
        else:
            story = _preprocess.get_story(data, random.randint(0, max_size))

        if config.remove == 1:
            story = _preprocess.remove_ans(story)

        for st in story:
            text_file.write(st+'\n')
        text_file.write('\n')

    text_file.close()
    valid_question("question.txt")    

if __name__ == "__main__":
    config = parse_config()
    _preprocess = DataProcess()
    main(config)