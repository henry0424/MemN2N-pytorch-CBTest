import argparse
from evaluation import Eval
import torch
import numpy as np
from data_utils import tokenize

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="CBTest/data")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--word_type", type=str, default="P")
    parser.add_argument("--perc_dict", type=float, default=1.0)
    parser.add_argument("--check_point_path", type=str, default="checkpoints")
    parser.add_argument("--file", type=str)


    return parser.parse_args()


def get_ans_index(line, pred_ans):
    ans = line[line.rfind('\t'):].strip().rstrip().split('|')
    idx = 0
    for _ans in ans:
        idx += 1
        if _ans == pred_ans:
            break
    return idx

def main(config):
    model = Eval(config)
    results = model.run_txt(config.file)

    print("Test accuracy: ", results[4])
    print("")

    for i in range(len(results[0])):
        counter = 0
        while True:
            line = results[1][results[0][i] + counter]
            if len(line) > 0:
                # print(line)
                # print("")
                counter += 1
                if int(tokenize(line)[0]) == 21:
                    idx = get_ans_index(line, results[5][str(results[2][i])])
                    break
        print("Real answer: ", results[5][str(results[3][i])], "     Predicted answer: ", results[5][str(results[2][i])])
        print("Index of ans: ",idx)  
        print("")

        # key = input("Press enter to continue...")
        # if key == 'q':
        #     break

    print("")
    print("Thanks!! :)")


if __name__ == "__main__":
    config = parse_config()
    main(config)
