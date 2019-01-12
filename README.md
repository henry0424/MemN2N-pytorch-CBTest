# MemN2N-pytorch-CBTest

### Reference
> Reference AI model
>> [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895)
>> Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus
>> ![End-To-End Memory Networks](https://i.imgur.com/YBBcbNy.png)
> 
> Reference repositories
>> https://github.com/nmhkahn/MemN2N-pytorch  
>> https://github.com/domluna/memn2n


---

### Clone Project
```
$ git clone https://github.com/henry0424/MemN2N-pytorch-CBTest.git
$ cd MemN2N-pytorch-CBTest
MemN2N-pytorch-CBTest$
```

### Dataset
```
MemN2N-pytorch-CBTest$ wget http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz
MemN2N-pytorch-CBTest$ tar -xzf CBTest.tgz
```

### Create Virtualenv
```
MemN2N-pytorch-CBTest$ virtualenv --system-site-packages -p python3 .
MemN2N-pytorch-CBTest$ source bin/activate
```

### Install Pytorch with CUDA
```
(MemN2N-pytorch-CBTest)MemN2N-pytorch-CBTest$ pip3 install torch torchvision
(MemN2N-pytorch-CBTest)MemN2N-pytorch-CBTest$ python -c "import torch; print(torch.__version__)"
```
![Verify the install](https://i.imgur.com/RR4lhhI.png)



---


### Train
```
$ python -W ignore ./memn2n/train.py --cuda
```
![Train](https://i.imgur.com/li1f6px.png)
> --cuda  
--dataset_dir  
--task  
--max_hops  
--batch_size  
--max_epochs  
--lr  
--decay_interval  
--decay_ratio  
--max_clip  
--word_type  
--perc_dict  



## Create Question File
```
(MemN2N-pytorch-CBTest)MemN2N-pytorch-CBTest$ python memn2n/question_generator.py --number=10 --file=./CBTest/data/cbtest_V_test_2500ex.txt --random=1 --remove=1
```
>--number 問題數目  
>--file 抽取檔案  
>--random 隨機抽取  
>--remove 移除答案  

## Evaluation / Prediction
```
(MemN2N-pytorch-CBTest)MemN2N-pytorch-CBTest$ python -W ignore ./memn2n/eval.py --cuda --check_point_path=./CBTest/data_epoch_25 --file=./Question.txt
```
![Evaluation](https://i.imgur.com/ocWDsNf.png)


---

## Project structure / description

| File          | description                                        |
| ------------- | -------------------------------------------------- |
| data_utils.py | define how to vectorize(tokenize) words/sentences. |
| dataset.py    | define how to load bAbi dataset                    |
| model.py      | define memeory network model                       |
| train.py      | define main process                                |
| trainer.py    | define train process                               |
| evaluation.py | define evaluation function                         |
| eval.py       | define evaluation process                          |
| question_generator.py | define question file                       |
