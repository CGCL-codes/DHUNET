# Temporal Knowledge Graph Reasoning via Time-Distributed Representation Learning

This is the release code of the following paper:

Kangzheng Liu, Feng Zhao, Guandong Xu, Xianzhi Wang, and Hai Jin. Temporal Knowledge Graph Reasoning via Time-Distributed Representation Learning. ICDM 2022.

![DHU-NET](https://github.com/Liudaxian1/FIG/blob/main/DHU-NET.png)

## Environment dependencies

```shell
python==3.6.5
torch==1.9.0+cu102
dgl-cu102==0.8.0.post1
tqdm==4.62.3
rdflib==5.0.0
numpy==1.19.5
pandas==1.1.5
```

## Process data

First, extract the repetitive patterns of facts under all historical-timestamp KGs and the global static KG:

```shell
python get_history_record.py --dataset YAGO
```

where YAGO  is the name of one dataset we used in the experiment. Other datasets includes WIKI, ICEWS14, ICEWS18, ICEWS05-15, and GDELT.

## Train models

Then use the following commands to train the proposed models under different datasets:

```shell
cd src
python main.py -d YAGO --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --gpu 0
```

## Evaluate models

Finally, use the following commands to evaluate the proposed models under different datasets:

```shell
python main.py -d YAGO --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --gpu 0 --add-static-graph --test
```

## Reproduce the results in our paper

We provide trained models for all datasets. It is noted that the commands in the $Process\ data$ section should be run first. Then directly use the following commands for different datasets to reproduce the results reported in our paper:

YAGO:

```shell
python main.py -d YAGO --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --gpu 0 --add-static-graph --test
```

WIKI:

```shell
python main.py -d WIKI --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --gpu 0 --add-static-graph --test
```

ICEWS14:

```shell
python main.py -d ICEWS14 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --gpu 0 --add-static-graph --test
```

ICEWS18:

```shell
python main.py -d ICEWS18 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --gpu 0 --add-static-graph --test
```

ICEWS05-15:

```shell
python main.py -d ICEWS05-15 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --gpu 0 --add-static-graph --test
```

GDELT:

```shell
python main.py -d GDELT --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --gpu 0 --add-static-graph --test
```

## Contacts

Contact us with the following email address: FrankLuis@hust.edu.cn.
