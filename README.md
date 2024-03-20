# Temporal Knowledge Graph Reasoning via Time-Distributed Representation Learning

This is the released codes of the following paper:

Kangzheng Liu, Feng Zhao, Guandong Xu, Xianzhi Wang, and Hai Jin. Temporal Knowledge Graph Reasoning via Time-Distributed Representation Learning. ICDM 2022.

![DHU-NET](https://github.com/Liudaxian1/FIG/blob/main/DHU-NET.png)

## Citation

Please find the citation information of our paper here:

```shell
@inproceedings{DBLP:conf/icdm/LiuZX0022,
  author       = {Kangzheng Liu and
                  Feng Zhao and
                  Guandong Xu and
                  Xianzhi Wang and
                  Hai Jin},
  title        = {Temporal Knowledge Graph Reasoning via Time-Distributed Representation
                  Learning},
  booktitle    = {{IEEE} International Conference on Data Mining, {ICDM} 2022, Orlando,
                  FL, USA, November 28 - Dec. 1, 2022},
  pages        = {279--288},
  publisher    = {{IEEE}},
  year         = {2022},
  url          = {https://doi.org/10.1109/ICDM54844.2022.00038},
  doi          = {10.1109/ICDM54844.2022.00038},
  timestamp    = {Tue, 21 Mar 2023 20:53:05 +0100},
  biburl       = {https://dblp.org/rec/conf/icdm/LiuZX0022.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

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

We provide trained models for all datasets. The trained models can be downloaded at https://github.com/Liudaxian1/TrainedModels/tree/main/DHUNET_Models. Then put the trained models in the "./models" folder. Note that the commands in the $Process\ data$ section should be run first and download the provided model files one by one due to the large-file constraints of Github. Then directly use the following commands for different datasets to reproduce the results reported in our paper:

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

## Acknowledgements

The source codes take [RE-GCN](https://github.com/Lee-zix/RE-GCN) as the backbone to implement our proposed method. Please cite both our work and [RE-GCN](https://github.com/Lee-zix/RE-GCN) if you would like to use our source codes.
