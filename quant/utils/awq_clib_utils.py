from ast import Not
from io import text_encoding
from pydoc import text
from random import sample
from tkinter import N

import torch 
import logging
from typing import List, Union
from datasets import load_dataset

def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=128,
    max_seq_len=512,
    split="train",
    text_column="text",
):
    # data可以是字符串，表示huggingface数据集的名字；也可以是一个字符串列表，表示文本数据；还可以是一个二维整数列表，表示已经tokenize好的数据
    if isinstance(data, str):
        if data == "pileval":
            # pileval数据集是一个文本数据集，包含了各种文本数据的子集，比如c4、wikitext2等，可以用来做校准
            dataset = load_dataset("mit-han-lab/pileval", split="validation")
        else:
            dataset = load_dataset(data, split=split)
        # shuffle数据集，设置随机种子为42，保证每次运行结果一致
        dataset = dataset.shuffle(seed=42)
    elif isinstance(data, list):
        if isinstance(data[0], str):
            # 如果data是一个字符串列表，直接使用这个列表作为文本数据
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0], list):
            # 如果data是一个二维整数列表，表示已经tokenize好的数据，直接返回这个列表
            dataset = data
    else:
        raise NotImplementedError("Unsupported data format")
    

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, List):
            line_encoder = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoder = tokenizer.encode(line)

        if len(line_encoder) > max_seq_len:
            continue
        # sample.shape = [1, seq_len], 这里的seq_len是每一行文本的token数量
        # 里面存的是token id, 代表一个词
        sample = torch.tensor([line_encoder])
        # numel()表示sample中元素的数量，如果为0，说明这个样本没有有效的token，跳过这个样本
        if sample.numel() == 0:
            continue
        samples.append(sample)

        n_run += 1
        if n_run >= n_samples:
            break
    
    # sampels是 [sample1, sample2, ...]，每个sample的shape是[1, seq_len]
    # 把它们cat起来，得到一个shape是[1, seq_len1 + seq_len2 + ...]的tensor
    # 然后按照max_seq_len切分成多个chunk，每个chunk的shape是[1, max_seq_len]
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logging.debug(f" * Split into {n_split} chunks with max_seq_len {max_seq_len}")

    return [
        cat_samples[:, i*max_seq_len:(i+1)*max_seq_len] for i in range(n_split)
    ]
        