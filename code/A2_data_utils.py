#! -*- coding: utf-8 -*-

import re
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
import numpy as np

#   区分大小写
entity_labels = ['rna', 'disease', 'NEG']
id2label = {i: j for i, j in enumerate(sorted(entity_labels))}
label2id = {j: i for i, j in id2label.items()}
# BIO标注
num_labels = len(entity_labels) * 2 + 1
# 分类
sentence_labels = ["RDI", "NEG", "RRI"]
a=np.diagflat([1]*len(sentence_labels))
label2id_list = {sentence_labels[i]: list(a[i]) for i in range(len(sentence_labels))}
label2id_list["neg"] = [0]*len(sentence_labels)
# 分词规则的wordpiece的txt文件
vocab_path = 'model_weights/vocab.txt'
tokenizer = Tokenizer(vocab_path, do_lower_case=True)


def load_data(data_path, max_len=260):
    """加载数据
    单条格式：[(片段1, 标签1), (片段2, 标签2), (片段3, 标签3), ...]
    注意查看返回值
    """
    datasets = []   #[['circ-eno1', 'rna'], ['and', 'o'], ['its', 'o'], ['host', 'o'], ['gene', 'o'], ['eno1', 'rna'], ['were', 'o'], ['identified', 'o'], ['to', 'o'], ['be', 'o'], ['upregulated', 'o'], ['in', 'o'], ['luad', 'disease'], ['cells', 'o'], ['.', 'o']]
    samples_len = []

    X = []
    y = []   #['b-rna', 'o', 'o', 'o', 'o', 'b-rna', 'o', 'o', 'o', 'o', 'o', 'o', 'b-disease', 'o', 'o']
    sentence = []
    labels = []

    split_pattern = re.compile(r'[,:;\.\?!]')
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            # sentence = [w1,w2,w3,...,wn], labels=[B-xx,I-xxx,,,...,O]
            line = line.strip().split()
            if (not line or len(line) < 2):
                X.append(sentence.copy())
                y.append(labels.copy())
                sentence.clear()
                labels.clear()
                continue
            word, tag = line[0], line[1]  # .replace('_','-')
            if split_pattern.match(word) and len(sentence) >= max_len:  # 判断分句、句末
                sentence.append(word)
                labels.append(tag)
                X.append(sentence.copy())
                y.append(labels.copy())
                sentence.clear()
                labels.clear()
            else:
                sentence.append(word)
                labels.append(tag)
    #这里是正常的长度
    if len(sentence):
        X.append(sentence.copy())
        sentence.clear()
        y.append(labels.copy())
        labels.clear()
    y_out = []  # 跟句子的长度一致，进行了label的筛选【与continue一致，句子过滤我也要跟着过滤】
    for token_seq, label_seq in zip(X, y):
        if len(token_seq) < 2:
            print(token_seq)
            print(label_seq)
            continue
        y_out.append(label_seq)
        sample_seq, last_flag = [], ''
        for token, this_flag in zip(token_seq, label_seq):
            # last_flag = this_flag
            if this_flag == 'o':
                sample_seq.append([token, 'o'])
            elif this_flag[:1] == 'b':
                sample_seq.append([token, this_flag[2:]])  # B-city
            else:  # I-
                if len(sample_seq) == 0:
                    sample_seq.append([token, this_flag[2:]])  # 有的label_seq直接第一个就是i-
                else:
                    sample_seq[-1][0] += " " + token
        datasets.append(sample_seq)  # sample_seq是这样的[['circ-eno1', 'rna'], ['and', 'o'], ['its', 'o'], ['host', 'o'], ['gene', 'o'], ['eno1', 'rna'], ['were', 'o'], ['identified', 'o'], ['to', 'o'], ['be', 'o'], ['upregulated', 'o'], ['in', 'o'], ['luad', 'disease'], ['cells', 'o'], ['.', 'o']]
        samples_len.append(len(token_seq))  # 但是sample_seq长度不等于token_seq长度，因为sample_seq会有B I多个单词合并情况

    # df = pd.DataFrame(samples_len)
    # print(data_path, '\n', df.describe())  # count、mean、std、min、25%、50%、75%、max
    return datasets, y_out


class data_generator(DataGenerator):
    """数据生成器/数据迭代器
    """
    def __iter__(self, random=True):
        max_len = 260  # 200
        batch_token_ids, batch_segment_ids, batch_labels, batch_label_sentence = [], [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]  # [CLS]
            #batch_label_sentence.append(label2id_list[item[-1][0]])

            for w, l in item[0:-1]:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < max_len:
                    token_ids += w_token_ids
                    # if l == 'O':
                    if l == 'o':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]  # [seq]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            batch_label_sentence.append(label2id_list[item[-1][0]])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], [np.asarray(batch_label_sentence), batch_labels]
                batch_token_ids, batch_segment_ids, batch_labels, batch_label_sentence = [], [], [], []


def load_data2(file_path, max_len=260):
    """加载数据 （用于文本分类）
    单条格式：(文本, 标签id)
    """
    # 保持与load_data一样的输出
    split_str = ",:;/.()"
    sample_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            for i in split_str:
                line = line.replace(i, " " + i + " ")  # 分割标点符号，与positive一致
            out = [[i, 'o'] for i in line.split()]
            # if split_pattern.match(word) and len(sentence) >= max_len:  # 判断分句、句末
            sample_list.append(out[0:max_len])

    return sample_list


def load_data3(file_path, labeled=False):
    """加载数据 （用于文本分类）
    单条格式：(文本, 标签id)
    """
    # 专用于evaluate
    split_str = ",:;/.()"
    sample_list, label_list = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if labeled:
                label, line = line.split("\t")
            else:
                label = "neg"
            for i in split_str:
                line = line.replace(i, " " + i + " ")  # 分割标点符号，与positive一致
            # if split_pattern.match(word) and len(sentence) >= max_len:  # 判断分句、句末
            sample_list.append(line)
            label_list.append(label)
    return sample_list, label_list


def file2generator(file_list_pos, max_len):
    train_data_all = []
    for file_path, file_type in file_list_pos:
        if file_type == "neg":
            train_data = load_data2(file_path, max_len)
        else:
            train_data, _ = load_data(file_path, max_len)
        train_data = [i + [[file_type, "sentence_label"]] for i in train_data]
        train_data_all = train_data_all + train_data
    train_generator = data_generator(train_data_all)
    return train_generator


def file2generator2(file_list_pos, max_len):
    train_data_all = []
    for file_path, file_type in file_list_pos:
        if file_type == "neg":
            train_data = load_data2(file_path, max_len)
        else:
            train_data, _ = load_data(file_path, max_len)
            if file_type == "RLI":
                train_data = train_data * 10
        train_data = [i + [[file_type, "sentence_label"]] for i in train_data]
        train_data_all = train_data_all + train_data
    train_generator = data_generator(train_data_all)
    return train_generator

