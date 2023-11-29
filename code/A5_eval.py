#! -*- coding: utf-8 -*-

from A4_metrics import *
from A3_train import *  # NER
from p_tqdm import t_map
from sklearn.metrics import  classification_report as cr
import pandas as pd

model, CRF = model_merge(config_path, checkpoint_path, num_labels, lstm_units, drop_rate, leraning_rate, class_nums, trainable_status=False)

class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        # while len(tokens) > max_len:
        while len(tokens) > max_len * 5:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])  # ndarray
        nodes = model.predict([token_ids, segment_ids])[1][0]  # [sqe_len,23]
        labels = self.decode(nodes)  # id [sqe_len,], [0 0 0 0 0 7 8 8 0 0 0 0 0 0 0]
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l) for w, l in entities]

NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

def predict_NER_single(d):
    text = ' '.join([i[0] for i in d])
    pred = NER.recognize(text)
    label = ['o' for _ in range(len(text.split(' ')))]
    b = 0  # b是文本的find函数的开始下标
    for item in pred:
        word, typ = item[0], item[1]
        start = text.find(word, b)
        liststart = text[:start].count(' ')
        listend = liststart + len(word.split(' '))
        label[liststart] = 'b-' + typ
        for k in range(liststart + 1, listend):
            label[k] = 'i-' + typ
        b = start + len(word)
    return label


def predict_NER(data):
    y_pred = t_map(predict_NER_single, data)
    return y_pred

def evaluate(data_path_list, out):
    max_len = 260
    test_data_all, y_true_all = [], []
    for data_path in data_path_list:
        test_data, y_true = load_data(data_path, max_len)
        test_data_all = test_data_all + test_data
        y_true_all = y_true_all + y_true

    y_pred = predict_NER(test_data_all)

    f1 = f1_score(y_true_all, y_pred, suffix=False)
    p = precision_score(y_true_all, y_pred, suffix=False)
    r = recall_score(y_true_all, y_pred, suffix=False)
    acc = accuracy_score(y_true_all, y_pred)

    print("A5 f1_score: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, accuracy_score: {:.4f}".format(f1, p, r, acc))
    temp = classification_report(y_true_all, y_pred, digits=4, suffix=False)
    print(temp)
    with open(out, "w", encoding="utf-8") as f:
        f.write("A5 f1_score: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, accuracy_score: {:.4f}".format(f1, p, r, acc))
        f.write(temp)
    return y_true_all, y_pred


# 多标签
def predict_single(test_text, maxlen=260):
    threshold = 0.5
    token_ids, segment_ids = tokenizer.encode(test_text, maxlen=maxlen)
    pred = model.predict([[token_ids], [segment_ids]])
    label_index = np.where(pred[0][0] > threshold)[0]  # 取概率值大于阈值的 onehot 向量索引, [12,34]
    labels = [sentence_labels[i] for i in label_index]
    one_hot_label = np.where(pred[0][0] > threshold, 1, 0)  # [[0,0,1,0,0,..],[0,0,1,0,1,..]]
    return one_hot_label, '|'.join(labels)


def evaluate_multi_label(data_path_list, out):
    max_len = 260
    # 注意输入与evaluate()不一样
    test_x_all, test_y_all = [], []
    for data_path, sentence_type in data_path_list:
        if sentence_type == "neg":
            test_x = load_data2(data_path, max_len)
        else:
            test_x, _ = load_data(data_path, max_len)
        test_x = [" ".join([i[0] for i in j]) for j in test_x]
        test_x_all = test_x_all + test_x
        test_y_all = test_y_all + [label2id_list[sentence_type]]*len(test_x)

    # test_x,test_y = load_data('./data/BioNLP_test.tsv')
    test_y_all = np.array(test_y_all)

    pred_y_list = t_map(predict_single, test_x_all, [max_len]*len(test_x_all))
    pred_y_list = np.array([i[0] for i in pred_y_list])

    # F1值
    temp = cr(test_y_all, pred_y_list, digits=4, target_names=sentence_labels)
    print(temp)  #
    with open(out, "w", encoding="utf-8") as f:
        f.write(temp)
    # for i in range(3):
    #     roc_plot(test_y_all[:, i], pred_y_list[:, i])

    return test_y_all, pred_y_list


def predict_single2(test_text, maxlen=260):
    threshold = 0.5
    token_ids, segment_ids = tokenizer.encode(test_text, maxlen=maxlen)
    pred = model.predict([[token_ids], [segment_ids]])
    one_hot_label = np.where(pred[0][0] > threshold, 1, 0)  # [[0,0,1,0,0,..],[0,0,1,0,1,..]]
    return one_hot_label, pred[0][0]

def evaluate_multi_label2(data_path_list, out):
    max_len = 260
    # 注意输入与evaluate()不一样
    test_x_all, test_y_all = load_data3(data_path_list[0], True)
    temp, temp1 = load_data3(data_path_list[1])
    test_x_all = test_x_all+temp
    test_y_all = test_y_all+temp1

    # test_x,test_y = load_data('./data/BioNLP_test.tsv')
    test_y_all = np.array([label2id_list[i] for i in test_y_all])

    pred_y_list = t_map(predict_single2, test_x_all, [max_len]*len(test_x_all))
    pred_y_prob = np.array([i[1] for i in pred_y_list])
    pred_y_list = np.array([i[0] for i in pred_y_list])


    # F1值
    temp = cr(test_y_all, pred_y_list, digits=4, target_names=sentence_labels)
    print(temp)  #
    with open(out, "w", encoding="utf-8") as f:
        f.write(temp)
    # for i in range(3):
    #     roc_plot(test_y_all[:, i], pred_y_list[:, i])

    return test_y_all, pred_y_prob

def evaluate_multi_label3(data_path_list, out):
    max_len = 260
    # 注意输入与evaluate()不一样
    test_x_all, test_y_all = load_data3(data_path_list[0], True)
    temp, temp1 = load_data3(data_path_list[1])
    test_x_all = test_x_all+temp
    test_y_all = test_y_all+temp1

    # test_x,test_y = load_data('./data/BioNLP_test.tsv')
    test_y_all = np.array([label2id_list[i] for i in test_y_all])

    pred_y_list = t_map(predict_single2, test_x_all, [max_len]*len(test_x_all))
    pred_y_prob = np.array([i[1] for i in pred_y_list])
    pred_y_list = np.array([i[0] for i in pred_y_list])
    #pred_y_list = pred_y_list[:, 2]
    #test_y_all = test_y_all[:, 2]
    # F1值
    temp = cr(test_y_all, pred_y_list, digits=3)
    print(temp)  #
    with open(out, "w", encoding="utf-8") as f:
        f.write(temp)
    # for i in range(3):
    #     roc_plot(test_y_all[:, i], pred_y_list[:, i])

    return test_y_all, pred_y_list

def entity_recognize(text):
    entity = NER.recognize(text[0:800])
    if len(entity) > 0:
        out_entity = [i[0] for i in entity if i[1] == "rna"]
        out = ", ".join(out_entity)
        return [out, len(out_entity)]
    else:
        return ["", 0]


if __name__ == '__main__':
    config_path = "model_weights/config.json"
    checkpoint_save_path = r'model_weights/bert_bilstm_crf.weights'
    CRF_path = r'model_weights/crf_trans.pkl'

    model.load_weights(checkpoint_save_path)
    NER.trans = pickle.load(open(CRF_path, 'rb'))
    # checkpoint_save_path = r"merged/checkpoint/bert_bilstm_crf_0.weights2"
    # crf_save_path = 'merged/checkpoint/crf_trans.pkl'
    file_list_test = [[f"../corpus/crossvalidation/nerdata/{x}/{path_order[i]}/uncased_train.txt", path_dataset[x]] for x in path_dataset.keys()]
    file_list_test_neg = [[f"../corpus/crossvalidation/multilabeldata/neg/{path_order[i]}/smalldata.txt", "neg"]]

    out_path = f"merged/test_result/out_{i}.txt"
    out_path2 = f"merged/test_result/out2_{i}.txt"
    out_path3 = f"merged/test_result/out3_{i}.txt"
    # NER
    #y_true, y_pred = evaluate([i[0] for i in file_list_test], out_path)

    # file_list_test = [[r"F:\study\study_RIscoper\RiscoperV2\corpus\ann2bio\MNDR-discretion-2022.02.18 ann2bio_lower.txt", "RDI"]]
    # evaluate([i[0] for i in file_list_test], r"F:\study\study_RIscoper\RiscoperV2\program\merged\temp\out.txt")

    #evaluate(["merged/test_result/train.tsv"], "bert_out2.txt")


    #y_true, y_pred = evaluate_multi_label(file_list_test + file_list_test_neg, out_path2)
    y_true, y_pred = evaluate_multi_label(["merged/test_result/train.tsv"], f"merged/test_result/out_temp.txt")

    # y_true, y_pred = evaluate_multi_label2([f"../corpus/crossvalidation/multilabeldata/XHL/{path_order[i]}_test.txt",
    #                                         f"../corpus/crossvalidation/multilabeldata/XHL/{path_order[i]}_test_neg.txt"], out_path2)
    #y_true, y_pred = evaluate_multi_label3([f"../corpus/crossvalidation/multilabeldata/XHL/temp.txt",
    #                                        f"../corpus/crossvalidation/multilabeldata/XHL/{path_order[i]}_test_neg.txt"], "bert_out.txt")
    out=pd.DataFrame([y_true[:, 0], y_true[:, 1], y_true[:, 2], y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]]).T
    out.to_csv(f"merged/test_result/out_temp.txt", header=False, sep="\t",index=False)








