#! -*- coding: utf-8 -*-
import pickle
from bert4keras.backend import K, keras, search_layer
from bert4keras.snippets import ViterbiDecoder, to_array
from A2_data_utils import *
from A1_build_model import *


np.set_printoptions(threshold=np.inf)

epochs = 10  # 1#2#3#4#5#6#7#8#9#10#11
max_len = 260  # 200#70
batch_size = 16  # 32
lstm_units = 512  # 128
drop_rate = 0.4  # 0.1
leraning_rate = 5e-5  # 1e-5
class_nums = 3

config_path = "model_weights/config.json"
checkpoint_path = "model_weights/PubMedBERT/bert_model.ckpt"
# 记得设置不同的cpkt的存储路径

def training(file_list_train, file_list_valid, file_list_train_neg, file_list_valid_neg, checkpoint_save_path,
             crf_save_path):
    train_generator = file2generator(file_list_train, max_len)
    valid_generator = file2generator(file_list_valid, max_len)

    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_save_path,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    model, CRF = model_merge(config_path, checkpoint_path, num_labels, lstm_units, drop_rate, leraning_rate, class_nums)

    history = model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator),
        epochs=epochs,
        callbacks=[checkpoint]
    )
    #
    pickle.dump(K.eval(CRF.trans), open(crf_save_path, 'wb'))

    print("--------- Step one finshed ---------")

    # 2. 针对negative数据
    # 在positive的基础上增加negative
    file_list_train2 = file_list_train + file_list_train_neg
    file_list_valid2 = file_list_valid + file_list_valid_neg
    train_generator = file2generator(file_list_train2, max_len)
    valid_generator = file2generator(file_list_valid2, max_len)

    checkpoint2 = keras.callbacks.ModelCheckpoint(
        checkpoint_save_path + "2",
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    model, CRF = model_merge(config_path, checkpoint_path, num_labels, lstm_units, drop_rate, leraning_rate,
                             class_nums, trainable_status=False)
    model.load_weights(checkpoint_save_path)
    history2 = model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator),
        epochs=epochs,
        callbacks=[checkpoint2]
    )
    pickle.dump(K.eval(CRF.trans), open(crf_save_path + "2", 'wb'))





