#! -*- coding: utf-8 -*-
import keras
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.layers import ConditionalRandomField


def textcnn(inputs, kernel_initializer):
    # 3,4,5
    cnn1 = keras.layers.Conv1D(
        256,  # [[0.1,0.2],[0.3,0.1],[0.4,0.2]],[[0.12,0.32],[0.31,0.12],[0.24,0.12]]
        3,
        strides=1,
        padding='same',  # 'valid'
        activation='relu',
        kernel_initializer=kernel_initializer
    )(inputs)  # shape=[batch_size,maxlen-2,256]
    cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]

    cnn2 = keras.layers.Conv1D(
        256,
        4,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_initializer
    )(inputs)
    cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

    cnn3 = keras.layers.Conv1D(
        256,
        5,
        strides=1,
        padding='same',
        kernel_initializer=kernel_initializer
    )(inputs)
    cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

    output = keras.layers.concatenate(
        [cnn1, cnn2, cnn3],name="textCNN",
        axis=-1)  # [batch_size,256*3]

    return output


def model_merge(config_path, checkpoint_path, num_labels, lstm_units, drop_rate, leraning_rate, class_nums, trainable_status=True, model_type='bert'):
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model=model_type,  # albert #bert、albert、albert_unshared、nezha、electra、gpt2_ml、t5
        # application='encoder',
        return_keras_model=False  # 返回Keras模型，还是返回bert4keras的模型类
    )

    x = bert.model.output  # [batch_size, seq_length, 768]

    # 1. NER
    lstm = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units,
            kernel_initializer='he_normal',
            return_sequences=True,
            trainable=trainable_status
        ), trainable=trainable_status, name="BiLSTM"
    )(x)  # [batch_size, seq_length, lstm_units * 2]

    x = keras.layers.concatenate(
        [lstm, x],
        axis=-1  # 在最后一维度上拼接
    )  # [batch_size, seq_length, lstm_units * 2 + 768]

    x = keras.layers.TimeDistributed(
        keras.layers.Dropout(drop_rate, trainable=trainable_status), trainable=trainable_status
    )(x)  # [batch_size, seq_length, lstm_units * 2 + 768]

    x = keras.layers.TimeDistributed(
        keras.layers.Dense(
            num_labels,
            activation='relu',
            kernel_initializer='he_normal',
            trainable=trainable_status
        ), trainable=trainable_status, name="merged_NER"
    )(x)  # [batch_size, seq_length, num_labels]

    crf = ConditionalRandomField(name="output_NER", trainable=trainable_status)
    # output_NER = crf(x, name="output_NER")
    output_NER = crf(x)

    # 2. 分类
    cls_features = keras.layers.Lambda(
        lambda x: x[:, 0],
        name='cls-token'
    )(bert.model.output)  # shape=[batch_size,768]
    all_token_embedding = keras.layers.Lambda(
        lambda x: x[:, 1:-1],
        name='all-token'
    )(bert.model.output)  # shape=[batch_size,maxlen-2,768]

    cnn_features = textcnn(
        all_token_embedding, 'he_normal')  # shape=[batch_size,cnn_output_dim]
    # bert模型输出cls向量跟textcnn的输出向量做拼接
    concat_features = keras.layers.concatenate(
        [cls_features, cnn_features],
        axis=-1)
    #  dropout
    concat_features = keras.layers.Dropout(0.2)(concat_features)
    #  256全连接层跟3全连接层
    dense = keras.layers.Dense(
        units=256,
        activation='relu',
        kernel_initializer='he_normal',
        name="merged_RE"
    )(concat_features)

    output_classify = keras.layers.Dense(
        units=class_nums,
        activation='sigmoid',  # 多分类模型变多标签模型 softmax --> sigmoid
        kernel_initializer='he_normal',
        name="output_classify"
    )(dense)


    # 3. merge
    model = keras.models.Model(bert.input, [output_classify, output_NER])
    # model.summary()
    losses = {
        "output_classify": 'binary_crossentropy',
        "output_NER": crf.sparse_loss,
    }
    lossWeights = {"output_classify": 1.0, "output_NER": 1.0}
    my_metrics = {
        "output_classify": 'accuracy',
        "output_NER": crf.sparse_accuracy,
    }
    if not trainable_status:
        number_targes = list(range(104, 113)) + [114,116,118,120]
        for layer in model.layers:
                layer.trainable = False
        for number_targe in number_targes:
            model.layers[number_targe].trainable = True
    model.compile(
        loss=losses,
        optimizer=Adam(leraning_rate),
        loss_weights=lossWeights,
        metrics=my_metrics
    )

    return model, crf

