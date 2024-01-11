import sys
import os
import tensorflow as tf
from tensorflow.keras.constraints import non_neg
from official.nlp import optimization
from tensorflow.keras.layers import Dense
from utilities.config import *
import ast

linear_ffn = Dense(1, activation=None, name='linear_ffn', kernel_initializer='ones', bias_initializer='zeros', kernel_constraint=non_neg())


def build_input_layer(part):
    input_layer = tf.keras.layers.Input(shape=(FEATURE_NUM,), dtype=tf.float32, name=(part + "_input_layer"))
    return input_layer


def pairwise_hinge_loss(actual, prediction):
    positive_passage_score = tf.gather(prediction, [0], axis=1)
    negative_passage_score = tf.gather(prediction, [1], axis=1)
    return tf.reduce_mean(tf.maximum(0.0, MARGIN - positive_passage_score + negative_passage_score))


def build_optimizer(training_size):
    steps_per_epoch = training_size / LINEAR_BATCH_SIZE
    num_train_steps = steps_per_epoch * TRAINING_EPOCHS
    num_warmup_steps = int(0.03 * num_train_steps)
    return optimization.create_optimizer(init_lr=INITIAL_LEARNING_RATE,
                                         num_train_steps=num_train_steps,
                                         num_warmup_steps=num_warmup_steps,
                                         optimizer_type="adamw")


def build_ranking_model(training_size):
    pos_input_layer = build_input_layer("pos")
    neg_input_layer = build_input_layer("neg")
    positive_relevance = linear_ffn(pos_input_layer)
    negative_relevance = linear_ffn(neg_input_layer)
    relevance = tf.concat([positive_relevance, negative_relevance], 1)
    model_to_rank = tf.keras.Model(inputs=[pos_input_layer, neg_input_layer],
                                   outputs=relevance)
    model_to_rank.compile(loss=pairwise_hinge_loss, optimizer=build_optimizer(training_size))
    return model_to_rank


def restore_model(checkpoint, training_size):
    ranking_model = build_ranking_model(training_size)
    ranking_model.load_weights(checkpoint)
    return ranking_model


def load_training_data(training_triple_file_path):
    pos_features_list = list()
    neg_features_list = list()
    count = 0
    training_triple_file = open(training_triple_file_path, "rt")
    while True:
        line = training_triple_file.readline().strip()
        if line == "":
            break
        pos_features = ast.literal_eval(line)
        neg_features = ast.literal_eval(training_triple_file.readline().strip())
        training_triple_file.readline()
        training_triple_file.readline()
        pos_features_list.append(pos_features)
        neg_features_list.append(neg_features)
        count += 1
    pos_features_list = tf.reshape(pos_features_list, [count, FEATURE_NUM])
    neg_features_list = tf.reshape(neg_features_list, [count, FEATURE_NUM])
    return pos_features_list, neg_features_list, count


def train(training_triple_file_path):
    # Wrap data in Dataset objects
    pos_features, neg_features, training_size = load_training_data(training_triple_file_path)
    eval_size = int(0.1 * training_size)
    train_examples = (pos_features[:-eval_size], neg_features[:-eval_size])
    test_examples = (pos_features[-eval_size:], neg_features[-eval_size:])
    ranking_model = build_ranking_model(training_size)

    train_labels = tf.zeros([training_size - eval_size, 1], tf.float32)
    test_labels = tf.zeros([eval_size, 1], tf.float32)

    train_data = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).batch(LINEAR_BATCH_SIZE)
    test_data = tf.data.Dataset.from_tensor_slices((test_examples, test_labels)).batch(LINEAR_BATCH_SIZE)

    # Disable AutoShard
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)
    test_data = test_data.with_options(options)

    ranking_model.fit(train_data, epochs=TRAINING_EPOCHS, validation_data=test_data)
    print(linear_ffn.weights)


if __name__ == "__main__":
    training_triple_file_path = "../data/conv_feature_vec_train.txt"
    train(training_triple_file_path)
