import argparse

import tensorflow as tf
from tensorflow import keras

from model import BaseModel
from utils import load_data


def eval(args):

    base_model = BaseModel()

    loss_func = [keras.losses.SparseCategoricalCrossentropy(from_logits=True),]
    metrics = ['accuracy']

    model = base_model(loss_func, metrics)
    model.load_weights(args.checkpoint_path)

    test_loader = load_data(data_path = args.test_path)

    res_eval =  model.evaluate(test_loader)
    score_acc = res_eval[-1]
    return score_acc


def parse_args(parser):
    parser.add_argument('--test_path', required=True, help='path to testing dataset')
    parser.add_argument('--checkpoint_path', required=True, help='path to checkpoint')
    
    args = parser.parse_args()
    return args 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    score_acc = eval(args)
    print("Test Accuracy: ", score_acc)