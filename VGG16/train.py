import tensorflow as tf
from tensorflow import keras

from model import BaseModel
from utils import load_data, plot_accuracy, plot_loss

import os 
import argparse


def train(args):
    train_loader = load_data(data_path = args.train_path)
    valid_loader = load_data(data_path = args.valid_path)

    base_model = BaseModel(args.num_channels, args.img_W, args.img_H, args.num_classes)
 
    loss_func = [keras.losses.SparseCategoricalCrossentropy(from_logits=True),]
    metrics = ['accuracy']
    model = base_model(loss_func, metrics)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                        filepath = os.path.join(args.save_path, 'weight'),
                                                        save_weights_only=True,
                                                        monitor = 'val_accuracy',
                                                        mode = 'max',
                                                        save_best_only = True)
    history = model.fit(train_loader,
                        epochs = args.total_epochs,
                        verbose = 2,
                        callbacks = [model_checkpoint_callback],
                        validation_data = valid_loader)

    return history 

def parse_args(parser):
    parser.add_argument('--train_path', required=True, help='path to training dataset')
    parser.add_argument('--valid_path', required=True, help='path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--img_H', type=int, default=224)
    parser.add_argument('--img_W', type=int, default=224)
    parser.add_argument('--total_epochs', type=int, default=10)
    parser.add_argument('--save_path',type=str, default='./save')
    
    args = parser.parse_args()
    return args 

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    history = train(args)

    plot_loss(history, args.save_path)
    plot_accuracy(history, args.save_path)