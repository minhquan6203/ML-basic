import tensorflow as tf

from model import AlexNet
from dataset_helper import read_cifar_10

INPUT_WIDTH = 70
INPUT_HEIGHT = 70
INPUT_CHANNELS = 3

NUM_CLASSES = 10

LEARNING_RATE = 0.01   # Original value: 0.01
MOMENTUM = 0.9
KEEP_PROB = 0.5

EPOCHS = 100
BATCH_SIZE = 128

print('Reading CIFAR-10...')
X_train, Y_train, X_test, Y_test = read_cifar_10(image_width=INPUT_WIDTH, image_height=INPUT_HEIGHT)

alexnet = AlexNet(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, input_channels=INPUT_CHANNELS,
                  num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, momentum=MOMENTUM, keep_prob=KEEP_PROB)

with tf.compat.v1.Session() as sess:
    print('Evaluating dataset...')
    print()

    sess.run(tf.compat.v1.global_variables_initializer())

    print('Loading model...')
    print()
    alexnet.restore(sess, './model')

    print('Evaluating...')

    train_accuracy = alexnet.evaluate(sess, X_train, Y_train, BATCH_SIZE)
    test_accuracy = alexnet.evaluate(sess, X_test, Y_test, BATCH_SIZE)

    print('Train Accuracy = {:.3f}'.format(train_accuracy))
    print('Test Accuracy = {:.3f}'.format(test_accuracy))
    print()