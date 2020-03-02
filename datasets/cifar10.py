from __future__ import print_function
import tensorflow_datasets as tfds
import tensorflow as tf

class mnist_dataset():
    def __init__(self, root, batch_size):
        file_path = root + '/datasets/tensorflow_datasets'
        mnist, meta = tfds.load('cifar10', data_dir=file_path, as_supervised=True, with_info=True)
        print(meta)
        self.train_dataset=mnist['train']
        self.batch_size = batch_size
        self.name = 'cifar10'
        return
    def parse(self, x, y):
        x=tf.cast(x, tf.float32)
        # x = tf.expand_dims(x, -1)
        x=x/255*2 - 1
        y = tf.cast(y, tf.int64)
        return x
    def get_train_dataset(self):
        train_dataset = self.train_dataset.map(self.parse).shuffle(10000).batch(self.batch_size)
        return train_dataset
