import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split

import utils

image_size = 256
image_channel = 3
label_size = 150
unknown_label_size = 50
train_data_path = 'dataset/train_images'
save_path = 'v_saved_model/'
show_every_n = 100
saved_every_n = 100
train_step = 10000


'''
使用CUB数据集训练一个image classifier，使用的是论文中提到的AlexNet的结构
在数据集的使用中，只用前label_size=150个类别进行训练，剩余的unknown_data_size=50作为
模型未见过的label用于DeViSE模型的测试
'''






class AlexNet(object):
    '''
    使用给定的训练数据集训练一个AlexNet模型
    '''
    def __init__(self,
                 batch_size=64,
                 num_units = 128,
                 num_classes = 150,
                 learning_rate = 0.002,
                 num_epoches = 1,
                 is_training=True):

        self.num_units = num_units
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epoches = num_epoches

        if is_training:
            self.batch_size = batch_size
            self.drop_prob = 0.5
        else:
            self.batch_size = 1
            self.drop_prob = 1.0

        self.build_model()

    def train_test_split(self, image_dataset, label_dataset):
        train_x, test_x, train_y, test_y = \
            train_test_split(image_dataset, label_dataset, test_size=0.1)
        return train_x, test_x,train_y, test_y

    def get_batch(self):
        for i in range(self.data_size // self.batch_size):
            batch_x = self.train_x[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = self.train_y[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_x, batch_y

    def build_input(self):
        input_x = tf.placeholder(
            tf.float32, [self.batch_size, image_size, image_size, image_channel], name='input_x')
        output_y = tf.placeholder(tf.float32, [self.batch_size, self.num_classes], name='output_y')
        keep_prob = tf.placeholder(tf.float32, name='drop_keep')
        return input_x, output_y, keep_prob

    def bulid_CNN(self, input_x):
        conv_layer1 = tf.layers.conv2d(inputs=input_x, filters=8, kernel_size=[8, 8],
                                       strides=[2, 2], padding='SAME',activation=tf.nn.relu)
        pooling_layer1 = tf.layers.max_pooling2d(
            inputs=conv_layer1, pool_size=[2, 2], strides=[2, 2])

        conv_layer2 = tf.layers.conv2d(inputs=pooling_layer1, filters=16, kernel_size=[4, 4],
                                       strides=[2, 2], padding='SAME',activation=tf.nn.relu)
        pooling_layer2 = tf.layers.max_pooling2d(
            inputs=conv_layer2, pool_size=[2, 2], strides=[2, 2])

        conv_layer3 = tf.layers.conv2d(inputs=pooling_layer2, filters=32, kernel_size=[4, 4],
                                       strides=[2, 2], padding='SAME',activation=tf.nn.relu)
        pooling_layer3 = tf.layers.max_pooling2d(
            inputs=conv_layer3, pool_size=[2, 2], strides=[2, 2])
        cnn_flat = tf.reshape(pooling_layer3, [self.batch_size, -1])
        return cnn_flat

    def bulid_full_connect(self, cnn_flat, keep_prob):
        fc_layer1 = tf.layers.dense(inputs=cnn_flat, units=self.num_units, activation=tf.nn.relu)
        fc_layer1 = tf.layers.dropout(fc_layer1, rate=keep_prob)
        output_layer = tf.layers.dense(inputs=fc_layer1, units=self.num_classes, activation=None)

        return output_layer

    def bulid_loss(self, logits, targets):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        loss = tf.reduce_mean(loss)
        return loss

    def bulid_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def bulid_accuracy(self, logits, targets):
        equality = tf.equal(tf.argmax(logits, axis=1), tf.argmax(targets, axis=1))
        equality = tf.cast(equality, tf.float32)
        accuracy = tf.reduce_mean(equality)
        return accuracy

    def build_model(self):
        tf.reset_default_graph()
        self.input_x, self.output_y, self.keep_prob = self.build_input()
        self.cnn_padding = self.bulid_CNN(self.input_x)
        self.output_layer = self.bulid_full_connect(self.cnn_padding, self.drop_prob)
        self.repsentation = self.output_layer
        self.loss = self.bulid_loss(self.output_layer, self.output_y)
        self.optimizer = self.bulid_optimizer(self.loss)
        self.accuracy = self.bulid_accuracy(self.output_layer, self.output_y)

    def train(self, image_dataset, label_dataset):
        self.train_x, self.test_x,self.train_y, self.test_y = \
            self.train_test_split(image_dataset, label_dataset)
        self.data_size = len(self.train_x)  # 4510

        saver = tf.train.Saver()
        #模型训练并将训练的结果保存在本地
        with tf.Session() as sess:
            print("AlexNet model training begins....")
            sess.run(tf.global_variables_initializer())
            global_steps = 0
            for epoch in range(self.num_epoches):
                generator = self.get_batch()
                for batch_x, batch_y in generator:
                    global_steps += 1
                    feed = {self.input_x: batch_x,
                            self.output_y: batch_y,
                            self.keep_prob: self.drop_prob}
                    show_loss, show_accu, _ = sess.run(
                        [self.loss, self.accuracy, self.optimizer], feed_dict=feed)

                    if global_steps % show_every_n == 0:
                        print('epoch: {}/{}..'.format(epoch+1, self.num_epoches+1),
                              'global_step: {}..'.format(global_steps),
                              'loss: {:.3f}..'.format(show_loss),
                              'accuracy: {:.2f}..'.format(show_accu))

                    if global_steps % saved_every_n == 0:
                        saver.save(sess, save_path+"e{}_s{}.ckpt".format(epoch, global_steps))
            saver.save(sess, save_path+"lastest.ckpt")

        print('training finished')


if __name__ == '__main__':
    #加载训练data，label已经被转换为one_hot
    train_x = utils.get_image_data()
    train_y = utils.get_one_hot_label_data()
    label2int, int2label = utils.get_parameter()

    model = AlexNet()
    model.train(train_x, train_y)


