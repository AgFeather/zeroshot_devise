import tensorflow as tf
import numpy as np
import visual_model
import word_to_vec
import utils

image_repesentation_size = 150
word_embedding_size = 200
visual_model_checkpoints_dir = 'v_saved_model/'
devise_model_checkpoints_dir = 'devise_saved_model/'
word2vec_saved_dir = 'word2vec_saved/text8model.model'
show_every_n = 50
save_every_n = 1000
num_labels = 150


def pre_train():
    print('pre-train of two models')

    word_to_vec.train_word2vec()

    train_x = utils.get_image_data()
    train_y = utils.get_one_hot_label_data()
    model = visual_model.AlexNet()
    model.train(train_x, train_y)
    print('Alex model and Word2Vec model have been trained..')


class DeViSE(object):

    def __init__(self,
                 batch_size=1,
                 learning_rate=0.001):
        self.alex_model = visual_model.AlexNet(is_training=False)
        self.word_model = word_to_vec.get_word2vec()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.bulid_model()
        print('devise model has been initialized')

    def bulid_input(self):
        input_x = tf.placeholder(dtype=tf.float32, shape=[image_repesentation_size, self.batch_size], name='input_x')
        label_y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, word_embedding_size], name='output_y')
        other_labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, word_embedding_size], name='other_labels')
        return input_x, label_y, other_labels

    def bulid_linear_model(self, input_x, label_y, other_labels):
        margin = 0.1
        weight_matrix = tf.get_variable('weight_matrix', shape=[word_embedding_size, image_repesentation_size])
        output_ = tf.matmul(weight_matrix, input_x)
        loss1 = tf.reduce_sum(tf.matmul(label_y, output_))
        loss2 = tf.reduce_sum(tf.matmul(other_labels, output_))
        loss = tf.maximum(0.0, margin-loss1+loss2)
        return output_, loss

    def bulid_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return optimizer

    def bulid_model(self):
        self.input_x, self.output_y, self.other_labels = self.bulid_input()
        self.prediction, self.loss = self.bulid_linear_model(self.input_x, self.output_y, self.other_labels)
        self.optimizer = self.bulid_optimizer(self.loss)

    def label_represent_set(self, label_data, int2label):
        label_set = list(set(label_data))
        label_set = [int2label[num] for num in label_set]
        self.vec_set = []
        for label in label_set:
            vector = self.word_model[label]
            self.vec_set.append(vector)


    def train(self, image_data, label_data, int2label):
        print('DeViSE model is training...')
        saver = tf.train.Saver()
        self.word_model_wv = self.word_model.wv
        with tf.Session() as sess:
            global_step = 0
            sess.run(tf.global_variables_initializer())
            for image, label in zip(image_data, label_data):
                global_step += 1
                image = np.array(image)
                image = np.reshape(image, (1, 256, 256, 3))
                feed = {self.alex_model.input_x: image}
                image_representation = sess.run(self.alex_model.repsentation, feed_dict=feed)
                image_representation = image_representation.T
                label = int2label[label]
                label = label.split('_')[0].lower()
                if label in self.word_model_wv.vocab:
                    label = self.word_model_wv[label]
                else:
                    label = self.word_model_wv['bird']
                other_label = np.random.randint(0, num_labels)
                other_label = int2label[other_label]
                other_label = other_label.split('_')[0].lower()
                if other_label in self.word_model_wv.vocab:
                    other_label = self.word_model_wv[other_label]
                else:
                    other_label = self.word_model_wv['bird']
                other_label_representation = np.reshape(other_label, (1,200))
                label_representation = np.reshape(label, (1, 200))

                feed = {self.input_x: image_representation,
                        self.output_y: label_representation,
                        self.other_labels: other_label_representation}

                show_loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed)

                if global_step % show_every_n == 0 and show_loss > 0:
                    print('step: {}'.format(global_step),
                          'loss: {:.3f}'.format(show_loss))
                if global_step % save_every_n == 0:
                    saver.save(sess, devise_model_checkpoints_dir + 's{}.ckpt'.format(global_step))
            saver.save(sess, devise_model_checkpoints_dir + 'lastest.ckpt')

    def predict(self, image):
        lastest_checkpoint = tf.train.latest_checkpoint(devise_model_checkpoints_dir)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, lastest_checkpoint)
            image = np.array(image).reshape((1, 256, 256, 3))
            feed = {self.alex_model.input_x:image}
            image_repesentation = sess.run(self.alex_model.repsentation, feed_dict=feed)
            feed = {self.input_x: image_repesentation.T}
            label_representation = sess.run(self.prediction, feed_dict=feed)

            label_representation = np.reshape(label_representation, (200))
            most_similar = self.word_model.similar_by_vector(label_representation, topn=1)

        prediction_label = most_similar[0][0]
        return prediction_label






if __name__ == '__main__':
    pre_train()
    train_x = utils.get_image_data()
    label2int, int2label = utils.get_parameter()
    numeral_labels = utils.get_numeral_label_data()
    d_model = DeViSE()
    d_model.train(train_x, numeral_labels, int2label)
    prediction = d_model.predict(train_x[0])
    print('real label is: {}'.format(int2label[numeral_labels[0]]),
          '\n predicting label is: {}'.format(prediction))