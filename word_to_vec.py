from gensim.models import word2vec

word_embedding_size = 200
corpus_dir = 'text8_data/text8'
word2vec_saved_dir = 'word2vec_saved/text8model.model'


def train_word2vec():
    sentences = word2vec.Text8Corpus('text8_data/text8')
    print('load text8 dataset...')
    model = word2vec.Word2Vec(sentences, size=word_embedding_size)
    model.save(word2vec_saved_dir)
    print('word2vec model has been trained and saved')


def get_word2vec():
    model = word2vec.Word2Vec.load(word2vec_saved_dir)
    return model


if __name__ == '__main__':
    train_word2vec()