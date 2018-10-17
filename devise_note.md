# DeViSE: A Deep Visual-Semantic Embedding Model

## abstract
1000 known classes and 20,000 unknown classes

## 主要思想
- 设计一个visual-semantic embedding model去识别视觉object，使用labeled iamges和text数据。
- 使用text数据学习label之间的语义关系。
- 使用这个可视化和语义化相似性关系可以正确的预测unseen类别

## 当前研究问题
传统图片分类模型CNN使用softmax进行分类，但是随着类别的增加，类别之间的区别模糊，如果想要区分这个模糊性，需要收集更大量的数据，这很难。

## 方法
使用从text data学习到的语义知识，然后将其转换为一个可以训练 visual object recognition model。
- pre-training a simple neural language model well- suited for learning semantically-meaningful, dense vector representations of words
- pre-train a state-of-the-art deep neural network for visual object recognition [11], complete with a traditional softmax output layer.
- construct a deep visual-semantic model by taking the lower layers of the pre-trained visual object recognition network and re-training them to predict the vector representation of the image label text as learned by the language model.

### 3.1 language model pre-training
- skip-gram :Our skip-gram model used a hierarchical softmax layer for predicting adjacent terms and was trained using a 20-word window with a single pass through the corpus
- trained skip-gram models of varying hidden dimensions, ranging from 100-D to 2,000-D, and found 500- and 1,000-D embeddings to be a good compromise between training speed, semantic quality,

- trained a skip-gram text model on a corpus of 5.7 million documents (5.4 billion words) extracted from wikipedia.org. The text of the web pages was tokenized into a lexicon of roughly 155,000 single- and multi-word terms consisting of common English words and phrases as well as terms from commonly used visual object recognition datasets

### 3.2 visual model pre-training
- CNN: consists of
    1. several convolutional filtering
    2. local contrast normalization
    3. max-pooling layers
    4. several fully connected neural network layers
    5. trained using the dropout technique
    6. a softmax output layer
- This trained model serves both as our benchmark for performance comparisons, as well as the initialization for our joint model.

### 3.3 deep visual-semantic embedding model
- initialized from these two pre-trained neu- ral network models
- embedding vectors learned by the language model are unit normed and used to map label terms into target vector representations
- The core visual model, with its softmax prediction layer now removed, is trained to predict these vectors for each image, by means of a projection layer and a similarity metric. The projection layer is a linear transformation that maps the 4,096-D representation at the top of our core visual model into the 500- or 1,000-D representation native to our language model.

#### loss function
- a combination of dot-product similarity and hinge rank loss
- to produce a higher dot-product similarity between the visual model output and the vector representation of the correct label



## dataset
一共有200种鸟类图片，划分数据集如下：
- 前150种鸟类都作为训练集
- 后50种鸟类作为unseen label进行测试
