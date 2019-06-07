ToxicComments
=====

####performance,train time iteration 319142:

    bi-lstm 0.98439
    double bi-lstm 0.98351 (effect of two biLSTM is not sofficient)
    bi-lstm-deep conv 0.98081 (overfit)
    lstm 0.98140

####report:
    
    competition:    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview
    evaluation:     6 class each binary, 
                    column-wise ROC AUC, 
                    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#evaluation
    dataset:        dir data/train.csv, data/test.csv
    technology:     deep learning
                    CNN http://cs231n.github.io/convolutional-networks/
                    lstm:   http://colah.github.io/posts/2015-08-Understanding-LSTMs/,
                            自己写的，扔了就跑 ： https://zhuanlan.zhihu.com/p/35756075
                    bi-lstm: 
                    word embedding: https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa
                    GloVe: embedding/glove.pdf
                    keras: https://keras.io/zh/preprocessing/text/
    network:        tokenize -> word embedding -> (bi-lstm or lstm) -> CNN (3*3) (or deeper, see lstm_keras.py) -> pooling -> dense net -> 6 binary sigmoid for classify

 
                            
    
    
