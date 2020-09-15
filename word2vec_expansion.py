import pickle

import gensim as gensim
import numpy as np
import pandas as pd
from nltk import TweetTokenizer
from nltk.corpus import stopwords

import os


class Word2VecExpansion:

    def __init__(self):
        """
        Load word2vec model from file if file exists otherwise create new model and save it on disk
        """
        print("Word2Vec initialisation")
        if os.path.exists("word2vec_model.pickle"):
            with open('word2vec_model.pickle', 'rb') as file:
                self.word2vec_model = pickle.load(file)
            print("Word2Vec was loaded from file")
            return
        word2vec_size = 32
        path_to_dataset = "E:\\Dataset\\questions.csv"
        data = np.concatenate((pd.read_csv(path_to_dataset).question1.astype(str).values,
                               pd.read_csv(path_to_dataset).question2.astype(str).values))
        stop_words = set(stopwords.words('english'))
        tk = TweetTokenizer()

        def words_preprocessing(list_of_words):
            words = [w.lower() for w in list_of_words if w.isalpha()]
            words = [w for w in words if w not in stop_words]
            return words

        def sentences_preprocessing(list_of_string):
            return [words_preprocessing(tk.tokenize(text)) for text in list_of_string]

        sentences = sentences_preprocessing(data)
        print("Word2Vec learning")

        self.word2vec_model = gensim.models.Word2Vec(
            sentences,
            size=word2vec_size,
            window=5,
            min_count=2,
            workers=10)
        self.word2vec_model.train(sentences, total_examples=len(sentences), epochs=20)
        print("Word2Vec initialisation is finished")
        with open('word2vec_model.pickle', 'wb') as file:
            pickle.dump(self.word2vec_model, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model has been saved")

    def word2vec_expansion(self, list_of_words):
        """

        :param list_of_words: list of words
        :return: return string with original and expanded words joint by space
        """
        answer = list(list_of_words.keys())

        for word in list_of_words:
            if word in self.word2vec_model:

                answer.append(self.word2vec_model.most_similar(word)[0][0])

        return " ".join(answer)
