from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from nltk import tokenize


def sentence_sim(sent1, sent2, ):
    '''
    :param sent1: sentence one
    :param sent2: sentencne two
    :return: float number
    '''
    sw = set(stopwords.words('english'))
    sent1 = [w.lower() for w in tokenize.word_tokenize(sent1) if w not in sw]
    sent2 = [w.lower() for w in tokenize.word_tokenize(sent2) if w not in sw]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1

    for w in sent2:
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)


def summary(text, top_n):
    '''
    :param text: string
    :param top_n: integer number
    :return: summary string
    '''
    sentences = tokenize.sent_tokenize(text)

    n = len(sentences)

    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # print(i,j)
            if i == j:
                continue
            similarity_matrix[i][j] = sentence_sim(sentences[i], sentences[j])

    matrix = similarity_matrix
    sentence_similarity_graph = nx.from_numpy_array(matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = ""
    for i in range(top_n):
        summary += (ranked_sentence[i][1])+"\n"
    return summary

