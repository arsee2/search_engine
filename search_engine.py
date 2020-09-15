from collections import OrderedDict

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from word2vec_expansion import Word2VecExpansion
from wordnet_expansion import WordNetExpansion
import numpy as np
import math


def inc_val(dict, key, val):
    if key in dict:
        dict[key] += val
    else:
        dict[key] = val


class SearchEngine:

    def words_preprocessing(self, list_of_words):
        '''
        :param list_of_words: list of words
        :return: list of words
        '''
        stemmer = PorterStemmer()
        words = [w.lower() for w in list_of_words]
        words = [w for w in words if w not in self.stop_words]
        return words

    def sentence_preprocessing(self, text):
        '''
        :param text: string
        :return: list of words
        '''
        return self.words_preprocessing(self.tk.tokenize(text))

    def query_preprocessing(self, query):
        '''
        :param query: string
        :return: dictionary. word:tf
        '''
        query = self.sentence_preprocessing(query)
        dictionary = {}
        for word in query:
            if word not in dictionary:
                dictionary[word] = 0
            dictionary[word] += 1;
        return dictionary

    def build_inverted_index(self, documents):
        '''
        :param documents: dictionary. doc_id: text
        '''
        dictionary = {}
        for doc_id, text in documents.items():
            terms = self.sentence_preprocessing(text)
            self.doc_lengths[doc_id] = len(terms)
            self.direct_index[doc_id] = terms
            for t in terms:
                if t in dictionary:
                    dictionary[t]['term_frequency'] = dictionary[t]['term_frequency'] + 1
                    if doc_id not in dictionary[t]:
                        dictionary[t][doc_id] = 0;
                    dictionary[t][doc_id] = dictionary[t][doc_id] + 1
                else:
                    dictionary[t] = OrderedDict()
                    dictionary[t]['term_frequency'] = 1
                    dictionary[t][doc_id] = 1

        for term, d in dictionary.items():
            self.inverted_index[term] = ([d['term_frequency']])

            for i, j in d.items():
                if i == "term_frequency":
                    continue
                self.inverted_index[term].append((int(i), int(j)))

    def __init__(self, documents, query_expansion=None, relevance=None):
        '''

        :param documents: dictionary of doc_id:text
        :param query_expansion: string. type of query expansion can be one of these :{Rocchio,Word2Vec,WordNet,None}
        :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
        '''
        self.doc_lengths = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.tk = TweetTokenizer()
        self.inverted_index = {}
        self.direct_index = {}
        self.build_inverted_index(documents)
        self.query_expansion = query_expansion
        if query_expansion == "Word2Vec":
            self.word2vec_exp = Word2VecExpansion()
        if query_expansion == "Rocchio":
            self.relevance = relevance

    def cosine_scoring(self, query):

        '''
        :param query: dict, string: number
        :return: dictionary doc_id: score
        '''
        scores = {}
        index = self.inverted_index
        doc_lengths = self.doc_lengths
        for i, j in doc_lengths.items():
            scores[i] = 0
        for term, tf in query.items():
            if term not in index:
                continue
            for i in range(1, len(index[term])):
                idf = math.log10(len(doc_lengths) / (len(index[term])))
                scores[index[term][i][0]] += tf * index[term][i][1] * idf * idf

        for i, j in doc_lengths.items():
            if j == 0:
                scores[i] = 0
                continue
            scores[i] /= j

        return {x: y for x, y in scores.items()}

    def pseudo_relevant_expansion(self, q, k, a=1, b=0.75, y=0.2, n=10):
        '''
        :param n: number of documents that will be considered as relevant
        :param query: dictionary term:tf
        :param k: top k documents to retrive
        :return: top k retrieved documents
        :param a: a param of rocchio
        :param b: b param of rocchio
        :param y: y param of rocchio
        '''
        query = dict(q)

        documents = self.cosine_scoring(query)

        sorted_docs = sorted(documents.items(), key=lambda x: x[1], reverse=True)
        retrieved = list(dict(sorted_docs[0:k]).keys())
        positive_docs = []
        negative_docs = []
        for doc in retrieved[0:int(n)]:
            positive_docs.append(doc)
            for term in self.direct_index[doc]:
                if term not in query:
                    query[term] = 0

        for doc in retrieved[int(n):]:
            negative_docs.append(doc)
            for term in self.direct_index[doc]:
                if term not in query:
                    query[term] = 0

        scores = {}
        index = self.inverted_index
        doc_lengths = self.doc_lengths
        positive_vector = {}
        negative_vector = {}
        for i, j in doc_lengths.items():
            scores[i] = 0
        for term, tf in query.items():
            positive_vector[term] = 0
            negative_vector[term] = 0
            if term not in index:
                continue
            for i in range(1, len(index[term])):
                if index[term][i][0] in positive_docs:
                    positive_vector[term] += index[term][i][1] / len(positive_docs)
                if index[term][i][0] in negative_docs:
                    negative_vector[term] += index[term][i][1] / len(negative_docs)

        for term, tf in query.items():
            if term not in index:
                continue
            for i in range(1, len(index[term])):
                idf = math.log10(len(doc_lengths) / (len(index[term]) - 1))
                scores[index[term][i][0]] += (a * tf + b * positive_vector[term] - y * negative_vector[term]) * \
                                             index[term][i][1] * idf * idf

        for i, j in doc_lengths.items():
            if j == 0:
                scores[i] = 0
                continue
            scores[i] /= math.sqrt(j)

        return scores

    def rocchio_expansion(self, query, k, query_id, a=1, b=0.75, y=0.15):
        '''
        :param query: dictionary term:tf
        :param k: top k documents to retrive
        :param query_id: id of query
        :param a: a param of rocchio
        :param b: b param of rocchio
        :param y: y param of rocchio
        :return: top k retrieved documents
        '''
        documents = self.cosine_scoring(query)
        sorted_docs = sorted(documents.items(), key=lambda x: x[1], reverse=True)
        retrieved = list(dict(sorted_docs[0:k]).keys())
        negative_docs = []
        positive_docs = []

        for doc in retrieved:
            if query_id not in self.relevance:
                continue
            if doc in dict(self.relevance[query_id]):
                positive_docs.append(doc)
                for term in self.direct_index[doc]:
                    if term not in query:
                        query[term] = 0

            else:
                negative_docs.append(doc)
                for term in self.direct_index[doc]:
                    if term not in query:
                        query[term] = 0

        scores = {}
        index = self.inverted_index
        doc_lengths = self.doc_lengths
        positive_vector = {}
        negative_vector = {}
        for i, j in doc_lengths.items():
            scores[i] = 0
        for term, tf in query.items():
            positive_vector[term] = 0
            negative_vector[term] = 0
            if term not in index:
                continue
            for i in range(1, len(index[term])):
                if index[term][i][0] in positive_docs:
                    positive_vector[term] += index[term][i][1] / len(positive_docs)
                if index[term][i][0] in negative_docs:
                    negative_vector[term] += index[term][i][1] / len(negative_docs)

        for term, tf in query.items():
            if term not in index:
                continue
            for i in range(1, len(index[term])):
                idf = math.log10(len(doc_lengths) / (len(index[term]) - 1))
                scores[index[term][i][0]] += (a * tf + b * positive_vector[term] - y * negative_vector[term]) * \
                                             index[term][i][1] * idf * idf

        for i, j in doc_lengths.items():
            if j == 0:
                scores[i] = 0
                continue
            scores[i] /= math.sqrt(j)

        return scores

    def extract_top_k_cos(self, query, k, query_id=None):
        '''
        :param query: text
        :param k: top k document to retrieve
        :param query_id: query id is needed for local expansion
        :return: top k retrieved documents
        '''
        query = self.query_preprocessing(query)
        if self.query_expansion is None:
            documents = self.cosine_scoring(query)
        if self.query_expansion == "Word2Vec":
            documents = self.cosine_scoring(self.query_preprocessing(self.word2vec_exp.word2vec_expansion(query)))
        if self.query_expansion == "WordNet":
            documents = self.cosine_scoring(self.query_preprocessing(WordNetExpansion().expand(query)))
        if self.query_expansion == "Rocchio":
            documents = self.rocchio_expansion(query, k, query_id)
        if self.query_expansion == "PseudoRelevant":
            documents = self.pseudo_relevant_expansion(query, k)

        sorted_docs = sorted(documents.items(), key=lambda x: x[1], reverse=True)

        return list(dict(sorted_docs[0:k]).keys())

    def extract_top_k_okapi(self, query, k, query_id=None):
        '''
        :param query: text
        :param k: top k document to retrieve
        :param query_id: query id is needed for local expansion
        :return: top k retrieved documents
        '''
        query = self.query_preprocessing(query)
        if self.query_expansion is None:
            documents = self.okapi_scoring(query)
        if self.query_expansion == "Word2Vec":
            documents = self.okapi_scoring(self.query_preprocessing(self.word2vec_exp.word2vec_expansion(query)))
        if self.query_expansion == "WordNet":
            documents = self.okapi_scoring(self.query_preprocessing(WordNetExpansion().expand(query)))
        if self.query_expansion == "Rocchio":
            return self.rocchio_expansion(query, k, query_id)
        if self.query_expansion == "PseudoRelevant":
            return self.pseudo_relevant_expansion(query, k)
        sorted_docs = sorted(documents.items(), key=lambda x: x[1], reverse=True)

        return list(dict(sorted_docs[0:k]).keys())

    def okapi_scoring(self, query, k1=1.2, b=0.75):
        '''
       :param query: dict, string: number
       :return: dictionary doc_id: score
       '''

        index = self.inverted_index
        doc_lengths = self.doc_lengths
        averageD = 0
        for i, j in doc_lengths.items():
            averageD += j
        averageD /= len(doc_lengths)
        answer = {}
        for doc_id, length in doc_lengths.items():
            ans = 0
            for i, j in query.items():
                if i not in index:
                    continue
                idf = math.log10(len(doc_lengths) / (len(index[i]) - 1))
                tf = 0
                for d in range(1, len(index[i])):
                    if index[i][d][0] == doc_id:
                        tf = tf + index[i][d][1]
                ans = ans + idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (length / averageD)))
            answer[doc_id] = ans

        return answer
