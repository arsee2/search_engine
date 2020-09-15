import math
import numpy as np
import matplotlib.pyplot as plt
import json
from search_engine import SearchEngine


def eleven_points_interpolated_avg(top_k_results, relevance, plot=False):
    """
    :param top_k_results: list of lists of ranked results for each query [[doc_id1, doc_id2,...], ...]
                          the i-th result corresponds to (i+1)-th query_id. There may be less than top_k
                          results returned for a query, but never more.
    :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    :param plot: whether to plot the graph or not
    :return: interpolated_avg, list of 11 values
    """
    interpolated_avg = []
    for x in range(11):
        interpolated_avg.append([])

    for i in range(len(top_k_results)):

        curve_points = []
        list_of_docs = top_k_results[i]
        tp = 0
        for j in range(len(list_of_docs)):
            doc = list_of_docs[j]
            if doc in dict(relevance[i + 1]):
                tp += 1
            curve_points.append([tp / len(dict(relevance[i + 1])), tp / (j + 1)])

        curve_points = sorted(curve_points)

        for j in range(0, 11):

            if len([a for a in range(len(curve_points)) if curve_points[a][0] >= .1 * j - 0.0001]) > 0:
                ind = [a for a in range(len(curve_points)) if curve_points[a][0] >= .1 * j - 0.0001][0]
                maxi = np.max([a[1] for a in curve_points][ind:])
                interpolated_avg[j].append(maxi)
            else:
                # interpolated_avg[j].append(0)
                pass

    interpolated_avg = [np.mean(a) for a in interpolated_avg]

    if plot:
        X = np.linspace(0, 1, 11)
        plt.plot(X, interpolated_avg)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()

    return interpolated_avg


def mean_avg_precision(top_k_results, relevance):
    """
    :param top_k_results: list of lists of ranked results for each query [[doc_id1, doc_id2,...], ...]
                          the i-th result corresponds to (i+1)-th query_id. There may be less than top_k
                          results returned for a query, but never more.
    :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    :return: calculated MAP score
    """
    # TODO write your code here
    ans = []
    for i in range(len(top_k_results)):
        tp = 0
        summ = 0
        num = 0
        docs = top_k_results[i]
        for j in range(len(docs)):
            if docs[j] in dict(relevance[i + 1]):
                tp += 1
                summ += tp / (j + 1)
                num += 1
        if num == 0:
            ans.append(0)
            continue
        ans.append(summ / num)

    return np.mean(ans)


def NDCG(top_k_results, relevance, top_k):
    """

    :param top_k_results: list of lists of ranked results for each query [[doc_id1, doc_id2,...], ...]
                          the i-th result corresponds to (i+1)-th query_id. There may be less than top_k
                          results returned for a query, but never more.
    :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    :param top_k: (max) number of results retrieved for each query
                  factor for each query
    :return: NDCG score
    """

    answer = []
    for i in range(len(top_k_results)):

        docs = top_k_results[i]
        summ = 0
        num = 0
        sc = []
        z = 0

        for j in range(min(top_k, len(docs))):
            doc = docs[j]
            if doc in dict(relevance[i + 1]):
                R = 5 - dict(relevance[i + 1])[doc]
                summ += (2 ** R - 1) / np.log2(j + 2)
                sc.append(R)
        for doc in dict(relevance[i + 1]).keys():
            if doc not in docs:
                sc.append(5 - dict(relevance[i + 1])[doc])

        sorted_sc = sorted(sc, reverse=True)
        for j in range(len(sorted_sc)):
            el = sorted_sc[j]
            z += (2 ** el - 1) / np.log2(j + 2)

        if z != 0:
            answer.append(summ / z)
        else:
            answer.append(.0)

    return np.mean(answer)



def relevance_of_query(relevance, query_num):
    '''
    :param relevance: parsed cranfield dataset i
    :param query_num: id of query
    :return: list of tuples (doc_id,position)
    '''
    answer = []
    for rel in relevance:

        if int(rel['query_num']) == query_num:
            answer.append((int(rel['id']), int(rel['position'])))
    return answer


def evaluate(search_engine, path_to_data, k, sim="cos"):
    '''

    :param search_engine: object of SearchEngine class
    :param path_to_data: path to cranfield dataset
    :param k: number of document to retrive
    :return: tuple  (NDCG, mean average precision)
    '''

    queries = [query for query in json.load(open((path_to_data + "/cran.qry.json"))) if query['query number'] != 0]
    relevance = json.load(open((path_to_data + "/cranqrel.json")))
    list_of_docs = []
    relevances = {}
    for query in queries:
        if sim == "cos":
            docs = search_engine.extract_top_k_cos(query['query'], k=k, query_id=int(query['query number']))

        else:
            docs = search_engine.extract_top_k_okapi(query['query'], k=k, query_id=int(query['query number']))
        list_of_docs.append(docs)
        relevances[int(query['query number'])] = relevance_of_query(relevance, int(query['query number']))

    eleven_points_interpolated_avg(list_of_docs,relevances)

    return NDCG(list_of_docs, relevances, top_k=k), mean_avg_precision(list_of_docs, relevances)
