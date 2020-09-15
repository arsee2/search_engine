import json

from evaluation import evaluate
from search_engine import SearchEngine

# Change this path
PATH_TO_DATA = "C://Users//arsee//Desktop//lab8//test_dir//data"


def build_search_engine(type=None, relevance=None):
    """
    :param type: string. type of query expansion can be one of these :{Rocchio,Word2Vec,WordNet,None}
    :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    :return: Search engine with specified
    """
    documents = {}

    for document in json.load(open((PATH_TO_DATA + "/cranfield_data.json"))):
        # documents[document['id']] = document["author"] + "\n" + " " + document['body'] + "\n" + document['title']
        documents[document['id']] = document['body']


    return SearchEngine(documents, type, relevance)


def relevance_of_query(relevance, query_num):
    '''
    :param relevance: parsed cranfield dataset i
    :param query_num: id of query
    :return: list of tuples (doc_id,position)
    '''
    answer = []
    for rel in relevance:

        if rel['query_num'] == str(query_num):
            answer.append((int(rel['id']), int(rel['position'])))
    return answer


# read quries
queries = [query for query in json.load(open((PATH_TO_DATA + "/cran.qry.json"))) if query['query number'] != 0]
# read relevance
relevance = json.load(open((PATH_TO_DATA + "/cranqrel.json")))
list_of_docs = []
relevances = {}
# transform relevance to convenient form
for query in queries:
    relevances[int(query['query number'])] = relevance_of_query(relevance, query['query number'])




search_engine = build_search_engine("WordNet", relevance=relevances)
print("NDCG and mean average precision using WordNet expansion and cosine similarity: ",
      evaluate(search_engine, path_to_data=PATH_TO_DATA, k=30, sim="cos"))

search_engine = build_search_engine("Rocchio", relevance=relevances)
print("NDCG and mean average precision using Rocchio expansion and cosine similarity: ",
      evaluate(search_engine, path_to_data=PATH_TO_DATA, k=30, sim="cos"))

search_engine = build_search_engine("PseudoRelevant", relevance=relevances)
print("NDCG and mean average precision using pseudo relevant expansion and cosine similarity: ",
      evaluate(search_engine, path_to_data=PATH_TO_DATA, k=30, sim="cos"))

search_engine = build_search_engine(relevance=relevances)
print("NDCG and mean average precision without expansion and cosine similarity: ",
      evaluate(search_engine, path_to_data=PATH_TO_DATA, k=30, sim="cos"))


search_engine = build_search_engine("WordNet", relevance=relevances)
print("NDCG and mean average precision using WordNet expansion and okapi score: ",
      evaluate(search_engine, path_to_data=PATH_TO_DATA, k=30, sim="okapi"))

search_engine = build_search_engine(relevance=relevances)
print("NDCG and mean average precision without expansion and okapi score: ",
      evaluate(search_engine, path_to_data=PATH_TO_DATA, k=30, sim="okapi"))

search_engine = build_search_engine("Word2Vec", relevance=relevances)
print("NDCG and mean average precision using Word2Vec expansion and cosine similarity: ",
      evaluate(search_engine, path_to_data=PATH_TO_DATA, k=30, sim="cos"))

search_engine = build_search_engine("Word2Vec", relevance=relevances)
print("NDCG and mean average precision using Word2Vec expansion and okapi score : ",
      evaluate(search_engine, path_to_data=PATH_TO_DATA, k=30, sim="okapi"))