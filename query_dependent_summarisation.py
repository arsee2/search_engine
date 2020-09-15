from nltk import tokenize


def summary(text, query, top_n):
    sentences = tokenize.sent_tokenize(text.lower())

    query = tokenize.word_tokenize(query.lower())

    relevantness = {}
    for word in query:
        for sent in sentences:
            words = tokenize.word_tokenize(sent)
            if sent not in relevantness:
                relevantness[sent] = 0
            relevantness[sent] += words.count(word)

    most_relevant = dict(sorted(relevantness.items(), key=lambda x: x[1], reverse=True)[0:top_n]).keys()
    sum = ""

    for sent in most_relevant:
        sum += sent

    return sum
