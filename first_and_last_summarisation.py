from nltk import tokenize
import re

def summary(text):
    """

    :param text: text of document
    :return: string with document summary
    """
    sentences = tokenize.sent_tokenize(text)
    sentences = [sent for sent in sentences if len(sent)>10]

    return sentences[0]+"\n"+sentences[1]+"\n"+sentences[-2]+"\n"+sentences[-1]
