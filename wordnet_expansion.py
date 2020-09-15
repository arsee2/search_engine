from nltk.corpus import wordnet as wn


class WordNetExpansion:
    def expand(self, list_of_words):
        '''
        :param list_of_words: list of words
        :return: expanded string
        '''
        answer = list(list_of_words.keys())
        for word in list_of_words:

            try:
                answer.append(wn.synsets(word)[0].lemmas()[0].name())
            except:
                pass
        return " ".join(answer)
