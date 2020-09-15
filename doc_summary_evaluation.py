from query_dependent_summarisation import summary as qds
from first_and_last_summarisation import summary as fls
from text_rank import summary as tr
from nltk import tokenize
from rouge_score import rouge_2,rouge_1

text = open("text_to_sum").read()
real_summary = open("summary one.txt").read()

summary = fls(text)
print("2 first and last sentences")
print("Real summary:", real_summary)
print("Summary",summary)
print("__________________________")
print("rogue 1 score:",(rouge_1(tokenize.word_tokenize(summary),[tokenize.word_tokenize(sent) for sent in tokenize.sent_tokenize(real_summary)] , 0.5)),"\n")


summary = qds(text, "museum", 3)
print("Query dependent")
print("Real summary:", real_summary)
print("Summary",summary)
print("__________________________")
print("rogue 1 score:",(rouge_1(tokenize.word_tokenize(summary),[tokenize.word_tokenize(sent) for sent in tokenize.sent_tokenize(real_summary)] , 0.5)),"\n")

summary = tr(text,3)
print("Text rank")
print("Real summary:", real_summary)
print("Summary",summary)
print("__________________________")
print("rogue 1 score:",(rouge_1(tokenize.word_tokenize(summary),[tokenize.word_tokenize(sent) for sent in tokenize.sent_tokenize(real_summary)] , 0.5)),"\n")


real_summary = open("summary two.txt").read()
text = open("text_to_sum2").read()


summary = fls(text)
print("2 first and last sentences")
print("Real summary:", real_summary)
print("Summary",summary)
print("__________________________")
print("rogue 1 score:",(rouge_1(tokenize.word_tokenize(summary),[tokenize.word_tokenize(sent) for sent in tokenize.sent_tokenize(real_summary)] , 0.5)),"\n")

summary = qds(text, "architecture", 3)
print("Query dependent")
print("Real summary:", real_summary)
print("Summary",summary)
print("__________________________")
print("rogue 1 score:",(rouge_1(tokenize.word_tokenize(summary),[tokenize.word_tokenize(sent) for sent in tokenize.sent_tokenize(real_summary)] , 0.5)),"\n")


summary = tr(text, 3)
print("Text rank")
print("Real summary:", real_summary)
print("Summary",summary)
print("__________________________")
print("rogue 1 score:",(rouge_1(tokenize.word_tokenize(summary),[tokenize.word_tokenize(sent) for sent in tokenize.sent_tokenize(real_summary)] , 0.5)),"\n")