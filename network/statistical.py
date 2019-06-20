import numpy as np
from collections import Counter

doc_train = open('../set/train_seg_new.txt', 'r', encoding='utf_8')
doc_test = open('../set/test_seg_new.txt', 'r', encoding='utf_8')

data = {}
for line in doc_train:
    terms = line.strip('\n').split('\t')
    label = terms[0]
    if label in data.keys():
        data[label].append(terms[1])
    else:
        sens = []
        sens.append(terms[1])
        data[label] = sens
for line in doc_test:
    terms = line.strip('\n').split('\t')
    label = terms[0]
    if label in data.keys():
        data[label].append(terms[1])
    else:
        sens = []
        sens.append(terms[1])
        data[label] = sens
all_word = []
for key in data.keys():
    words = {}
    for sens in data[key]:
        for word in sens.split(' '):
            words[word] = words.get(word, 0) + 1
            all_word.append(word)
    topn = sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    print(key, topn)
print(len(all_word))


from bert_serving.client import BertClient
bc = BertClient()
print(bc.encode(['中国', '美国']))