import random
import math
import jieba


#根据大类别标签分词数据
def parse_label():
    out = open('../dataset/test.txt', 'w', encoding='utf-8')
    with open('../data/testquestion.txt', 'r', encoding='utf-8') as file:
        for sentence in file:
            sens = sentence.strip('\n').split('\t')
            label = sens[0].split('_')[0]
            out.write(label + '\t' + sens[1] + '\n')


#数据解析
def parse_data():
    train = open('../dataset/train_data.txt', 'r', encoding='utf-8')
    test = open('../dataset/test_data.txt', 'r', encoding='utf-8')
    hum = open('../question/des.txt', 'r', encoding='utf-8')
    time = open('../question/time1.txt', 'r', encoding='utf-8')
    data_set = {}
    for sentence in train:
        sen = sentence.strip('\n').split('\t')
        sens = []
        label = sen[0].strip('\ufeff')
        if label in data_set:
            data_set[label].append(sen[1])
        else:
            data_set[label] = sens

    for sentence in test:
        sen = sentence.strip('\n').split('\t')
        sens = []
        label = sen[0].strip('\ufeff')
        if label in data_set:
            data_set[label].append(sen[1])
        else:
            data_set[label] = sens
    for sentence in hum:
        sen = sentence.strip('\n').split('\t')
        sens = []
        if sen[0] in data_set:
            data_set[sen[0]].append(sen[1])
        else:
            data_set[sen[0]] = sens
    for sentence in time:
        sen = sentence.strip('\n').split('\t')
        sens = []
        if sen[0] in data_set:
            data_set[sen[0]].append(sen[1])
        else:
            data_set[sen[0]] = sens
    return data_set


def divi_data():
    data_set = parse_data()
    test_set = {}
    train_set = {}
    for label in data_set:
        sample = data_set[label]
        random.shuffle(sample)
        test_set[label] = sample[:math.ceil(len(data_set[label]) * 0.2)]
        train_set[label] = sample[math.ceil(len(data_set[label]) * 0.2):]
    return train_set, test_set


def write_data_tofile():
    train_set, test_set = divi_data()
    print('train_set, hum({}),loc({}),num({}),time({}),obj({}),des({})'.
          format(len(train_set['HUM']),
                 len(train_set['LOC']),
                 len(train_set['NUM']),
                 len(train_set['TIME']),
                 len(train_set['OBJ']),
                 len(train_set['DES'])))
    print('test_set, hum({}),loc({}),num({}),time({}),obj({}),des({})'.
          format(len(test_set['HUM']),
                 len(test_set['LOC']),
                 len(test_set['NUM']),
                 len(test_set['TIME']),
                 len(test_set['OBJ']),
                 len(test_set['DES'])))
    train = open('../set/train_data.txt', 'w', encoding='utf-8')
    test = open('../set/test_data.txt', 'w', encoding='utf-8')
    for label in train_set:
        for sen in train_set[label]:
            train.write(label + '\t' + sen + '\n')
    for label in test_set:
        for sen in test_set[label]:
            test.write(label + '\t' + sen + '\n')


def load_stopword(path):
    stop_word = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            stop_word.append(line.strip('\n'))
    return stop_word


def word_seg():
    label_index = {}
    train = open('../irdata/test_seg.txt', 'w', encoding='utf-8')
    stop_word = load_stopword('../data/stop.txt')
    with open('../data/testquestion.txt', 'r', encoding='utf-8') as file:
        for line in file:
            terms = line.strip('\n').split('\t')
            label = terms[0].split('_')[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            words = list(jieba.cut(terms[1].strip(' ')))
            train.write(label + '\t' + ' '.join([word for word in words if (len(word) > 0 and word.strip(' ') not in stop_word)]) + '\n')


def word_seg_char():
    label_index = {}
    train = open('../irdata/train_seg_char.txt', 'w', encoding='utf-8')
    stop_word = load_stopword('../data/stop.txt')
    with open('../data/trainquestion.txt', 'r', encoding='utf-8') as file:
        for line in file:
            words = []
            terms = line.strip('\n').split('\t')
            label = terms[0].split('_')[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            for word in terms[1]:
                words.append(word)
            train.write(label + '\t' + ' '.join(words) + '\n')


#n_gram分词
def word_n_grams(tokens, stop_words=None, ngram_range=(1, 1)):
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens.append("".join(original_tokens[i: i + n]))
    return tokens


def max_count(path):
    max = 0;
    maxList = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip('\n').split('\t')
            if max < len(words[1].split(' ')):
                max = len([word for word in words[1].split(' ') if len(word) > 0])
                maxList = [word for word in words[1].split(' ') if len(word) > 0]
    print(max)
    print(maxList)


word_seg()