import jieba

label_index = {}


def load_stopword(path):
    stop_word = []
    with open(path) as file:
        for line in file:
            stop_word.append(line.strip('\n'))
    return stop_word


#字符分词
def load_data(in_path, out_path, stop_path):
    out_file = open(out_path, 'w')
    stop_word = load_stopword(stop_path)
    with open(in_path, 'r') as file:
        for line in file:
            words = line.strip('\n').split('\t')
            label = words[0].split('_')
            if label not in label_index:
                label_index[label] = len(label_index)
            terms = [term for term in words[1] if term not in stop_word]
            terms.append('__label__' + str(label_index[label]))
            out_file.write(' '.join(terms))


#汉语分词
def load_jieba_data(in_path, out_path, stop_path):
    f_test = open(out_path, 'w')
    stop_word = load_stopword(stop_path)
    with open(in_path, 'r') as file:
        for line in file:
            words = jieba.cut(line)
            f_test.write(' '.join([word for word in words if word not in stop_word]))



