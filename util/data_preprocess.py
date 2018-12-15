import jieba

label_index = {}


def load_data(path):
    x_train = []
    y_train = []

    with open(path, 'r') as file:
        for line in file:
            words = line.strip('\n').split('\t')
            label = words[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            x_train.append(words[1].strip('\n').split(' '))
            y_train.append(label_index[label])
    return x_train, y_train


def load_stopword(path):
    stop_word = []
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file:
            stop_word.append(line.strip('\n'))
    return stop_word


def particle_word(in_path, out_path, stop_path):
    f_test = open(out_path, 'w')
    stop_word = load_stopword(stop_path)
    with open(in_path, 'r') as file:
        for line in file:
            terms = line.strip('\n').split('\t')
            label = terms[0].split('_')[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            words = jieba.cut(terms[1].strip(' '))
            f_test.write(label + '\t' + ' '.join([word for word in words if (len(word) > 0 and word not in stop_word)]) + '\n')