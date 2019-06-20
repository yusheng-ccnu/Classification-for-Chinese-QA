import jieba

label_index = {}


def load_data(path):
    x_ = []
    y_ = []
    label_index.clear()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip('\n').strip('\ufeff').split('\t')
            label = words[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            x_.append(words[1].strip('\n').split(' '))
            y_.append(label_index[label])
    return x_, y_


def load_data_line(path):
    x_ = []
    y_ = []
    label_index.clear()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip('\n').strip('\ufeff').split('\t')
            label = words[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            x_.append(words[1].strip('\n'))
            y_.append(label_index[label])
    return x_, y_


def load_stopword(path):
    stop_word = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            stop_word.append(line.strip('\n'))
    return stop_word


def particle_word(in_path, out_path, stop_path):
    label_index.clear()
    f_test = open(out_path, 'w', encoding='utf-8')
    stop_word = load_stopword(stop_path)
    with open(in_path, 'r', encoding='utf-8') as file:
        for line in file:
            terms = line.strip('\n').split('\t')
            label = terms[0].split('_')[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            words = jieba.cut(terms[1].strip(' '))
            f_test.write(label + '\t' + ' '.join([word for word in words if (len(word) > 0 and word not in stop_word)]) + '\n')


def get_labels():
    return label_index


def particle_word_char(in_path, out_path):
    label_index.clear()
    f_test = open(out_path, 'w', encoding='utf-8')
    with open(in_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = []
            terms = line.strip('\n').split('\t')
            label = terms[0].split('_')[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            for word in terms[1]:
                words.append(word)
            f_test.write(label + '\t' + ' '.join(words) + '\n')


labels = {}


def load_data_pred(path, data_path):
    x_ = []
    y_ = []
    x_data = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip('\n')
            x_.append(words)
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip('\n')
            x_data.append(words.strip('\n'))
    return x_, y_, x_data


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


def n_gram(n=2):
    x_train, y_train = load_data('../data/test_seg_char.txt')
    x_train_new = []
    for line in x_train:
        new_line = []
        for i in range(len(line) - n + 1):
            new_line.append(''.join(line[i: i + n]))
        x_train_new.append(new_line)
    f = open('../data/test_seg_' + str(n) + 'gram.txt', 'w', encoding='utf-8')
    print(x_train_new)
    for i in range(len(x_train_new)):
        f.write(str(y_train[i]) + '\t' + ' '.join(x_train_new[i]) + '\n')

#n_gram(n=2)
#particle_word('../question/hum.txt', '../question/HUM.txt', '../data/stopword.txt')
#particle_word('../data/nlp_qa.txt', '../data/nlp_question.txt', '../data/stopword.txt')
#print(label_index)
#max_count('../data/train_question.txt')
#particle_word_char('../data/testquestion.txt', '../data/test_seg_char.txt')
#print(len('\n'))
