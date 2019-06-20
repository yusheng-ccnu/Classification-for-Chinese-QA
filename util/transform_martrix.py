from gensim.models import Word2Vec


def transform_to_matrix(padding_size = 25, vec_size = 300, X=[]):
    res = []
    if vec_size == 100:
        model = Word2Vec.load('../embedding/word_embedding_100')
    elif vec_size == 200:
        model = Word2Vec.load('../embedding/word_embedding_200')
    else:
        model = Word2Vec.load('../embedding/word_embedding_300')
    for sen in X:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res


def transform_to_matrix_char(padding_size=25, vec_size=300, X=[]):
    res = []
    if vec_size == 100:
        model = Word2Vec.load('../embedding/word_embedding_100_char')
    elif vec_size == 200:
        model = Word2Vec.load('../embedding/word_embedding_200_char')
    else:
        model = Word2Vec.load('../embedding/word_embedding_300_char')
    for sen in X:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res


def transform_to_matrix_gram(padding_size=25, vec_size=300, X=[]):
    res = []
    if vec_size == 100:
        model = Word2Vec.load('../embedding/word_embedding_100_char_2gram')
    elif vec_size == 200:
        model = Word2Vec.load('../embedding/word_embedding_200_char_2gram')
    else:
        model = Word2Vec.load('../embedding/word_embedding_300_char_2gram')
    for sen in X:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res