from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence

# 训练模型
sentences = LineSentence('../data/zh_wiki_2gram_char')

# size：词向量的维度
# window：上下文环境的窗口大小
# min_count：忽略出现次数低于min_count的词
model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=10)

model.save('../embedding/word_embedding_300_char_2gram')
#model = Word2Vec.load("../embedding/word_embedding_128")

items = model.most_similar(u'<')
for item in items:
    # 词的内容，词的相关度
    print(item[0], item[1])


print(model['中'])
