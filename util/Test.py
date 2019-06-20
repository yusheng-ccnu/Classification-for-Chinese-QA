def _word_ngrams(tokens, stop_words=None, ngram_range=(1, 1)):
    """Turn tokens into a sequence of n-grams after stop words filtering"""
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

    # handle token n-grams
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n,
                        min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens.append(" ".join(original_tokens[i: i + n]))

    return tokens


text = "我去云南旅游，不仅去了玉龙雪山，还去丽江古城，很喜欢丽江古城"
import jieba

cut = jieba.cut(text)
listcut = list(cut)
print(listcut)
n_gramWords = _word_ngrams(tokens=listcut, ngram_range=(2, 2))
print(n_gramWords)
