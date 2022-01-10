import numpy as np

def preprocess(text):
    '''
    input:
        text
    output:
        corpus: テキストをidで置き換えたarray
        word_to_id: 単語とその出現順(id)を対応させた辞書
        id_to_word: 単語の出現順とその単語を対応させた辞書
    '''
    text = text.lower() # 小文字に変換
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


# 共起行列作成
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus): # 1単語ごとに見ていく
        for i in range(1, window_size+1): # ウィンドウ内を見る
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix


# コサイン類似度
## 同じ向きを向いていると1, 正反対に向いていると-1
def cos_similarity(x, y, eps=1e-8):
    # 0除算を回避するために、分母にepsを足す
    nx = x / np.sqrt(np.sum(x**2) + eps) # xの正規化
    ny = y / np.sqrt(np.sum(y**2) + eps) # yの正規化
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 1.クエリを取り出す
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    
    print('\n[query] ' + query)
    query_id = word_to_id[query] # query(単語)のidを取り出す
    query_vec = word_matrix[query_id] # vec化

    # 2.コサイン類似度の算出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # コサイン類似度の高い結果から、高い順に出力
    count = 0
    for i in (-1 * similarity).argsort(): # 要素の値が小さい順にソート。インデックスが返る。-1をかけることで、降順にソートする。
        if id_to_word[i] == query: # コサイン類似度が大きい順に単語を探し、queryと同じだったらその値を出力。
            continue
        print( '%s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


# Positive PMI(正の相互情報量)
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
        
    return M