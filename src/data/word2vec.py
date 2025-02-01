import pandas as pd
from gensim.models import Word2Vec

item = pd.read_csv('data/train/details.csv')
item['overview'] = item['overview'].fillna("")

sentences = [str(text).split() for text in item['overview']]  

#일단 차원 100으로 설정해뒀음. bert와 다르게 그냥 처음부터 vector_size를 설정해주면 됨.
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4) 

word_vectors = model.wv
vocabs = list(word_vectors.key_to_index.keys())  
word_vectors_list = [word_vectors[v] for v in vocabs] 

word_vectors = pd.DataFrame(word_vectors_list)

word_vectors = pd.DataFrame(word_vectors_list)
word_vectors.to_csv('data/train/word_vectors.csv', index=False)