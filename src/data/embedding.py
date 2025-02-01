import torch
import pandas as pd
from transformers import BertModel, BertTokenizerFast
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

def bert_pca(input_csv_path, batch_size=8, pca_components=128, model_name="bert-base-uncased"):
    """
    주어진 CSV 파일의 'overview' 컬럼에 대해 BERT 임베딩을 생성한 후,
    PCA를 적용하여 차원을 축소한 임베딩을 반환합니다.
    
    Parameters:
        input_csv_path (str): 입력 CSV 파일 경로. 'overview' 컬럼을 포함해야 합니다.
        batch_size (int, optional): 한 번에 처리할 텍스트 개수 (기본값: 8).
        pca_components (int, optional): PCA로 축소할 차원 수 (기본값: 128).
        model_name (str, optional): 사용할 BERT 모델 이름 (기본값: "bert-base-uncased").
    
    Returns:
        numpy.ndarray: PCA 적용 후 임베딩 배열.
    """
    df = pd.read_csv(input_csv_path)
    overview_texts = df['overview'].dropna().tolist()
    
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    sentence_embeddings = []
    
    for i in range(0, len(overview_texts), batch_size):
        batch = overview_texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        sentence_embeddings.append(outputs.pooler_output.cpu())
    
    sentence_embeddings_np = torch.cat(sentence_embeddings, dim=0).numpy()
    pca = PCA(n_components=pca_components)
    reduced_embeddings_pca = pca.fit_transform(sentence_embeddings_np)
    
    return reduced_embeddings_pca

def word2vec(input_csv_path, vector_size=100, window=5, min_count=1, workers=4):
    """
    주어진 CSV 파일의 'overview' 컬럼 텍스트에 대해 Word2Vec 임베딩을 생성한 후,
    각 단어의 벡터를 DataFrame 형태로 반환합니다.
    
    Parameters:
        input_csv_path (str): 입력 CSV 파일 경로. 'overview' 컬럼을 포함해야 합니다.
        vector_size (int, optional): 임베딩 벡터의 차원 (기본값: 100).
        window (int, optional): Word2Vec 모델의 윈도우 크기 (기본값: 5).
        min_count (int, optional): 고려할 최소 단어 빈도 (기본값: 1).
        workers (int, optional): 모델 학습 시 사용할 프로세스 수 (기본값: 4).
    
    Returns:
        pandas.DataFrame: 각 단어의 벡터를 담은 DataFrame (인덱스는 단어).
    """
    item = pd.read_csv(input_csv_path)
    item['overview'] = item['overview'].fillna("")
    
    sentences = [str(text).split() for text in item['overview']]
    
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    
    word_vectors = model.wv
    vocabs = list(word_vectors.key_to_index.keys())
    word_vectors_list = [word_vectors[v] for v in vocabs]
    
    return pd.DataFrame(word_vectors_list, index=vocabs)
