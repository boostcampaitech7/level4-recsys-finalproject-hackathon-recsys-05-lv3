import torch
import pandas as pd
from transformers import BertModel, BertTokenizerFast
from sklearn.decomposition import PCA  

df = pd.read_csv('data/train/details.csv')

overview_texts = df['overview'].dropna().tolist()

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

batch_size = 8
sentence_embeddings = []

for i in range(0, len(overview_texts), batch_size):
    batch = overview_texts[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    sentence_embeddings.append(outputs.pooler_output.cpu()) 

sentence_embeddings_np = torch.cat(sentence_embeddings, dim=0).numpy()
print("원본 임베딩 크기:", sentence_embeddings_np.shape)  

pca = PCA(n_components=128)
reduced_embeddings_pca = pca.fit_transform(sentence_embeddings_np)

print("PCA 적용 후 임베딩 크기:", reduced_embeddings_pca.shape) 

df_embeddings_pca = pd.DataFrame(reduced_embeddings_pca)
df_embeddings_pca.to_csv('data/train/sentence_embeddings_pca.csv', index=False)