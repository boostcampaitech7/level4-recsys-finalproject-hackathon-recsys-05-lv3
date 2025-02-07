import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_user_history(df):
    """
    사용자별로 (title, genres) 리스트를 만들고,
    user_query(문자열)을 생성하여 반환.
    
    df 컬럼 예: [userId, movieId, rating, title, genres, ...]
    """
    user_group = (df.groupby('userId')[['title','genres']]
                    .apply(lambda x: list(zip(x['title'], x['genres'])))
                    .reset_index(name='title_genres_list'))

    
    def make_user_query(title_genres_pairs):
        if not title_genres_pairs:
            return "The user has no preferred items."
        
        items_str = [f"{t} ({g})" for (t, g) in title_genres_pairs]
        query_text = "The user has interacted with: " + ", ".join(items_str)
        return query_text

    user_group['user_query'] = user_group['title_genres_list'].apply(make_user_query)
    
    return user_group


def sample_two_cold_items(all_items, seen_list):
    """
    간단 예시: 
    - all_items: 전체 (title, genres) 튜플 리스트
    - seen_list: 사용자가 이미 본 (title, genres) 리스트
    
    unseen_list = all_items - seen_list
    그 중 2개를 랜덤 샘플링
    """
    unseen_list = list(set(all_items) - set(seen_list))
    if len(unseen_list) < 2:
        raise ValueError("Not enough unseen items.")
    return random.sample(unseen_list, 2)


def make_pairwise_prompt(user_query, itemA, itemB):
    """
    쌍별 비교 프롬프트를 생성
    """
    prompt = (
        f"{user_query}\n\n"
        "Predict which product the user would prefer:\n"
        f"A: {itemA[0]} ({itemA[1]})\n"
        f"B: {itemB[0]} ({itemB[1]})\n"
        "Answer A or B."
    )
    return prompt


def create_pairwise_df(user_history_df, all_items, n_samples=3):
    """
    user_history_df: create_user_history()에서 생성된 DF
      [userId, title_genres_list, user_query]
    
    all_items: 전체 (title, genres) 튜플 리스트
    n_samples: 사용자당 샘플링 횟수
    """
    results = []
    for idx, row in user_history_df.iterrows():
        user_id = row['userId']
        user_query = row['user_query']
        seen_list = row['title_genres_list']
        
        for _ in range(n_samples):
            try:
                itemA, itemB = sample_two_cold_items(all_items, seen_list)
            except ValueError:
                # unseen이 2개 미만이면 스킵
                continue
            
            prompt = make_pairwise_prompt(user_query, itemA, itemB)
            
            results.append({
                'userId': user_id,
                'prompt': prompt,
                'itemA': itemA,
                'itemB': itemB
            })
            
    return pd.DataFrame(results)


def load_model_and_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cpu"):
    """
    Hugging Face 모델과 토크나이저를 로딩
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def generate_text_batch(df, model, tokenizer, device="cpu", max_new_tokens=50):
    """
    pairwise_df에서 각 'prompt'로부터 model_response를 생성하여 df에 추가
    """
    results = []
    for idx, row in df.iterrows():
        prompt = row['prompt']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=50
        )

        resp_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        row_dict = row.to_dict()
        row_dict['model_response'] = resp_text
        results.append(row_dict)
    
    return pd.DataFrame(results)


def main(input_csv_path, output_csv_path, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):

    # 1) CSV 읽기
    df = pd.read_csv(input_csv_path)
    print(f"Input DF size: {df.shape}")

    # 2) user_history 생성
    user_history_df = create_user_history(df)  # [userId, title_genres_list, user_query]
    print(f"user_history_df size: {user_history_df.shape}")

    # 2-1) 전체 (title, genres) 튜플 만들기
    #      단, title, genres가 문자열이라고 가정
    temp_df = df[['title','genres']].drop_duplicates()
    # list of (title, genres) as tuple
    all_items = list(temp_df.itertuples(index=False, name=None))  
    # ex) ("Mechanic: Resurrection (2016)", "Action|Crime|Thriller"), ...

    # 3) pairwise_df 생성
    pairwise_df = create_pairwise_df(user_history_df, all_items, n_samples=3)
    print(f"Pairwise DF size: {pairwise_df.shape}")

    # 4) 모델 로딩
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_name=model_name, device=device)

    # 5) generate_text_batch로 inference
    result_df = generate_text_batch(pairwise_df, model, tokenizer, device=device)

    result_df.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")


if __name__ == "__main__":
    """
    예시 실행:
    python generate_pairwise.py --input_csv_path=./ratings.csv --output_csv_path=./results.csv --model_name="TinyLlama/..."
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv_path", type=str, required=True)
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()

    main(
        input_csv_path=args.input_csv_path,
        output_csv_path=args.output_csv_path,
        model_name=args.model_name
    )