import pandas as pd
from sklearn.model_selection import train_test_split
import os

def data2txt():
    # 파일 경로 설정
    file_path = 'data/MovieLens1M/final/train.txt'

    # 파일 존재 여부 확인
    if os.path.exists(file_path):
        print("train.txt 파일이 존재합니다.")
    else:
        print("train.txt 파일이 존재하지 않습니다.")
        
        try : 
            rating_df = pd.read_csv('data/MovieLens1M/raw/ratings.dat', sep='::', engine='python',
                                    names=['userId','movieId','rating', 'timestamp'], header=None)
        except FileNotFoundError :
            print('파일이 없습니다.')

        # threshold
        threshold = 4.0
        df = rating_df[rating_df['rating']>=threshold]

        df['newUserId'] = pd.factorize(df['userId'])[0]
        df['newItemId'] = pd.factorize(df['movieId'])[0]

        grouped = df.groupby('newUserId')['newItemId'].apply(list).reset_index()

        train_data = []
        test_data = []
        for row in grouped.itertuples():
            user_id = row.newUserId
            item_list = row.newItemId
            if len(item_list) <= 1:
                train_data.append((user_id, item_list))
            else:
                train_part, test_part = train_test_split(
                    item_list,
                    test_size=0.2,
                    random_state=42
                )
                train_data.append((user_id, train_part))
                test_data.append((user_id, test_part))

        final_path = 'data/MovieLens1M/final'

        if not os.path.exists(final_path):
            os.makedirs(final_path)
        
        with open(final_path+'/train.txt','w') as f:
            for (u, items) in train_data:
                line = str(u) + ' ' + ' '.join(map(str, items))
                f.write(line + '\n')
        with open(final_path+'/test.txt', 'w') as f:
            for (u, items) in test_data:
                line = str(u) + ' ' + ' '.join(map(str, items))
                f.write(line + '\n')