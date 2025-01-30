import pandas as pd
# from sklearn.model_selection import train_test_split
import os
from src.data.split_methods import train_test_split, leave_one_out, k_fold

SPLIT_METHODS = {
    "train_test_split": train_test_split,
    "leave_one_out": leave_one_out,
    "k_fold": lambda grouped: k_fold(grouped, k=5),  # 기본 k=5 설정
}

class PreprocessingData:
    def __init__(self, config):
        self.dataset_path = config.dataset.data_dir+config.dataset.data
        self.threshold = config.dataloader.threshold
        self.preprocessed_path = os.path.join(self.dataset_path, 'preprocessed')
        self.train_file = os.path.join(self.preprocessed_path, 'train.txt')
        self.test_file = os.path.join(self.preprocessed_path, 'test.txt')

        self.split_strategy = SPLIT_METHODS.get(config.dataset.split_method, train_test_split)
        self.process_data()
 
    def load_ratings_data(self):
        try:
            file_path = os.path.join(self.dataset_path, 'raw', 'ratings.dat')
            rating_df = pd.read_csv(file_path, sep='::', engine='python',
                                    names=['userId','movieId','rating', 'timestamp'], header=None)
            return rating_df
        except FileNotFoundError:
            print(f"파일 {file_path}이 없습니다.")
            return 
    
    def process_data(self):
        rating_df = self.load_ratings_data()
        if rating_df is None:
            return
        
        df = rating_df[rating_df['rating'] >= self.threshold].copy()

        df['newUserId'] = pd.factorize(df['userId'])[0]
        df['newItemId'] = pd.factorize(df['movieId'])[0]

        grouped = df.groupby('newUserId')['newItemId'].apply(list).reset_index()

        train_data, test_data = self.split_strategy(grouped)

        self.save_data(train_data, test_data)

    def save_data(self, train_data, test_data):
        if not os.path.exists(self.preprocessed_path):
            os.makedirs(self.preprocessed_path)

        with open(self.train_file, 'w', encoding="utf-8") as f:
            for (u, items) in train_data:
                f.write(f"{u} {' '.join(map(str, items))}\n")

        with open(self.test_file, 'w', encoding="utf-8") as f:
            for (u, items) in test_data:
                f.write(f"{u} {' '.join(map(str, items))}\n")
