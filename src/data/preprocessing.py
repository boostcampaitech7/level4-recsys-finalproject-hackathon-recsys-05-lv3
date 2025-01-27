import pandas as pd
from sklearn.model_selection import train_test_split
import os



class preprocessing_data:
    def __init__(self, dataset_path, threshold=4.0):
        self.dataset_path = dataset_path
        self.threshold = threshold
        self.preprocessed_path = os.path.join(dataset_path, 'preprocessed')
        self.train_file = os.path.join(self.preprocessed_path, 'train.txt')
        self.test_file = os.path.join(self.preprocessed_path, 'test.txt')
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
        
        df = rating_df[rating_df['rating'] >= self.threshold]

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

        self.save_data(train_data, test_data)

    def save_data(self, train_data, test_data):
        if not os.path.exists(self.preprocessed_path):
            os.makedirs(self.preprocessed_path)

        with open(self.train_file, 'w') as f:
            for (u, items) in train_data:
                line = str(u) + ' ' + ' '.join(map(str, items))
                f.write(line + '\n')

        with open(self.test_file, 'w') as f:
            for (u, items) in test_data:
                line = str(u) + ' ' + ' '.join(map(str, items))
                f.write(line + '\n')
