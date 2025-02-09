import pandas as pd
import os



class PreprocessingData:
    def __init__(self, config):
        self.dataset_path = config.dataset.data_dir + config.dataset.data
        self.threshold = config.dataloader.threshold
        self.timestamp = config.dataloader.timestamp
        self.release_date = int(self.timestamp[:4])-1
        self.preprocessed_path = os.path.join(self.dataset_path, 'preprocessed')
        self.train_file = os.path.join(self.preprocessed_path, 'train.txt')
        self.test_file = os.path.join(self.preprocessed_path, 'test.txt')
        self.cold_train_file = os.path.join(self.preprocessed_path, 'cold_train.txt')
        self.cold_test_file = os.path.join(self.preprocessed_path, 'cold_test.txt')
        self._group_user_item()
        self.popular_items = self._get_popular_items()
 

    def _get_popular_items(self, top_n=100):
        item_counts = {}
        with open(self.train_file, "r", encoding="utf-8") as f:
            for line in f:
                items = list(map(int, line.strip().split(" ")[1:]))
                for item in items:
                    if item in item_counts:
                        item_counts[item] += 1
                    else:
                        item_counts[item] = 1
        
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        popular_items = [item[0] for item in sorted_items[:top_n]]
        print(f"상위 {top_n}개 인기 아이템: {popular_items}")
        return popular_items

    def _load_ratings_data(self):
        try:
            file_path = os.path.join(self.dataset_path, 'raw')
            rating_df = pd.read_csv(file_path + '/ratings_updated.csv')
            movie_df = pd.read_csv(file_path + '/movies_and_links_updated.csv')
            return rating_df, movie_df
        
        except FileNotFoundError:
            print(f"파일 {file_path}이 없습니다.")
            return 
    

    def _delete_rating_by_nanitem(self, rating_df, movie_df):
        filtered_movie_df = movie_df[movie_df['movieId'].isin(list(rating_df['movieId'].unique()))]
        
        deleted_nan_rating_df = rating_df[rating_df['movieId'].isin(list(movie_df['movieId'].unique()))]
        
        return deleted_nan_rating_df, filtered_movie_df


    def _delete_rating_by_futures(self, rating_df, movie_df, time):
        rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], unit='s')
        rating_df['year'] = rating_df['timestamp'].dt.year

        # 첫 상호작용이 timestamp 이후인 user의 모든 상호작용 제거
        first_interaction = rating_df.groupby('userId')['timestamp'].min()
        
        future_users = first_interaction[first_interaction >= time].index
        
        deleted_future_user_df = rating_df[~rating_df['userId'].isin(future_users)]

        # 개봉년도가 timestamp 이후인 item의 모든 상호작용 제거
        future_movie = movie_df[movie_df['year'] > self.release_date]['movieId'].unique()
        deleted_future_item_df = deleted_future_user_df[~deleted_future_user_df['movieId'].isin(list(future_movie))]

        return deleted_future_item_df
    

    def _filter_rating_by_interaction(self, rating_df, min, max):
        group_user = rating_df.groupby('userId')['movieId'].count().reset_index()
        
        group_user_minmax = group_user[(group_user['movieId'] >= min) & (group_user['movieId'] <= max)]
        
        filtered_df = rating_df[rating_df['userId'].isin(list(group_user_minmax['userId'].unique()))]

        return filtered_df
    

    def _process_data(self):
        rating_df, movie_df = self._load_ratings_data()
        
        deleted_nan_rating_df, filtered_movie_df = self._delete_rating_by_nanitem(rating_df, movie_df)

        positive_df = deleted_nan_rating_df[deleted_nan_rating_df['rating'] >= self.threshold].copy()

        deleted_future_item_df = self._delete_rating_by_futures(positive_df, filtered_movie_df, self.timestamp)

        filtered_df = self._filter_rating_by_interaction(deleted_future_item_df, 10, 100)
        filtered_df['newUserId'] = pd.factorize(filtered_df['userId'])[0]
        filtered_df['newItemId'] = pd.factorize(filtered_df['movieId'])[0]

        return filtered_df, filtered_movie_df


    def _split_data(self):
        filtered_df, filtered_movie_df = self._process_data()
        
        train_df = filtered_df[filtered_df['timestamp'] < self.timestamp]
        test_df = filtered_df[filtered_df['timestamp'] >= self.timestamp]

        return train_df, test_df, filtered_movie_df
    

    def _extract_colditem(self, rating_df, movie_df):        
        new_item_movieId = movie_df[movie_df['year'] == self.release_date]['movieId'].unique()
        new_item_rating_df = rating_df[rating_df['movieId'].isin(list(new_item_movieId))]
        new_item_rating_group = new_item_rating_df.groupby('movieId')['userId'].count().reset_index()
        
        cold_item = list(new_item_rating_group[new_item_rating_group['userId']<=10]['movieId'].unique())
        
        return cold_item
    

    def _split_cold_data(self):
        train_df, test_df, filtered_movie_df = self._split_data()
        
        cold_item = self._extract_colditem(train_df, filtered_movie_df)
        cold_train_df = train_df[train_df['movieId'].isin(cold_item)]
        cold_test_df = test_df[test_df['movieId'].isin(cold_item)]
        
        return train_df, test_df, cold_train_df, cold_test_df

    def _group_user_item(self):
        train_df, test_df, cold_train_df, cold_test_df = self._split_cold_data()
        
        train_group = train_df.groupby('newUserId')['newItemId'].apply(list).reset_index()
        test_group = test_df.groupby('newUserId')['newItemId'].apply(list).reset_index()
        
        cold_train_group = cold_train_df.groupby('newUserId')['newItemId'].apply(list).reset_index()
        cold_test_group = cold_test_df.groupby('newUserId')['newItemId'].apply(list).reset_index()
        
        self._save_data(train_group, test_group, cold_train_group, cold_test_group)


    def _save_data(self, train_group, test_group, cold_train_group, cold_test_group):
        if not os.path.exists(self.preprocessed_path):
            os.makedirs(self.preprocessed_path)
            
        with open(self.train_file, 'w', encoding="utf-8") as f:
            for _, row in train_group.iterrows():
                u = row['newUserId']
                items = row['newItemId']
                f.write(f"{u} {' '.join(map(str, items))}\n")

        with open(self.test_file, 'w', encoding="utf-8") as f:
            for _, row in test_group.iterrows():
                u = row['newUserId']
                items = row['newItemId'] 
                f.write(f"{u} {' '.join(map(str, items))}\n")
                
        with open(self.cold_train_file, 'w', encoding="utf-8") as f:
            for _, row in cold_train_group.iterrows():
                u = row['newUserId']
                items = row['newItemId']  
                f.write(f"{u} {' '.join(map(str, items))}\n")

        with open(self.cold_test_file, 'w', encoding="utf-8") as f:
            for _, row in cold_test_group.iterrows():
                u = row['newUserId']
                items = row['newItemId']  
                f.write(f"{u} {' '.join(map(str, items))}\n")