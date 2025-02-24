import os
import abc

import pandas as pd


class BasePreprocessingData(metaclass=abc.ABCMeta):
    '''
    MovieLens32M 과 MovieLens1M 에서 공통적으로 적용되는 전처리
    '''
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.dataset.data_dir + config.dataset.data
        self.threshold = config.dataloader.threshold
        self.timestamp = config.dataloader.timestamp
        
        self.release_date = self._calculate_release_date()

        self.preprocessed_path = os.path.join(self.dataset_path, 'preprocessed')
        self.train_file = os.path.join(self.preprocessed_path, 'train.txt')
        self.test_file = os.path.join(self.preprocessed_path, 'test.txt')
        self.cold_train_file = os.path.join(self.preprocessed_path, 'cold_train.txt')
        self.cold_test_file = os.path.join(self.preprocessed_path, 'cold_test.txt')

        self._group_user_item()
        self.popular_items = self._get_popular_items()

    @abc.abstractmethod
    def _calculate_release_date(self):

        pass

    @abc.abstractmethod
    def _load_ratings_data(self):
        pass

    @abc.abstractmethod
    def _extract_colditem(self, train_df, movie_df):
        """
        cold item 추출 로직은은 자식 클래스가 구현
        - MovieLens32M: (연(year) == release_date) & user 수 <= 10
        - MovieLens1M: (월(month) == release_date) (user count 필터 없음)
        """
        pass

    def _delete_rating_by_nanitem(self, rating_df, movie_df):
        valid_movie_df = movie_df[movie_df['movieId'].isin(rating_df['movieId'].unique())]
        valid_rating_df = rating_df[rating_df['movieId'].isin(valid_movie_df['movieId'].unique())]
        return valid_rating_df, valid_movie_df


    def _delete_rating_by_futures(self, rating_df, movie_df, time):
        rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], unit='s', errors='coerce')
        first_interaction = rating_df.groupby('userId')['timestamp'].min()
        future_users = first_interaction[first_interaction >= time].index
        rating_df = rating_df[~rating_df['userId'].isin(future_users)]

        if self.config.dataset.data == 'MovieLens32M':  # MovieLens32M의 경우 year 기준으로
            rating_df['year'] = rating_df['timestamp'].dt.year
            future_movie = movie_df[movie_df['year'] > self.release_date]['movieId'].unique()
        else:  # MovieLens1M의 경우 month 기준으로
            movie_df['timestamp'] = pd.to_datetime(movie_df['timestamp'], errors='coerce')
            movie_df['month'] = movie_df['timestamp'].dt.month
            rating_df['month'] = rating_df['timestamp'].dt.month
            future_movie = movie_df[movie_df['month'] > self.release_date]['movieId'].unique()

        rating_df = rating_df[~rating_df['movieId'].isin(future_movie)]
        return rating_df


    def _process_data(self):
        rating_df, movie_df = self._load_ratings_data()
        rating_df, movie_df = self._delete_rating_by_nanitem(rating_df, movie_df)
        rating_df = rating_df[rating_df['rating'] >= self.threshold]
        rating_df = self._delete_rating_by_futures(rating_df, movie_df, self.timestamp)

        rating_df['newUserId'] = pd.factorize(rating_df['userId'])[0]
        rating_df['newItemId'] = pd.factorize(rating_df['movieId'])[0]
        return rating_df, movie_df


    def _split_data(self):
        rating_df, movie_df = self._process_data()
        rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], unit='s', errors='coerce')
        train_df = rating_df[rating_df['timestamp'] < self.timestamp]
        test_df = rating_df[rating_df['timestamp'] >= self.timestamp]
        return train_df, test_df, movie_df


    def _split_cold_data(self):
        train_df, test_df, movie_df = self._split_data()
        cold_item_ids = self._extract_colditem(train_df, movie_df)
        cold_train_df = train_df[train_df['movieId'].isin(cold_item_ids)]
        cold_test_df = test_df[test_df['movieId'].isin(cold_item_ids)]
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

        with open(self.train_file, 'w', encoding='utf-8') as f:
            for _, row in train_group.iterrows():
                u = row['newUserId']
                items = row['newItemId']
                f.write(f"{u} {' '.join(map(str, items))}\n")

        with open(self.test_file, 'w', encoding='utf-8') as f:
            for _, row in test_group.iterrows():
                u = row['newUserId']
                items = row['newItemId']
                f.write(f"{u} {' '.join(map(str, items))}\n")

        with open(self.cold_train_file, 'w', encoding='utf-8') as f:
            for _, row in cold_train_group.iterrows():
                u = row['newUserId']
                items = row['newItemId']
                f.write(f"{u} {' '.join(map(str, items))}\n")

        with open(self.cold_test_file, 'w', encoding='utf-8') as f:
            for _, row in cold_test_group.iterrows():
                u = row['newUserId']
                items = row['newItemId']
                f.write(f"{u} {' '.join(map(str, items))}\n")


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
        return popular_items


class ML32M_PreprocessingData(BasePreprocessingData):
    '''
    MovieLens32M 데이터에 해당하는 전처리
    '''
    def _calculate_release_date(self):
        return int(self.timestamp[:4]) - 1


    def _load_ratings_data(self):
        file_path = os.path.join(self.dataset_path, 'raw')
        rating_df = pd.read_csv(file_path + '/ratings_updated.csv')
        movie_df = pd.read_csv(file_path + '/movies_and_links_updated.csv')
        return rating_df, movie_df


    def _extract_colditem(self, train_df, movie_df):
        new_item_ids = movie_df[movie_df['year'] == self.release_date]['movieId'].unique()
        new_item_rating_df = train_df[train_df['movieId'].isin(new_item_ids)]
        
        group_cnt = new_item_rating_df.groupby('movieId')['userId'].count().reset_index()
        cold_item_ids = group_cnt[group_cnt['userId'] <= 10]['movieId'].unique()
        return cold_item_ids
    

class ML1M_PreprocessingData(BasePreprocessingData):
    '''
    MovieLens1M 데이터에 해당하는 전처리
    '''
    def _calculate_release_date(self):
        return int(self.timestamp[5:7]) - 2


    def _load_ratings_data(self):
        file_path = os.path.join(self.dataset_path, 'raw')
        rating_df = pd.read_csv(
            file_path + '/ratings.dat',
            sep='::',
            engine='python',
            names=['userId','movieId','rating','timestamp'],
            header=None
        )
        movie_df = pd.read_csv(file_path + '/movies_and_links_updated.csv')
        detail = pd.read_csv(file_path + '/details_updated.csv')[['tmdbId','release_date']].dropna(axis=0)
        merge_df = pd.merge(movie_df, detail, on='tmdbId', how='right')
        merge_df.rename(columns={'release_date':'timestamp'}, inplace=True)
        return rating_df, merge_df


    def _extract_colditem(self, train_df, movie_df):
        movie_df['timestamp'] = pd.to_datetime(movie_df['timestamp'], errors='coerce')
        movie_df['month'] = movie_df['timestamp'].dt.month

        new_item_ids = movie_df[movie_df['month'] == self.release_date]['movieId'].unique()
        return new_item_ids
    
