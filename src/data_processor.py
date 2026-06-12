import pandas as pd

from config import *
  
class DataProcessor:
    def __init__(self, name):
        self.name = name
        self.df = self._load_data()
        # self.metadata = self._load_metadata()
    
    def _load_data(self):
        try:
            df = pd.read_csv(f"data/{self.name}")
            if self.name in ["train.csv", "test.csv"]:
                df = df.sort_values(["user_id", "timestamp"])
            return df
        except FileNotFoundError:
            print(f"File not found: {self.name}")
            return pd.DataFrame()
    
    def add_price_bin(self):
        def bin_within_category(group):
            try:
                group['price_label'] = pd.qcut(
                    group['price'], 
                    q=3, 
                    labels=['low', 'medium', 'high']
                )
            except ValueError:
                # Too few unique values in this category to bin
                group['price_label'] = 'medium'
            return group
        
        self.df = self.df.groupby('main_category', group_keys=False).apply(bin_within_category)
        return self.df

    def add_sequence(self):
        self.df = self.add_price_bin()
        text_fields = ["main_category", "title", "features", 
                   "description", "categories", "price_label"]
        self.df["sequence"] = self.df[text_fields].fillna('').agg(' '.join, axis=1) #type: ignore
        return self.df
    
    def split_data(self):
        # Drop duplicate user+item+timestamp triplets first
        self.df = self.df.drop_duplicates(
            subset=['user_id', 'item_id', 'timestamp']
        # sort values 
        ).sort_values(
            ['user_id', 'timestamp', 'item_id'],
            kind='mergesort'
        ).reset_index(drop=True)
        
        val = self.df.groupby('user_id').tail(1).reset_index(drop=True)
        train = self.df.groupby('user_id').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        
        return train, val
