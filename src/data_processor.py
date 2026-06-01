import pandas as pd
import os

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
    
    def add_embeddings(self, embeddings):
        # Join the data and metadata on the parent_asin column
        joined_df = pd.merge(self.df, embeddings, on="item_id", how="left")
        return joined_df

    def add_sequence(self):
      self.df["sequence"] = (
          self.df["main_category"].astype(str) +
          self.df["title"].astype(str) +
          self.df["features"].astype(str) +
          self.df["price"].astype(str)+
          self.df["store"].astype(str) +
          self.df["description"].astype(str) +
          self.df["categories"].astype(str)
      )
      return self.df

