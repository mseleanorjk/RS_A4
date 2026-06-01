import pandas as pd
import os

from config import *
  
class DataProcessor:
    def __init__(self, name, max_history=MAX_HISTORY):
        self.name = name
        self.max_history = max_history
        self.df = self._load_data()
        self.user_histories = self._build_user_histories()
        self.metadata = self._load_metadata()
    
    def _load_data(self):
        try:
            df = pd.read_csv(f"data/{self.name}")
            return df
        except FileNotFoundError:
            print(f"File not found: {self.name}")
            return pd.DataFrame()

    def _load_metadata(self):
        for fname in os.listdir("data/"):
            if os.path.isfile("data/" + os.sep + fname):
                # Full path
                f = open("data/" + os.sep + fname, 'r')

                if "meta" in f.read():
                    meta = pd.read_csv("data/" + os.sep + fname)
                else:
                    raise FileNotFoundError("Metadata file not found in data directory.")
                f.close()
        return meta
    
    def join_data(self):
        # Join the data and metadata on the parent_asin column
        joined_df = pd.merge(self.df, self.metadata, on="item_id", how="left")
        return joined_df

    def process_data(self):
      self.df["sequence"] = (
          self.df["main_category"].astype(str) +
          self.df["title"].astype(str) +
          self.df["features"].astype(str) +
          self.df["price"].astype(str)+
          self.df["store"].astype(str) +
          self.df["description"].astype(str) +
          self.df["categories"].astype(str)
      )
      self.df = self.df.sort_values(["user_id", "timestamp"])
      return self.df

    def _build_user_histories(self):
        user_histories = {}
        for user_id, group in self.df.groupby("user_id"):
            user_histories[user_id] = group["sequence"].tolist()
        return user_histories
    
    def get_user_history(self, user_id):
        history = self.user_histories.get(user_id, [])
        return history[-self.max_history:]