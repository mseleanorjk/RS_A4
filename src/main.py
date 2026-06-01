import torch
import os
import numpy as np

from config import *
from data_processor import DataProcessor
from rqgat import RQGAT
from snail import SNAIL
from utils import *

train = DataProcessor("train.csv").df
metadata_processor = DataProcessor("item_meta.csv")
test = DataProcessor("test.csv").df
metadata = metadata_processor.add_sequence()

emb_dict = get_item_embeddings(metadata)
