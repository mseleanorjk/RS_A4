from collections import defaultdict
from sentence_transformers import SentenceTransformer
import pickle
import os
import torch
import random
import numpy as np
from config import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reconstruction_loss(x_pred, x_true):
        return ((x_pred - x_true)**2).mean(axis=-1)

def get_item_embeddings(metadata):
    if os.path.exists(os.path.join("embeddings", "embeddings.txt")):
        with open(os.path.join("embeddings", "embeddings.txt"), "rb") as semid:
            emb_dict = pickle.load(semid)
        return emb_dict
    model = SentenceTransformer('all-MiniLM-L6-v2')
    item_ids = metadata['item_id'].tolist()
    sequences = metadata['sequence'].tolist()
    embeddings = model.encode(sequences, convert_to_tensor=True)
    emb_dict = dict(zip(item_ids, embeddings))
    with open(os.path.join("embeddings", "embeddings.txt"), "wb") as semid:
        pickle.dump(emb_dict, semid)
    return emb_dict

def build_disambiguation(item_semantic_ids):
  """
  Function to compute suffixes for different items with the same semantic ID.
  This suffix will be added at the end of the ID to differentiate them if there are collisions
  """
  # hash the items in ID buckets
  groups = defaultdict(list)
  for asin, codes in item_semantic_ids.items():
    # for each code put in the bucket the item ids that have it
    groups[tuple(codes)].append(asin)

    # Assign suffix per item
  item_suffix = {}
  for codes, asins in groups.items():
    for suffix, asin in enumerate(asins):
      # for each item in a bucket, save a unique suffix
      item_suffix[asin] = suffix  # append the ID within the bucket
  return item_suffix

def tokenise(code, suffix):
  """
  Function to add the disambiguation suffices to a semantic ID and shift
  adjacent codebook codes to match their codebook number. Takes one item
  and applies the respective suffix from the lookup table created earlier.
  """
  tokens = []
  offset = 0
  for i, codebook_idx in enumerate(code):
      tokens.append(int(codebook_idx) + offset) # append the shifted codebook index in the code
      # new offsetting process for dynamic codebook sizing
      offset += CENTROIDS * 2**i
  tokens.append(DIS_TOKEN + suffix)  # append disambiguation token
  return tokens
