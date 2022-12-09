import torch
import numpy as np
import glob
import os
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import argparse

def tokenize_sentence(sentence, tokenizer):
    sentence_tokens = []
    start = None
    end = None
    flag = False

    for item in sentence:
        splits = item.split('\t')
        word = splits[0]
        word_label = splits[1]

        if 'b' in word_label:
            sentence_tokens += ['[START]']
            start = len(sentence_tokens)
            flag = True
        elif 'o' in word_label and flag == True:
            end = len(sentence_tokens)
            sentence_tokens += ['[END]']
            flag = False   

        tokens = tokenizer.tokenize(word)
        for token in tokens:
            sentence_tokens += [token]
    if end is None:
        end = len(sentence_tokens)
        sentence_tokens += ['[END]']
    sentence_tokens = tokenizer.convert_tokens_to_ids(sentence_tokens)

    return sentence_tokens, [start, end]


class MentionEntityDataset(Dataset):
  def __init__(self, data_dir, 
               dictionary_file,
               candidate_file, 
               tokenizer, max_len=256, n_test= 50):

    super().__init__()

    self.tokenizer = tokenizer
    self.context_data = []
    self.mentions = []
    self.labels = []
    self.entity_description_dict = {}
    self.candidates_dict = {}
    self.pair_indices = []

    # load context data
    files = glob.glob(os.path.join(data_dir, "*.context"))

    for file_name in tqdm(files):
        with open(file_name, "r") as f:
            list_sents = f.read().split('\n\n')
            list_sents = list_sents[:-1]
        for sent in list_sents:
            sent_token_ids, mention_index = tokenize_sentence(sent.lower().split('\n'), tokenizer)
            self.context_data.append((sent_token_ids, mention_index))

        mention_file = os.path.join(data_dir, os.path.basename(file_name).replace(".context", ".txt"))
        with open(mention_file, "r") as f:
          lines = f.read().split('\n')
          for line in lines: 
            line = line.split('||')
            self.mentions.append(line[1].lower())
            self.labels.append(line[0])

        n_test -= 1
        if n_test == 0:
          break
          
 
    # load dictionary
    with open(dictionary_file, "r") as f:
      lines = f.read().split('\n')
      
      for line in lines:
        line = line.split('||')
        cui = line[0]
        description = line[1] + '|' + line[2]
        self.entity_description_dict[cui] = description.lower() 

    # load candidates
    with open(candidate_file, "r") as f:
      lines = f.read().split('\n')
      for line in lines:
        line = line.split('||')
        self.candidates_dict[line[0]] = line[1].split(' ')


    # create list pair of index for training 
    anchor_negative_pairs = []
    for i in range(len(self.mentions)):
      hardest_negative_cuis = [cui for cui in self.candidates_dict[self.mentions[i].lower()] if cui != self.labels[i]] 
      for cui in hardest_negative_cuis:
        anchor_negative_pairs.append((0, i, cui))

    anchor_positive_pairs = []
    for i in range(len(self.mentions)):
      anchor_positive_pairs.append((1, i, self.labels[i]))
    
    self.pair_indices = anchor_negative_pairs + anchor_positive_pairs
    random.shuffle(self.pair_indices)
    self.pair_indices = np.array(self.pair_indices)
    
    self.max_len = max_len

    print('There are {} positive pairs and {} negative pairs\n'.format(len(anchor_positive_pairs), len(anchor_negative_pairs)))

  def __len__(self):
    return len(self.pair_indices)
  
  def __getitem__(self, idx):
    pair = self.pair_indices[idx]
    label = int(pair[0])

    data_index = int(pair[1])
    cui = pair[2]
    entity_description = self.entity_description_dict[cui]
    entity_description_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity_description))

    item = MentionEntityDataset.generate_input(data_index,
                                               self.context_data,
                                               entity_description_tokens,
                                               self.max_len)

    return item, label

  @staticmethod
  def generate_input(data_index: int, 
                     context_data: list, 
                     entity_description_tokens: list, 
                     max_len=256):
    """
    
    """
    sentence_tokens,  [start_mention_token, end_mention_token]  = context_data[data_index]   

    if len(sentence_tokens) > max_len//2 -2:
      # generate a random position to truncate the sentence:
      rand_num = np.random.randint(start_mention_token - 60, start_mention_token)
      if start_mention_token == 0 or rand_num < 0:
        rand_num = 0
      sentence_tokens = sentence_tokens[rand_num:rand_num+max_len//2-2]
      start_mention_token -= rand_num
      end_mention_token -= rand_num
    
    start_mention_token += 1 # add cls token 
    end_mention_token += 1

    if len(sentence_tokens) + len(entity_description_tokens) > max_len-1:
      new_length = max_len -1 - len(sentence_tokens)
      entity_description_tokens = entity_description_tokens[:new_length]

    input_ids = [101] + sentence_tokens + [102] + entity_description_tokens + [102]
    
    input_len = len(input_ids)
    input_ids = input_ids[:max_len] + [0]*(max_len-input_len)
    token_type_ids = [0]*(2+len(sentence_tokens)) + [1]*(max_len - len(sentence_tokens) -2)
    attention_mask = [1]*input_len + [0]*(max_len-input_len)
    attention_mask = attention_mask[:max_len]
    
    return torch.tensor([input_ids, token_type_ids, attention_mask])

  def get_pairs(self):
    return self.pairs

  def get_context_data(self):
    return self.context_data

  def get_entity_description_dict(self):
    return self.entity_description_dict


class MentionMentionDataset(Dataset):
  def __init__(self, data_dir, tokenizer, max_len=256, n_test = 50):

    super().__init__()

    self.tokenizer = tokenizer
    self.max_len = max_len
    
    self.mentions = []
    self.labels = []
    self.context_data = []
    self.pair_indices = []

    doc_ids = []
    # load context data
    files = glob.glob(os.path.join(data_dir, "*.context"))

    for file_name in tqdm(files):
        with open(file_name, "r") as f:
            list_sents = f.read().split('\n\n')
            list_sents = list_sents[:-1]
        for sent in list_sents:
            sent_token_ids, mention_index = tokenize_sentence(sent.lower().split('\n'), tokenizer)
            self.context_data.append((sent_token_ids, mention_index))

        mention_file = os.path.join(data_dir, os.path.basename(file_name).replace(".context", ".txt"))
        with open(mention_file, "r") as f:
          lines = f.read().split('\n')
          for line in lines: 
            line = line.split('||')
            self.mentions.append(line[1].lower())
            self.labels.append(line[0])
            id = os.path.basename(mention_file).replace(".txt", "")
            doc_ids.append(id)

        n_test -= 1
        if n_test == 0:
          break
          
    # create pair indices for training
    
    # anchor_positive pairs
    anchor_positive_indices = []
    for anchor in tqdm(range(len(self.labels))):
      for positive in range(len(self.labels)):
        if anchor!=positive and self.labels[anchor] == self.labels[positive]:
          anchor_positive_indices.append([1, anchor, positive])
    
    # anchor negative pairs
    anchor_negative_indices = []
    for _, anchor, _ in anchor_positive_indices:
      for negative in (range(len(self.labels))):
        if doc_ids[negative] == doc_ids[anchor] and self.labels[anchor] != self.labels[negative]:
          anchor_negative_indices.append([0, anchor, negative])

    print('There are {} anchor positive pairs, {} anchor negative pairs'.format(len(anchor_positive_indices), len(anchor_negative_indices)))


    self.pair_indices = anchor_positive_indices + anchor_negative_indices
    random.shuffle(self.pair_indices)
    self.pair_indices = np.array(self.pair_indices)

  def __len__(self):
    return len(self.pair_indices)
  
  def __getitem__(self, idx):
    pair = self.pair_indices[idx]
    label = int(pair[0])

    context_a = self.context_data[int(pair[1])]
    context_b = self.context_data[int(pair[2])]

    inputs = MentionMentionDataset.generate_input(context_a=context_a,
                                                  context_b=context_b,
                                                  max_len=self.max_len)
  
    return inputs, label

  @staticmethod
  def generate_input(context_a, context_b, max_len):

    sentence_tokens_a, [start_mention_token_a, end_mention_token_a] = context_a
    sentence_tokens_b, [start_mention_token_b, end_mention_token_b] = context_b

    if len(sentence_tokens_a) > max_len//2 - 2:
      # generate a random position to truncate the sentence:
      rand_num = np.random.randint(start_mention_token_a-60, start_mention_token_a)
      if start_mention_token_a == 0 or rand_num <0:
        rand_num = 0
      sentence_tokens_a = sentence_tokens_a[rand_num:rand_num+max_len//2-2]
      start_mention_token_a -= rand_num
      end_mention_token_a -= rand_num
    
    start_mention_token_a += 1
    end_mention_token_a += 1

    if len(sentence_tokens_a) + len(sentence_tokens_b) > max_len-1:
      # generate a random position to truncate the sentence:
      new_length = 255 - len(sentence_tokens_a)
      rand_num = np.random.randint(start_mention_token_b-60, start_mention_token_b)
      if start_mention_token_b == 0 or rand_num <0:
        rand_num = 0
      sentence_tokens_b = sentence_tokens_b[rand_num:rand_num+new_length]

      start_mention_token_b -= rand_num
      end_mention_token_b -= rand_num

    start_mention_token_b += 2 + len(sentence_tokens_a)
    end_mention_token_b += 2 + len(sentence_tokens_a)

    input_ids = [101] + sentence_tokens_a + [102] + sentence_tokens_b + [102]
    input_len = len(input_ids)
    
    input_ids = input_ids[:256] + [0]*(256-input_len)
    token_type_ids = [0]*(2+len(sentence_tokens_a)) + [1]*(256 - len(sentence_tokens_a) -2)
    attention_mask = [1] * input_len + [0]*(256-input_len)
    attention_mask = attention_mask[:256]

    mention_mask_a = [0]*256
    mention_mask_b = [0]*256
    mention_mask_a[start_mention_token_a:end_mention_token_a] = [1]*(end_mention_token_a-start_mention_token_a)
    mention_mask_b[start_mention_token_b:end_mention_token_b] = [1]*(end_mention_token_b-start_mention_token_b)
    
    
    return torch.tensor([input_ids, token_type_ids, attention_mask]), torch.tensor(mention_mask_a, dtype=torch.float32), torch.tensor(mention_mask_b, dtype=torch.float32)

