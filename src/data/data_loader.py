import torch
import numpy as np
import glob
import os
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import BertTokenizer
from transformers import BertModel

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
               tokenizer, max_len=256, n_test= 100):

    super().__init__()

    self.context_data = []
    self.mentions = []
    self.labels = []
    self.entity_description_dict = {}
    self.candidates_dict = {}
    self.pairs = []

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
        description = line[1]
        self.entity_description_dict[cui] = description 

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
    
    self.pairs = anchor_negative_pairs + anchor_positive_pairs
    random.shuffle(self.pairs)
    
    self.max_len = max_len

  def __len__(self):
    return len(self.pair_indices)
  
  def __getitem__(self, idx):
    pair = self.pair_indices[idx]
    label = int(pair[0])

    data_index = int(pair[1])
    cui = pair[2]
    entity_description_tokens = self.entity_description_dict[cui]

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
    #[start_mention_token, end_mention_token] = training_mention_pos[mention_index]
    

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

class MentionEntityBatchSampler(object):
  def __init__(self, model, device, 
               context_data,
               pair_indices, 
               neg_pair_indices_dict,
               entity_description_tokens_dict,
               batch_size = 16,
               max_len = 256,
               top_k_neg = 1):
    
    """
    Parameters:
      training_mention_tokens: list of training token ids
      pair_indices_labels: list of training indices  
    """
    super().__init__()

    self.model = model
    self.device = device
    self.max_len = max_len

    self.context_data = context_data
    self.pair_indices = pair_indices
    self.entity_description_tokens_dict = entity_description_tokens_dict
    self.neg_indices_dict = neg_pair_indices_dict


    self.batch_size = batch_size
    self.top_k_neg = top_k_neg

    self.len_pos_pairs = (self.pair_indices[:,0] == '1').sum()
    
    # each positive pair correspond to top_k_neg pairs
    self.num_pos_per_batch = self.batch_size//1
    self.num_neg_per_pairs = self.batch_size 

    # num of iterations:
    self.num_iterations = self.len_pos_pairs//self.num_pos_per_batch

  
  def __iter__(self):
    # get list of positive pair indices and negative pair indices from pair indices labels
    all_pos_pair_indices = np.where(self.pair_indices[:,0] == '1')[0]
    start_pos_index = 0

    for i in range(self.num_iterations):
      batch_indices = [] 
      pos_batch_indices = all_pos_pair_indices[start_pos_index:start_pos_index + self.num_pos_per_batch ]
      start_pos_index += self.num_pos_per_batch
      
      with torch.no_grad():
        neg_batch_indices = []
        for i, pos_pair in enumerate(self.pair_indices[pos_batch_indices]):
          anchor_idx = int(pos_pair[1])
          # get all candidate from pair_indices first
          neg_candidates = np.array(self.neg_indices_dict[anchor_idx])

          # narrow down to top-k
          neg_candidates_pairs = self.pair_indices[neg_candidates]
          
          input_tokens_buffer = []
          for neg_pair in neg_candidates_pairs:
            #label = neg_pair[0]
            data_index = int(neg_pair[1])
            mention_cui = neg_pair[2]
            entity_description_tokens = self.entity_description_tokens_dict[mention_cui]

            input_tokens= MentionEntityDataset.generate_input(data_index, 
                                                              self.context_data,
                                                              entity_description_tokens,
                                                              self.max_len)

            input_tokens_buffer.append(input_tokens)

          input_tokens_buffer = torch.stack(input_tokens_buffer).to(self.device)
          neg_affin = self.model(input_tokens_buffer)

          top_k_neg = torch.topk(neg_affin,  1, dim=0, largest=False, sorted=False)[1].cpu()
          top_k_neg = neg_candidates[top_k_neg].flatten().tolist()
          neg_batch_indices.extend(top_k_neg)


      batch_indices.extend(pos_batch_indices)
      batch_indices.extend(neg_batch_indices)

      yield batch_indices 
  
  def __len__(self):
    return self.num_iterations


