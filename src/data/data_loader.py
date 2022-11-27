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

def tokenize_sentence(sentence, tokenizer):
    sentence_tokens = []
    start = None
    end = None
    label = None
    flag = False

    for item in sentence:
        splits = item.split('\t')
        word = splits[0]
        word_label = splits[1]

        if 'B' in word_label:
            sentence_tokens += ['[START]']
            start = len(sentence_tokens)
            label = word_label.split(':')[1]
            flag = True
        elif 'O' in word_label and flag == True:
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


class MentionEntityAffinityDataset(Dataset):
  def __init__(self, data_dir, dictionary_file, candidate_file, tokenizer, max_len=256):

    super().__init__()

    self.context_data = []
    self.mentions = []
    self.labels = []
    self.dictionary = {}
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

        mention_file = '.' +file_name.split('.')[1] + '.txt'
        with open(mention_file, "r") as f:
          lines = f.read().split('\n')
          for line in lines: 
            line = line.split('||')
            self.mentions.append(line[1].lower())
            self.labels.append(line[0])
          
 
    # load dictionary
    with open(dictionary_file, "r") as f:
      lines = f.read().split('\n')
      
      for line in lines:
        line = line.split('||')
        id = line[0]
        description = line[1]
        self.dictionary[id] = description 

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

  
  def get_pair(self):
    return self.pairs


  def __len__(self):
    return len(self.pair_indices)
  
  def __getitem__(self, idx):
    pair = self.pair_indices[idx]
    label = int(pair[0])

    mention_idx = int(pair[1])
    mention_cui = pair[2]
    entity_description_tokens = self.all_entity_description_tokens_dict[mention_cui]

    item = MentionEntityAffinityDataset.generate_mention_entity_affinity_model_input(mention_idx,
                                                                        self.training_mention_tokens, 
                                                                        self.training_mention_pos,
                                                                        entity_description_tokens,
                                                                        self.max_len)



    return item, label

  @staticmethod
  def generate_input(mention_index, 
                     training_mention_tokens, 
                     training_mention_pos, 
                     entity_description_tokens, 
                     max_len):
    """
    
    """
    mention_tokens = training_mention_tokens[mention_index]
    [start_mention_token, end_mention_token] = training_mention_pos[mention_index]
    

    if len(mention_tokens) > max_len//2 -2:
      # generate a random position to truncate the sentence:
      rand_num = np.random.randint(start_mention_token - 60, start_mention_token)
      if start_mention_token == 0 or rand_num < 0:
        rand_num = 0
      mention_tokens = mention_tokens[rand_num:rand_num+max_len//2-2]
      start_mention_token -= rand_num
      end_mention_token -= rand_num
    
    start_mention_token += 1 # add cls token 
    end_mention_token += 1

    if len(mention_tokens) + len(entity_description_tokens) > max_len-1:
      new_length = max_len -1 - len(mention_tokens)
      entity_description_tokens = entity_description_tokens[:new_length]

    input_ids = [101] + mention_tokens + [102] + entity_description_tokens + [102]
    
    input_len = len(input_ids)
    input_ids = input_ids[:max_len] + [0]*(max_len-input_len)
    token_type_ids = [0]*(2+len(mention_tokens)) + [1]*(max_len - len(mention_tokens) -2)
    attention_mask = [1]*input_len + [0]*(max_len-input_len)
    attention_mask = attention_mask[:max_len]
    
    return torch.tensor([input_ids, token_type_ids, attention_mask])

def main(args):
  data_dir = args.data_dir
  dictionary_file = args.dictionary_file
  candidate_file = args.candidate_file

  tokenizer = BertTokenizer.from_pretrained("nlpie/bio-distilbert-uncased",use_fast=True)

  MentionEntityAffinityDataset(data_dir=data_dir, dictionary_file=dictionary_file, candidate_file=candidate_file, tokenizer=tokenizer)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default="./data/processed/st21pv/train",
                      help = 'directory of data')
  parser.add_argument('--dictionary_file', type=str,
                  default="./data/processed/umls/dictionary.txt",
                  help='path of input file (train/test)')                
  parser.add_argument('--candidate_file', type=str,
                  default="./models/candidates/st21pv/train/candidates.txt", 
                  help='path of candidates of data')

  args = parser.parse_args()

  main(args)
          
