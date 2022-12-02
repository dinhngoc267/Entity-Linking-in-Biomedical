import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from transformers import BertModel
from transformers import BertTokenizer


class MentionMentionAffinityModel(nn.Module):
  def __init__(self, tokenizer, bert_base,  max_len):
    super().__init__()

    self.bert_base = bert_base  
    self.linear = nn.Linear(in_features = 768*2, out_features = 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x, mention_mask_a, mention_mask_b):
    """
    Feed input to BERT and compute affinity score between two mentions
    Parameters:
      - inputs: list of tensors: [input_ids, token_type_ids, attention_mask]. Each one is tensor has shape (batch_size, max_len)
      - pos_mention: positiion of two mentions token in the input [start_1, end_1, start_2, end_2]
      - mention_mask: torch.Tensor - a 1D tensor of 0 and 1. 1 indicates for mention
    Return:
      - torch.Tensor: score of affinity
    """
    input_ids, token_type_ids, attention_mask = x[:,0,:], x[:,1,:], x[:,2,:]
    # feed forward

    outputs = self.bert_base(input_ids = input_ids, token_type_ids = token_type_ids,attention_mask = attention_mask)
    outputs = outputs[0]
    # shape of outputs: [batch_size, max_len, num_of_hidden]

    mention_a = outputs*mention_mask_a[:,:,None]
    num_token = torch.sum(mention_mask_a, dim=1, keepdims=False)
    mention_a = torch.sum(mention_a,dim = 1)/num_token[:,None]

    mention_b = outputs*mention_mask_b[:,:,None]
    num_token = torch.sum(mention_mask_b, dim=1, keepdims=False)
    mention_b = torch.sum(mention_b,dim = 1)/num_token[:,None]

    concat_mentions = torch.cat((mention_a, mention_b), dim=1)

    affinity_score = self.linear(concat_mentions)
    affinity_score = self.sigmoid(affinity_score)
    
    return affinity_score


class MentionEntityAffinityModel(nn.Module):
  def __init__(self, base_model, num_hidden_layer=768):
    super().__init__()

    self.base_model = base_model  
    #else:
    #  self.base_model = pickle.load(base_model_path)
    self.linear = nn.Linear(in_features = num_hidden_layer, out_features = 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    """
    Feed input to BERT and compute affinity score between two mentions
    Parameters:
      - inputs: list of tensors: [input_ids, token_type_ids, attention_mask]. Each one is tensor has shape (batch_size, max_len)
      - pos_mention: positiion of two mentions token in the input [start_1, end_1, start_2, end_2]
      - mention_mask: torch.Tensor - a 1D tensor of 0 and 1. 1 indicates for mention
    Return:
      - torch.Tensor: score of affinity
    """
    input_ids, token_type_ids, attention_mask = x[:,0,:], x[:,1,:], x[:,2,:]
    # feed forward

    outputs = self.base_model(input_ids = input_ids, token_type_ids = token_type_ids,attention_mask = attention_mask)
    outputs = outputs[0]
    # shape of outputs: [batch_size, max_len, num_of_hidden]

    cls_tokens = outputs[:,0,:]
    linear = self.linear(cls_tokens)
    affinity_score = self.sigmoid(linear) 

    return affinity_score

