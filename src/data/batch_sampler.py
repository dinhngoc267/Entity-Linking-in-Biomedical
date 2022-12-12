from collections import defaultdict
from data.dataset import MentionEntityDataset, MentionMentionDataset
import torch
import numpy as np


class MentionEntityBatchSampler(object):
  def __init__(self, model, device,
               tokenizer,  
               context_data,
               pair_indices, 
               entity_description_dict,
               batch_size = 16,
               max_len = 256,
               num_neg_per_pos = 1, margin = 0.5):
    
    """
    Parameters:
      training_mention_tokens: list of training token ids
      pair_indices_labels: list of training indices  
    """
    super().__init__()
    self.margin = margin
    self.model = model
    self.device = device
    self.tokenizer = tokenizer
    self.max_len = max_len

    self.context_data = context_data
    self.pair_indices = pair_indices
    self.entity_description_dict = entity_description_dict
    #self.neg_indices_dict = neg_pair_indices_dict

    negative_pair_indices_dict = defaultdict(list)
    for idx, pair in enumerate(self.pair_indices):
      label = pair[0]
      if label == '0':
        negative_pair_indices_dict[pair[1]].append(idx)
    self.negative_pair_indices_dict = negative_pair_indices_dict

    self.batch_size = batch_size
    self.num_neg_per_pos = num_neg_per_pos

    self.len_pos_pairs = (self.pair_indices[:,0] == '1').sum()
    
    # each positive pair correspond to top_k_neg pairs
    self.num_pos_per_batch = self.batch_size//num_neg_per_pos
    #self.num_neg_per_pair = self.batch_size 

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
          anchor_idx = pos_pair[1]

          data_index = int(pos_pair[1])
          mention_cui = pos_pair[2]
          entity_description = self.entity_description_dict[mention_cui]
          entity_description_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity_description))
          input_tokens= MentionEntityDataset.generate_input(data_index, 
                                                            self.context_data,
                                                            entity_description_tokens,
                                                            self.max_len)
          input_tokens = input_tokens[None,:].to(self.device)
        
          pos_affin = self.model(input_tokens)[0]
          # get all candidate from pair_indices first
          #neg_candidates = np.array(self.negative_pair_indices_dict[anchor_idx])
          if len(self.negative_pair_indices_dict[anchor_idx]) >= 8:
            neg_candidates = np.random.choice(np.array(self.negative_pair_indices_dict[anchor_idx]), 8, replace=False)
          else:
            neg_candidates = np.array(self.negative_pair_indices_dict[anchor_idx])

          # narrow down to top-k
          neg_candidates_pairs = self.pair_indices[neg_candidates]
          
          input_tokens_buffer = []

          for neg_pair in neg_candidates_pairs:
            #label = neg_pair[0]
            data_index = int(neg_pair[1])
            mention_cui = neg_pair[2]
            entity_description = self.entity_description_dict[mention_cui]
            entity_description_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity_description))
            input_tokens= MentionEntityDataset.generate_input(data_index, 
                                                              self.context_data,
                                                              entity_description_tokens,
                                                              self.max_len)
            input_tokens_buffer.append(input_tokens)
          
          input_tokens_buffer = torch.stack(input_tokens_buffer).to(self.device)
          neg_affin = self.model(input_tokens_buffer)

          top_k_neg = torch.topk(neg_affin,  1, dim=0, largest=True, sorted=True)[1].cpu()
          top_k_neg = neg_candidates[top_k_neg].flatten().tolist()
          neg_batch_indices.extend(top_k_neg)

          # flag = False
          # max_affin = 0
          # max_idx = 0
          # for idx, affin in enumerate(neg_affin): 
          #   if affin > (pos_affin - self.margin) and affin < pos_affin and max_affin < affin:
          #     max_idx = idx
          #     max_affin = affin
          #     flag = True

          # if flag == True:
          #   top_k_neg = neg_candidates[max_idx].flatten().tolist()
          #   neg_batch_indices.extend(top_k_neg)
          # else: #  flag == False:
          #   if neg_affin[0] >= pos_affin:
          #     top_k_neg = torch.topk(neg_affin, 1, dim=0, largest=False, sorted=True)[1].cpu()
          #     top_k_neg = neg_candidates[top_k_neg].flatten().tolist()
          #     neg_batch_indices.extend(top_k_neg)
          #   else:
          #     top_k_neg = torch.topk(neg_affin, 1, dim=0, largest=True, sorted=True)[1].cpu()
          #     top_k_neg = neg_candidates[top_k_neg].flatten().tolist()
          #     neg_batch_indices.extend(top_k_neg)
          #top_k_neg = neg_candidates[1].flatten().tolist()

      batch_indices.extend(pos_batch_indices)
      batch_indices.extend(neg_batch_indices)


      yield batch_indices 
  
  def __len__(self):
    return self.num_iterations



class MentionMentionBatchSampler(object):
  def __init__(self, model, device, 
               context_data, 
               pair_indices, 
               batch_size = 16, 
               max_len = 256, 
               num_neg_per_pos = 1, margin = 0.5):
    
    """
    Parameters:
      training_mention_tokens: list of training token ids
      pair_indices: list of training indices  
    """
    super().__init__()

    self.model = model
    self.device = device
    self.max_len = max_len
    self.margin = margin

    self.context_data = context_data
    self.pair_indices = pair_indices

    self.batch_size = batch_size
    self.num_neg_per_pos = num_neg_per_pos

    self.len_positive_pairs = (self.pair_indices[:,0] == 1).sum()
    
    # each positive pair correspond to top_k_neg pairs
    self.num_pos_per_batch = self.batch_size//1
    self.num_neg_per_pairs = self.batch_size 

    # num of iterations:
    self.num_iterations = self.len_positive_pairs//self.num_pos_per_batch

    negative_pair_indices_dict = defaultdict(list)
    for idx, pair in enumerate(self.pair_indices):
      label = pair[0]
      if label == 0:
        negative_pair_indices_dict[pair[1]].append(idx)
    self.negative_pair_indices_dict = negative_pair_indices_dict

    self.top_k_neg = 1

  
  def __iter__(self):
    # get list of positive pair indices and negative pair indices from pair indices labels
    all_pos_pair_indices = np.where(self.pair_indices[:,0] == 1)[0]
    start_pos_index = 0

    for i in range(self.num_iterations):
      batch_indices = [] 
      pos_batch_indices = all_pos_pair_indices[start_pos_index:start_pos_index + self.num_pos_per_batch ]
      start_pos_index += self.num_pos_per_batch
      
      with torch.no_grad():
        neg_batch_indices = []
        for i, pos_pair in enumerate(self.pair_indices[pos_batch_indices]):
          label = pos_pair[0]
          context_data_a = self.context_data[int(pos_pair[1])]
          context_data_b = self.context_data[int(pos_pair[2])]
          input_tokens, mention_mask_a, mention_mask_b = MentionMentionDataset.generate_input(context_data_a, 
                                                                                                context_data_b,
                                                                                               self.max_len)
          input_tokens = input_tokens[None,:].to(self.device)
        
          pos_affin = self.model(input_tokens, [mention_mask_a], [mention_mask_b])[0]
          anchor_idx = int(pos_pair[1])
          # get all candidate from pair_indices first
          if len(self.negative_pair_indices_dict[anchor_idx]) >= 16:
            neg_candidates = np.random.choice(np.array(self.negative_pair_indices_dict[anchor_idx]), 16, replace=False)
          else:
            neg_candidates = np.array(self.negative_pair_indices_dict[anchor_idx])

          # narrow down to top-k
          neg_candidates_pairs = self.pair_indices[neg_candidates]
          
          input_tokens_buffer = []
          mention_mask_a_buffer = []
          mention_mask_b_buffer = []

          for neg_pair in neg_candidates_pairs:
            label = neg_pair[0]
            context_data_a = self.context_data[int(neg_pair[1])]
            context_data_b = self.context_data[int(neg_pair[2])]

            input_tokens, mention_mask_a, mention_mask_b = MentionMentionDataset.generate_input(context_data_a, 
                                                                                                context_data_b,
                                                                                                self.max_len)
          
            input_tokens_buffer.append(input_tokens)
            mention_mask_a_buffer.append(mention_mask_a)
            mention_mask_b_buffer.append(mention_mask_b)

          input_tokens_buffer = torch.stack(input_tokens_buffer).to(self.device)
          neg_affin = self.model(input_tokens_buffer, mention_mask_a_buffer, mention_mask_b_buffer)

          flag = False
          max_affin = 0
          max_idx = 0
          for idx, affin in enumerate(neg_affin): 
            if affin > (pos_affin - self.margin) and affin < pos_affin and max_affin < affin:
              max_idx = idx
              max_affin = affin
              #top_k_neg = neg_candidates[idx].flatten().tolist()
              #neg_batch_indices.extend(top_k_neg)
              flag = True
              #break
          if flag == True:
            top_k_neg = neg_candidates[max_idx].flatten().tolist()
            neg_batch_indices.extend(top_k_neg)
          else: #  flag == False:
            if neg_affin[0] >= pos_affin:
              top_k_neg = torch.topk(neg_affin, 1, dim=0, largest=False, sorted=True)[1].cpu()
              top_k_neg = neg_candidates[top_k_neg].flatten().tolist()
              neg_batch_indices.extend(top_k_neg)
            else:
              top_k_neg = torch.topk(neg_affin, 1, dim=0, largest=True, sorted=True)[1].cpu()
              top_k_neg = neg_candidates[top_k_neg].flatten().tolist()
              neg_batch_indices.extend(top_k_neg)

      batch_indices.extend(pos_batch_indices)
      batch_indices.extend(neg_batch_indices)
      yield batch_indices 
    
  def __len__(self):
    return self.num_iterations

