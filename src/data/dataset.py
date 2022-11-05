import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



class MentionEntityAffinityDataset(Dataset):
  def __init__(self, training_mention_tokens,training_mention_pos, 
               pair_indices,
               all_entity_description_tokens_dict, 
               max_len=256):

    super().__init__()

    self.training_mention_tokens = training_mention_tokens
    self.training_mention_pos = training_mention_pos
    self.pair_indices = pair_indices
    self.all_entity_description_tokens_dict = all_entity_description_tokens_dict
    self.max_len = max_len


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
  def generate_mention_entity_affinity_model_input(mention_index, 
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



class MentionMentionAffinityDataset(Dataset):
  def __init__(self, training_mention_tokens,training_mention_pos, 
              pair_indices, 
              max_len=256):

    super().__init__()

    self.training_mention_tokens = training_mention_tokens
    self.training_mention_pos = training_mention_pos
    self.pair_indices = pair_indices
    self.max_len = max_len

  def __len__(self):
    return len(self.pair_indices)
  
  def __getitem__(self, idx):
    pair = self.pair_indices[idx]
    label = int(pair[0])

    mention_idx = int(pair[1])
    mention_cui = pair[2]
    entity_description_tokens = self.all_entity_description_tokens_dict[mention_cui]

    inputs = MentionMentionAffinityDataset.generate_mention_mention_affinity_model_input(mention_idx,
                                                                        self.training_mention_tokens, 
                                                                        self.training_mention_pos,
                                                                        mention_cui,
                                                                        entity_description_tokens,
                                                                        self.max_len)
  
    return inputs, label

  @staticmethod
  def generate_mention_mention_affinity_model_input(index_a, index_b, 
                                                    training_mention_tokens,
                                                    training_mention_pos,
                                                    max_len):

    mention_tokens_a = training_mention_tokens[index_a]
    mention_tokens_b = training_mention_tokens[index_b]
    [start_mention_token_a, end_mention_token_a] = training_mention_pos[index_a]
    [start_mention_token_b, end_mention_token_b] = training_mention_pos[index_b]
   

    if len(mention_tokens_a) > max_len//2 - 2:
      # generate a random position to truncate the sentence:
      rand_num = np.random.randint(start_mention_token_a-60, start_mention_token_a)
      if start_mention_token_a == 0 or rand_num <0:
        rand_num = 0
      mention_tokens_a = mention_tokens_a[rand_num:rand_num+max_len//2-2]
      start_mention_token_a -= rand_num
      end_mention_token_a -= rand_num
    
    start_mention_token_a += 1
    end_mention_token_a += 1

    if len(mention_tokens_a) + len(mention_tokens_b) > max_len-1:
      # generate a random position to truncate the sentence:
      new_length = 255 - len(mention_tokens_a)
      rand_num = np.random.randint(start_mention_token_b-60, start_mention_token_b)
      if start_mention_token_b == 0 or rand_num <0:
        rand_num = 0
      mention_tokens_b = mention_tokens_b[rand_num:rand_num+new_length]

      start_mention_token_b -= rand_num
      end_mention_token_b -= rand_num

    start_mention_token_b += 2 + len(mention_tokens_a)
    end_mention_token_b += 2 + len(mention_tokens_a)

    input_ids = [101] + mention_tokens_a + [102] + mention_tokens_b + [102]
    input_len = len(input_ids)
    
    input_ids = input_ids[:256] + [0]*(256-input_len)
    token_type_ids = [0]*(2+len(mention_tokens_a)) + [1]*(256 - len(mention_tokens_a) -2)
    attention_mask = [1] * input_len + [0]*(256-input_len)
    attention_mask = attention_mask[:256]

    mention_mask_a = [0]*256
    mention_mask_b = [0]*256
    mention_mask_a[start_mention_token_a:end_mention_token_a] = [1]*(end_mention_token_a-start_mention_token_a)
    mention_mask_b[start_mention_token_b:end_mention_token_b] = [1]*(end_mention_token_b-start_mention_token_b)
    
    
    return torch.tensor([input_ids, token_type_ids, attention_mask]), torch.tensor(mention_mask_a, dtype=torch.float32), torch.tensor(mention_mask_b, dtype=torch.float32)




class MentionEntityBatchSampler(object):
  def __init__(self, model, device, training_mention_tokens, 
               training_mention_pos,
               pair_indices_labels, 
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
    self.neg_indices_dict = neg_pair_indices_dict
    self.max_len = max_len

    self.training_mention_tokens = training_mention_tokens
    self.training_mention_pos = training_mention_pos
    self.pair_indices_labels = pair_indices_labels
    self.entity_description_tokens_dict = entity_description_tokens_dict

    self.batch_size = batch_size
    self.top_k_neg = top_k_neg

    self.len_pos_pairs = (self.pair_indices_labels[:,0] == '1').sum()
    
    # each positive pair correspond to top_k_neg pairs
    self.num_pos_per_batch = self.batch_size//1
    self.num_neg_per_pairs = self.batch_size 

    # num of iterations:
    self.num_iterations = self.len_pos_pairs//self.num_pos_per_batch

  
  def __iter__(self):
    # get list of positive pair indices and negative pair indices from pair indices labels
    all_pos_pair_indices = np.where(self.pair_indices_labels[:,0] == '1')[0]
    start_pos_index = 0

    for i in range(self.num_iterations):
      batch_indices = [] 
      pos_batch_indices = all_pos_pair_indices[start_pos_index:start_pos_index + self.num_pos_per_batch ]
      start_pos_index += self.num_pos_per_batch
      
      with torch.no_grad():
        neg_batch_indices = []
        for i, pos_pair in enumerate(self.pair_indices_labels[pos_batch_indices]):
          anchor_idx = int(pos_pair[1])
          # get all candidate from pair_indices first
          neg_candidates = np.array(self.neg_indices_dict[anchor_idx])

          # narrow down to top-k
          neg_candidates_pairs = self.pair_indices_labels[neg_candidates]
          

          input_tokens_buffer = []
          for neg_pair in neg_candidates_pairs:
            label = neg_pair[0]
            mention_index = int(neg_pair[1])
            mention_cui = neg_pair[2]
            entity_description_tokens = self.entity_description_tokens_dict[mention_cui]

            input_tokens= MentionEntityAffinityDataset.generate_mention_entity_affinity_model_input(mention_index, 
                                                                                        self.training_mention_tokens,
                                                                                        self.training_mention_pos,
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

class MentionMentionBatchSampler(object):
  def __init__(self, model, device, 
               training_mention_tokens, 
               training_mention_pos,
               pair_indices_labels, 
               neg_pair_indices_dict,
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
    self.neg_indices_dict = neg_pair_indices_dict
    self.max_len = max_len

    self.training_mention_tokens = training_mention_tokens
    self.training_mention_pos = training_mention_pos
    self.pair_indices_labels = pair_indices_labels

    self.batch_size = batch_size
    self.top_k_neg = top_k_neg

    self.len_pos_pairs = (self.pair_indices_labels[:,0] == '1').sum()
    
    # each positive pair correspond to top_k_neg pairs
    self.num_pos_per_batch = self.batch_size//1
    self.num_neg_per_pairs = self.batch_size 

    # num of iterations:
    self.num_iterations = self.len_pos_pairs//self.num_pos_per_batch

  
  def __iter__(self):
    # get list of positive pair indices and negative pair indices from pair indices labels
    all_pos_pair_indices = np.where(self.pair_indices_labels[:,0] == 1)[0]
    start_pos_index = 0

    for i in range(self.num_iterations):
      batch_indices = [] 
      pos_batch_indices = all_pos_pair_indices[start_pos_index:start_pos_index + self.num_pos_per_batch ]
      start_pos_index += self.num_pos_per_batch
      
      with torch.no_grad():
        neg_batch_indices = []
        for i, pos_pair in enumerate(self.pair_indices_labels[pos_batch_indices]):
          anchor_idx = int(pos_pair[1])
          # get all candidate from pair_indices first
          neg_candidates = np.array(self.neg_indices_dict[anchor_idx])

          # narrow down to top-k
          neg_candidates_pairs = self.pair_indices_labels[neg_candidates]
          
          input_tokens_buffer = []
          mention_mask_a_buffer = []
          mention_mask_b_buffer = []

          for neg_pair in neg_candidates_pairs:
            label = neg_pair[0]
            mention_a_idx = neg_pair[1]
            mention_b_idx = neg_pair[2]
            input_tokens, mention_mask_a, mention_mask_b = MentionMentionAffinityDataset.generate_mention_mention_affinity_model_input(mention_a_idx, mention_b_idx, 
                                                                                                                                       self.training_mention_tokens, 
                                                                                                                                       self.training_mention_pos, 
                                                                                                                                       self.max_len)
            
            input_tokens_buffer.append(input_tokens)
            mention_mask_a_buffer.append(mention_mask_a)
            mention_mask_b_buffer.append(mention_mask_b)

          input_tokens_buffer = torch.stack(input_tokens_buffer).to(self.device)
          mention_mask_a_buffer = torch.stack(mention_mask_a_buffer).to(self.device)  
          mention_mask_b_buffer = torch.stack(mention_mask_b_buffer).to(self.device) 
          neg_affin = self.model(input_tokens_buffer, mention_mask_a_buffer, mention_mask_b_buffer)

          top_k_neg = torch.topk(neg_affin, self.top_k_neg, dim=0, largest=False, sorted=False)[1].cpu()
          top_k_neg = neg_candidates[top_k_neg].flatten().tolist()
          neg_batch_indices.extend(top_k_neg)


      batch_indices.extend(pos_batch_indices)
      batch_indices.extend(neg_batch_indices)

      yield batch_indices 
    
  
  def __len__(self):
    return self.num_iterations