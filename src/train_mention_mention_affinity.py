import argparse
import torch
from transformers import BertTokenizer
from transformers import BertModel
from data.dataset import MentionMentionDataset
from data.batch_sampler import MentionMentionBatchSampler
from torch.utils.data import DataLoader
from models.affinity_models import MentionMentionAffinityModel
from models.loss import TripletLosss
from tqdm import tqdm 

def train(args):
  data_dir = args.data_dir

  tokenizer = BertTokenizer.from_pretrained("nlpie/bio-distilbert-uncased",use_fast=True)
  tokenizer.add_special_tokens({'additional_special_tokens': ['[START]', '[END]']})
  model = BertModel.from_pretrained("nlpie/bio-distilbert-uncased") #dmis-lab/biobert-base-cased-v1.2") # ")
  model.resize_token_embeddings(len(tokenizer))

  mention_mention_model = MentionMentionAffinityModel(bert_base=model)

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


  mention_mention_dataset = MentionMentionDataset(data_dir=data_dir,
                                                 tokenizer=tokenizer)

  batch_sampler = MentionMentionBatchSampler(model=mention_mention_model,
                                            device=device, 
                                            context_data=mention_mention_dataset.context_data,
                                            pair_indices = mention_mention_dataset.pair_indices)

  data_loader = DataLoader(dataset= mention_mention_dataset,
                           batch_sampler= batch_sampler, 
                           shuffle=False)
 
  mention_mention_model.to(device)
  optimizer = torch.optim.Adam(mention_mention_model.parameters(), lr=1e-5)
  loss_func = TripletLosss(margin = 0.8)
  BATCH_SIZE = 16
  TOP_K_NEG = 1
  
  with tqdm(data_loader, unit="batch") as tepoch:
          mention_mention_model.train()
          for dataset, _ in tepoch: 
              pairs = dataset
              
              anchor_pos_batch = pairs[:BATCH_SIZE//TOP_K_NEG].to(device)
              anchor_neg_batch = pairs[BATCH_SIZE//TOP_K_NEG:].to(device)

              anchor_pos_batch = anchor_pos_batch.to(device)
              anchor_neg_batch = anchor_neg_batch.to(device)

              for i in range(TOP_K_NEG):
                  anchor_pos_affin = mention_mention_model(anchor_pos_batch)
                  anchor_neg_affin = mention_mention_model(anchor_neg_batch[i::TOP_K_NEG]) # anchor_neg_batch_indices[i::TOP_K_NEG])      print(time.time()-s)

                  loss = loss_func(anchor_pos_affin, anchor_neg_affin)
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  tepoch.set_postfix(loss=loss.item())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default="./data/processed/st21pv/train",
                      help = 'directory of data')

  args = parser.parse_args()

  train(args)
          