import torch
import argparse
from torch.utils.data import DataLoader
from transformers import BertModel
from transformers import BertTokenizer
from data.dataset import AffinityDataset, BatchSampler
from src.models.affinity_models import MentionEntityAffinityModel, MentionMentionAffinityModel
from src.models.loss import TripletLosss
from src.utils import (
    create_pair_indices, 
    create_input_sentences, 
    load_processed_medmention, 
    create_neg_pair_indices_dict,
    load_sentence_tokens,
    load_all_entity_descriptions
    )

from tqdm import tqdm
import logging
import os

tokenizer = BertTokenizer.from_pretrained("nlpie/bio-distilbert-uncased",use_fast=True) #nlpie/bio-distilbert-uncased")
LOGGER = logging.getLogger()


def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='Train models')

    # Required
    parser.add_argument('--umls_dir_path', required=True, help='Directory of UMLs dataset')
    parser.add_argument('--st21pv_dir_path', required=True, help='Directory of ST21pv dataset')
    parser.add_argument('--st21pv_corpus_IOB2_format_dir_path', required=True, help='Directory of ST21pv corpus IOB2 format')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output model weights')
    
    # Optionals 
    parser.add_argument('--training_sentences_tokens_path', help='Path of training sentence tokens file')
    parser.add_argument('--all_entity_description_tokens_path', help='Path of all training description tokens file')
    

    # Tokenizer settings
    parser.add_argument('--max_length', default=256, type=int)

    # Train config
    parser.add_argument('--learning_rate', help='learning rate', default=0.00001, type=float)
    parser.add_argument('--batch_size', help='train batch size', default=16, type=int)
    parser.add_argument('--epoch_mention_entity', help='epoch to train mention entity model', default=2, type= int)
    parser.add_argument('--epoch_mention_mention', help='epoch to train mention mention model', default=5, type = int)

    args = parser.parse_args()

    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def main(args):
    init_logging()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
      
    if args.training_sentences_tokens_path is not None:
        sentence_tokens, mention_pos = load_sentence_tokens(args.training_sentences_tokens_path)
        
    if args.all_entity_description_tokens_dict is not None:
        all_entity_description_tokens_dict = load_all_entity_descriptions(args.all_entity_description_tokens_path)

   

    st21pv_corpus = load_processed_medmention('./data/processed/ST21pv_IOB2_formmat.txt')
    list_sentences, list_labels, list_sentence_docids = create_input_sentences(st21pv_corpus, './data/raw/ST21pv/data/corpus_pubtator_pmids_trng.txt')
    pair_indices = create_pair_indices(list_labels, list_sentence_docids)
    neg_pair_indices_dict = create_neg_pair_indices_dict(pair_indices)
    mention_entity_model = MentionEntityAffinityModel(tokenizer, base_model_path = None)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mention_entity_model.to(device)
    optimizer = torch.optim.Adam(mention_entity_model.parameters(), lr=1e-5)
    loss_func = TripletLosss(margin = 0.8)

    BATCH_SIZE = 16
    TOP_K_NEG = 1

    training_dataset = AffinityDataset(training_mention_tokens = training_mention_tokens, 
                                    training_mention_pos= training_mention_pos,
                                    pair_indices=pair_indices,
                                    all_entity_description_tokens_dict = all_entity_description_tokens_dict)

    train_data_loader = DataLoader(training_dataset, 
                                batch_sampler=BatchSampler(model=mention_entity_model, 
                                                            device= device,
                                                            training_mention_tokens = training_mention_tokens,
                                                            training_mention_pos = training_mention_pos,
                                                            pair_indices_labels = pair_indices,
                                                            neg_pair_indices_dict = neg_pair_indices_dict,
                                                            entity_description_tokens_dict = all_entity_description_tokens_dict)
                                )
    

    with tqdm(train_data_loader, unit="batch") as tepoch:
        mention_entity_model.train()
        for dataset, labels in tepoch: 
            pairs = dataset
            
            anchor_pos_batch = pairs[:BATCH_SIZE//TOP_K_NEG].to(device)
            anchor_neg_batch = pairs[BATCH_SIZE//TOP_K_NEG:].to(device)

            anchor_pos_batch = anchor_pos_batch.to(device)
            anchor_neg_batch = anchor_neg_batch.to(device)

            for i in range(TOP_K_NEG):
                anchor_pos_affin = mention_entity_model(anchor_pos_batch)
                anchor_neg_affin = mention_entity_model(anchor_neg_batch[i::TOP_K_NEG]) # anchor_neg_batch_indices[i::TOP_K_NEG])      print(time.time()-s)

                loss = loss_func(anchor_pos_affin, anchor_neg_affin)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())