from transformers import BertTokenizer
from transformers import BertModel



def main(args):
  data_dir = args.data_dir
  dictionary_file = args.dictionary_file
  candidate_file = args.candidate_file

  tokenizer = BertTokenizer.from_pretrained("nlpie/bio-distilbert-uncased",use_fast=True)
  tokenizer.add_special_tokens({'additional_special_tokens': ['[START]', '[END]']})
  model = BertModel.from_pretrained("nlpie/bio-distilbert-uncased") #dmis-lab/biobert-base-cased-v1.2") # ")
  model.resize_token_embeddings(len(tokenizer))


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
          