import numpy as np
import nltk
nltk.download('stopwords')
import math
import glob
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from tqdm import tqdm 


class GenerateCandidateModel():
    def __init__(self, data_dir, dictionary_file, char_ngram_range = (2,5), max_features = 400000, metrix = "cosine"):
        """
        Params:
            data_dir: directory of data files
            dictionary_file: file path of dictionary
        """
        super().__init__()
        self.char_tfidf = TfidfVectorizer(analyzer='char',
                                    lowercase=True,
                                    ngram_range=char_ngram_range,
                                    max_features=400000, 
                                    dtype=np.float32)

        self.word_tfidf = TfidfVectorizer(analyzer='word', 
                                     lowercase =True, 
                                     ngram_range=(1, 1), 
                                     dtype=np.float32, 
                                     stop_words = stopwords.words('english'), 
                                     token_pattern='[a-zA-Z0-9_]{1,}')
        
        entities = []
        entity_cuis = []
        mentions = []
        mention_cuis = []

        # load mentions
        data_files = glob.glob(os.path.join(data_dir, "*.txt"))

        for file in data_files:
            with open(file, "r") as f:
                lines = f.read().split('\n')
                
                for line in lines:
                    cui = line.split('||')[0]
                    name = line.split('||')[1]
                    mentions.append(name)
                    mention_cuis.append(cui)

        # load entities
        with open(dictionary_file, "r") as f:
            lines = f.read().split('\n')
            
            for line in lines:
                names = line.split('||')[2].split('|')
                cui = line.split('||')[0]
                entities.extend(names)
                entity_cuis += len(names)*[cui]

        corpus = list(set(mentions + entities))

        self.mentions = list(set(mentions))
        self.mention_cuis = mention_cuis
        self.entities = entities
        self.entity_cuis = entity_cuis
        self.fit(corpus)
        
    def fit(self, corpus):
        self.char_tfidf.fit(corpus)
        self.word_tfidf.fit(corpus)
    

    def generate_candidates(self, output_file_path, top_k = 128, batch_size = 128):
        """
        Return indices of top k candiates in corpus 
        """   

        mention_char_sparse_matrix = self.char_tfidf.transform(self.mentions)
        entity_char_sparse_matrix = self.char_tfidf.transform(self.entities)
        
        mention_word_sparse_matrix = self.word_tfidf.transform(self.mentions)
        entity_word_sparse_matrix = self.word_tfidf.transform(self.entities)       

        candidates = {}
        count = 0
        with tqdm(range(math.ceil(mention_char_sparse_matrix.shape[0]/batch_size)), unit="batch") as tepoch:
            for n_batch in tepoch:  
            # compute cosine similarity
                cosine_sim = linear_kernel(mention_char_sparse_matrix[n_batch*batch_size:(n_batch+1)*batch_size], entity_char_sparse_matrix)
                for row in cosine_sim:
                    # get index of the k highest elements
                    top_k_ind = np.argpartition(row, -top_k)[-top_k:]
                    top_k_ind = top_k_ind[np.argsort(row[top_k_ind])][::-1]
                    list_cui_candidates = [self.entity_cuis[idx] for idx in top_k_ind]
                    candidates[self.mentions[count]] = list(set(list_cui_candidates))   
                    count += 1

        count = 0
        with tqdm(range(math.ceil(mention_word_sparse_matrix.shape[0]/batch_size)), unit="batch") as tepoch:
            for n_batch in tepoch:  
            # compute cosine similarity
                cosine_sim = linear_kernel(mention_word_sparse_matrix[n_batch*batch_size:(n_batch+1)*batch_size], entity_word_sparse_matrix)
                for row in cosine_sim:
                    # get index of the k highest elements
                    top_k_ind = np.argpartition(row, -top_k)[-top_k:]
                    top_k_ind = top_k_ind[np.argsort(row[top_k_ind])][::-1]
                    list_cui_candidates = [self.entity_cuis[idx] for idx in top_k_ind]
                    candidates[self.mentions[count]] += list(set(list_cui_candidates))   
                    count += 1       

        with open(output_file_path, 'w') as f:
            data = []
            for mention, list_candidates in candidates.items():
                data.append(mention + '||' + ' '.join(list_candidates))

            f.write('\n'.join(data))

        return candidates


def main(args):


    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                    default="./data/raw/ST21pv/corpus_pubtator.txt",
                    help='path of corpus file')
    parser.add_argument('--input_file', type=str,
                    default="./data/raw/ST21pv/corpus_pubtator_pmids_trng.txt",
                    help='path of input file (train/test)')                
    parser.add_argument('--output_dir', type=str,
                    default="./data/processed/st21pv/train", 
                    help='path of output directionary')
    parser.add_argument('--ab3p_output_file', type=str,
                    default="./data/externel/st21pv/ab3p_output.txt", 
                    help='path of Ab3P output file')

    args = parser.parse_args()
    main(args)