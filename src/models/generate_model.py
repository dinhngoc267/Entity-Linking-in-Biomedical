from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm 
import math
from sklearn.metrics.pairwise import linear_kernel

class GenerateCandidateModel():
    def __init__(self, corpus, char_ngram_range = (2,5), max_features = 400000, metrix = "cosine"):
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
        
        self.corpus = corpus
        self.fit()
        
    def fit(self):
        self.char_tfidf.fit(self.corpus)
        self.word_tfidf.fit(self.corpus)
    
    def sparse_transform(self, x, use_char_tfidf = True):
        """
        if use_char_tfidf is False, model will use word_tfidf
        Return:
            - sparse matrix has shape: [x.size(), vocab_length]
        """

        if use_char_tfidf:
            return self.char_tfidf.transform(x)
        else:
            return self.word_tfidf.transform(x)

    def generate_candidates(self, query, corpus, top_k, output_file_path, use_char_tfidf = True):
        """
        Return indices of top k candiates in corpus 
        """        
        query_sparse_matrix = self.char_tfidf.transform(query)
        corpus_sparse_matrix = self.char_tfidf.transform(corpus)

        batch_size = 100

        MAX_K = 64
        with open(output_file_path, 'w') as f:
            line = 0
            #count = 0
            #result_top_k = [0] * 7
            #result_top_k = np.array(result_top_k)
            #top_k = [1, 2, 4, 8, 16, 32, 64]

            with tqdm(range(math.ceil(query_sparse_matrix.shape[0]/batch_size)), unit="batch") as tepoch:
                for n_batch in tepoch:  
                # compute cosine similarity
                    cosine_sim = linear_kernel(query_sparse_matrix[n_batch*batch_size:(n_batch+1)*batch_size], corpus_sparse_matrix)
                    for row in cosine_sim:
                        # get index of the k highest elements
                        top_k_ind = np.argpartition(row, -MAX_K)[-MAX_K:]
                        top_k_ind = top_k_ind[np.argsort(row[top_k_ind])][::-1]

                        # write result to file
                        f.write(str(line)+ '||' + " ".join([str(x) for x in top_k_ind]) + "\n")

                        # check if the candidates contain ground truth
                        # for i, index in enumerate(top_k_ind):
                        #     if all_entities_label[index] == all_mentions_labels[line]:    
                        #         for idx, k in enumerate(top_k):
                        #         if k >=i:
                        #             result_top_k[idx] += 1
                        #         break
                        #     line += 1
                    #tepoch.set_postfix(recall_top_k = ' '.join([str(round(x,2)) for x in result_top_k/line]))
        return
