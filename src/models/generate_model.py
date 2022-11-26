from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm 
import math
from sklearn.metrics.pairwise import linear_kernel

class CandidateGenerator():
    def __init__(self, data_dir, dictionary_file, char_ngram_range = (2,5), max_features = 400000, metrix = "cosine"):
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
        

        files = glob.glob(os.path.join(data_dir, "*.context"))

        for file in tqdm(files):
            with open(file, "r") as f:
                list_sents = f.read().split('\n\n')

            for sent in list_sents.split:
                sent_token_ids, mention_index, label = tokenize_sentence(sent.split('\n'), tokenizer)
                self.context_data.append((sent_token_ids, mention_index))
                self.labels.append(label)
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
        result = []
        with open(output_file_path, 'w') as f:
            line = 0

            with tqdm(range(math.ceil(query_sparse_matrix.shape[0]/batch_size)), unit="batch") as tepoch:
                for n_batch in tepoch:  
                # compute cosine similarity
                    cosine_sim = linear_kernel(query_sparse_matrix[n_batch*batch_size:(n_batch+1)*batch_size], corpus_sparse_matrix)
                    for row in cosine_sim:
                        # get index of the k highest elements
                        top_k_ind = np.argpartition(row, -top_k)[-top_k:]
                        top_k_ind = top_k_ind[np.argsort(row[top_k_ind])][::-1]

                        result.append(top_k_ind)
                        # write result to file
                        f.write(str(line)+ '||' + " ".join([str(x) for x in top_k_ind]) + "\n")
        return result
