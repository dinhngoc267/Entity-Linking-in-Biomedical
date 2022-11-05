from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm

def load_umls(mrconso_file_path: str) -> dict:
    """
    Return a dictionary which key is concept CUI and value is list of synonyms
    """
    with open(mrconso_file_path, "r") as f:
        data = f.read().split('\n')
        data = data[:-1]

        synonyms_dict = defaultdict(list)

        for row in data:
            row = row.split('|')
            if row[1] == 'ENG':
                synonyms_dict[row[0]].append(row[-5])

        for key, value in synonyms_dict.items():
            synonyms_dict[key] = list(set(value))

    return synonyms_dict

def load_semantic_type(mrsty_file_path:str) -> dict:
    """
    Return a dictionary which key is concept CUI and value is semantic type of that concept
    """

    with open(mrsty_file_path, "r") as f:
        data = f.read().split('\n')
        data = data[:-1]

        semantic_types = defaultdict(str)
        for row in data:
            row = row.split('|')
            semantic_types[row[0]] = row[-4]

    return semantic_types

def create_list_of_all_entities(synonyms_dict: dict) -> tuple:
    """
    From synonym dictionary, create list of all entities (concept names) and correspondding cuis, and set of all cuis in umls
    """
    all_entities_label = []
    all_entities = []
    set_cuis= []
    for key, values in synonyms_dict.items():
        for value in values:
            all_entities.append(value)
            all_entities_label.append(key)
        set_cuis.append(key)

    return all_entities, all_entities_label, set_cuis

def load_processed_medmention(processed_file_path:str)->dict:
    """
    Load the processed med mention data and save to dictionary.
    Where:
        key is id of the document
        value is string of the content of document and list of mentions 
    """
    with open(processed_file_path,"r") as f:
        st21pv_corpus = {}
        data = f.read().split('\n\n\n')

        for item in data:
            id = item.split('\n')[0]
            item = item[len(id) + 1 :]
            
            sents = item.split('\n\n')
            st21pv_corpus[id] = sents

    return st21pv_corpus


def create_input_sentences(st21pv_corpus: dict, data_docid_file_path: str) ->tuple:
    """
    Create input sentences. 
    Return:
        list_sentences: list of sentences tokenized from documents. Each sentence is a list of tokens (words) with tag.
        list_labels: the cui of the mention which sentence contains
        list_sentence_docids: list of docid of the sentence
        list_mentions: 
    """
    with open(data_docid_file_path) as f:
        data = f.read().split('\n')

        documents = []
        document_ids = []
        for id in data:
            documents.append(st21pv_corpus[id])
            document_ids.append(id)
    
    list_sentences = []
    list_labels = []
    list_sentence_docids = []
    

    for docid, doc in enumerate(documents):
        for sent in doc:
            tokens = sent.split('\n')
            mention_count = 0

            for token in tokens:
                splits = token.split('\t')
                token_str = splits[0]
                token_label = splits[1]

                if 'B' in token_label:
                    mention_count += 1 
                    
            flag = 1
            
            for i in range(mention_count):
                label = None
                count_mention = 0
                tmp = []
                already_take = False
                entity = ""
                for token in tokens:
                    splits = token.split('\t')
                    token_str = splits[0]
                    token_label = splits[1]    
                    
                    if 'B' in token_label:
                        if already_take == False:
                            count_mention += 1
                        if count_mention == flag:
                            tmp += [token]
                            entity += token_str + " "
                            havent_met_O = True
                            label = token_label.split(':')[1]
                            flag += 1
                            already_take = True
                        else:
                            tmp += [token_str + '\t' + 'O' + '\n']
                    else: 
                        if 'I' in token_label and havent_met_O == True:
                            entity += token_str + " "
                        if 'O' in token_label:
                            havent_met_O = False
                    tmp += [token]
            
            list_sentences.append(tmp)
            list_labels.append(label)
            list_sentence_docids.append(docid)
    
    return list_sentences, list_labels, list_sentence_docids


def create_pair_indices(list_labels: list, list_sentence_docids: list) ->tuple:
    """
    label_index_dict: a dictionary where:
                        key: CUI of mention in the sentence. 
                        value: indices of all sentences contain mention bellong to that CUI
    docid_index_dict: a dictionary where:
                        key: docid of a sentence
                        value: indices of all sentences in that document  
    """
    labels_index_dict = defaultdict(list)    
    docid_index_dict = defaultdict(list)

    for i,label in enumerate(list_labels):
        labels_index_dict[label].append(i)

    for i,docid in enumerate(list_sentence_docids):
        docid_index_dict[docid].append(i)

    pos_pair_indices = []
    for key, value in labels_index_dict.items():  
        res = []
        i = 0
        while i < len(value)//2:
            res.append((value[i],value[i+1]))
            i += 2
        #res = [(x,y) for i,x in enumerate(value) for y in value[i+1:]]
        pos_pair_indices += res
    
    neg_indices_dict = defaultdict(list)

    for anchor_index in range(len(list_labels)):
        # get id of document of anchor
        docid = list_sentence_docids[anchor_index]
        sentence_label = list_labels[anchor_index]

        # get all mentions in that docid
        sentence_indices = docid_index_dict[docid]

        # filter negative sentence
        neg_indices = list(set([x for x in sentence_indices if list_labels[x] != sentence_label]))
        neg_indices_dict[anchor_index] = neg_indices


    neg_pair_indices = []
    pair_indices = []

    for key, values in neg_indices_dict.items():
        for value in values:
            neg_pair_indices.append([key, value])

    for index_1, index_2 in pos_pair_indices:
        pair_indices.append([1, index_1, index_2])
    for index_1, index_2 in neg_pair_indices:
        pair_indices.append([0, index_1, index_2])

    # shuffle
    pair_indices = np.array(pair_indices)
    idx = np.arange(len(pair_indices))
    random.shuffle(idx)
    pair_indices = pair_indices[idx]

    return pair_indices


def create_neg_pair_indices_dict(pair_indices: list) -> dict:
    neg_pair_indices_dict = defaultdict(list)
    for i, pair in enumerate(pair_indices):
        if int(pair[0]) == 0:
            anchor_index = pair[1]
            neg_pair_indices_dict[anchor_index].append(i)
    for key, value in neg_pair_indices_dict.items():
        neg_pair_indices_dict[key] = np.array(value)

    return neg_pair_indices_dict

def load_negative_candidates(candidates_file_path: str, all_entity_labels: list, all_mention_labels) -> dict:
    with open(candidates_file_path, "r") as f:
        data = f.read().split('\n')
        data = data[:-1]
        all_negative_candidates_indices = defaultdict(list)
        
        for idx, row in enumerate(data):
            row = row.split('||')
            all_negative_candidates_indices[int(row[0])] = list(set([all_entity_labels[int(x)] for x in row[1].split(" ") if all_entity_labels[int(x)] != all_mention_labels[idx]]))
    
    return all_negative_candidates_indices

def tokenize_sentences(sentences, tokenizer):
    sentence_tokens = []
    start = None
    end = None
    flag = False

    for item in enumerate(sentences):
        splits = item.split('\t')
        word = splits[0]
        word_label = splits[1]

        if 'B' in word_label:
            sentence_tokens += ['[START]']
            start = len(sentence_tokens)
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


def load_sentence_tokens(file_path):
    sentence_tokens = []
    mention_pos = []
    with open(file_path, 'r') as f:
        data = f.read().split('\n')
        data = data[:-1]

        for row in data:
            row = row.split('||')
            sentence_tokens.append([int(x) for x in row[1].split(" ")])
            mention_pos.append([int(x) for x in row[2].split(" ")])
    
    return sentence_tokens, mention_pos

def load_all_entity_descriptions(file_path):
    with open(file_path, "r") as f:
        data = f.read().split('\n')
        data = data[:-1]
        
        all_entity_description_tokens_dict = {}

        for row in data:
            row = row.split('||')
            cui = row[0]
            entity_description_tokens = [int(x) for x in row[1].split(" ")]
            all_entity_description_tokens_dict[cui] = entity_description_tokens
    
    return all_entity_description_tokens_dict


def tokenize_sentences(sentences, tokenizer):
  sentence_tokens = []
  mention_positions = []

  for sentence in tqdm(sentences):
    tmp, [s, e] = tokenize_sentences(sentence, tokenizer)
    sentence_tokens.append(tmp)
    mention_positions.append([s,e])

  return sentence_tokens, mention_positions

