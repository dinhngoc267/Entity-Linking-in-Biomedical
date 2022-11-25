from tqdm import tqdm
from collections import defaultdict
import stanza
import os
import argparse

def load_corpus_into_dictionary(corpus_file: str):
    """
    input_file: string - file path of corpus
    """
    corpus = {}
    
    with open(corpus_file, "r") as f:
        data = f.read().split('\n\n')

    for item in data:
        lines = item.split('\n')

        id = None
        title = ''
        abstract = ''
        list_entity_annotations = []

        for line in lines:
            if '|t|' in line:
                title = line.split('|t|')[1]
                id = line.split('|t|')[0]
            elif '|a|' in line:
                abstract = line.split('|a|')[1]   
            else:
                entity_annotation = '\t'.join(line.split('\t')[1:4] + [line.split('\t')[5].split(':')[1]])
                list_entity_annotations.append(entity_annotation)
        
        if len(list_entity_annotations) > 0:
            text = title + '\n' + abstract + '\n'
            corpus[id] = text + '\n'.join(list_entity_annotations)

    return corpus

def delete_overlapping_mentions(corpus: dict):

    deleted_terms = []

    for id, item in corpus.items():
        lines = item.split('\n')
        text = lines[0] + '\n' + lines[1]
        entities = lines[2:]

        positions = []
        for entity in entities:
            splits = entity.split('\t')
            s = splits[0]
            e = splits[1]
            positions.append([s,e, splits[2], splits[3],id])
        
        i = 1
        while i < len(positions):
            if int(positions[i][0]) < int(positions[i-1][1]):
                if int(positions[i][0]) == int(positions[i-1][0]) and int(positions[i-1][1]) < int(positions[i][1]):
                    deleted_terms.append(positions[i-1])
                    positions.pop(i-1)
                    continue
                else:
                    deleted_terms.append(positions[i])
                    positions.pop(i)
            else:
                i += 1


    new_corpus = {}
    for id, item in corpus.items():
        lines = item.split('\n')
        text = lines[0] + '\n' + lines[1]
        entities = lines[2:]
        new_entities = []

        for entity in entities:
            splits = entity.split('\t')
            tmp = splits[:4] + [id]

            if tmp in deleted_terms:
                deleted_terms.remove(tmp)
            else: 
                new_entities.append(entity)

        tmp = text + '\n' + '\n'.join(new_entities)
        new_corpus[id] = tmp

    return new_corpus  

def load_Ab3P_output(file_path):
    with open(file_path, "r") as f:
        data = f.read().split('\n\n')
        abbreviation_dict = defaultdict(list)

        for item in data:
            arr = item.split('\n')
            id = arr[0]
            for row in arr[1:]:
                if '|' in row and len(row.split('|')) == 3:
                    row = row.split('|')
                    abbreviation_dict[id].append([row[0].replace('\t',''), row[1]])

    return abbreviation_dict

def load_Ab3P_output(Ab3P_output_file):
    with open(Ab3P_output_file, "r") as f:
        data = f.read().split('\n\n')
        abbreviation_dict = defaultdict(list)

        for item in data:
            arr = item.split('\n')
            id = arr[0]
            for row in arr[1:]:
                if '|' in row and len(row.split('|')) == 3:
                    row = row.split('|')
                    abbreviation_dict[id].append([row[0].replace('\t',''), row[1]])

    return abbreviation_dict

def replace_abbr_with_long_form(corpus: dict, abbreviation_dict: dict):

    new_corpus = {}

    for id, item in tqdm(corpus.items()):
        lines = item.split('\n')
        
        if id not in abbreviation_dict:
            new_corpus[id] = item   
            continue
    
        title = lines[0] + "."
        len_title = len(title)
        abstract = lines[1]
        entities = lines[2:]
        
        text = title + '\n' + abstract

        list_entities = []
        add = 0
        flag = True
        
        for i,entity in enumerate(entities):
            entity = entity.split('\t')
            if i >= 1 and entities[i].split('\t')[:-1] == entities[i-1].split('\t')[:-1]:
                tmp = '\t'.join(list_entities[-1].split('\t')[:-1]) + '\t' + entity[-1] + '\n'
                list_entities.append(tmp)
                continue
            name = entity[2]
            new_name = name
            s = entity[0]
            e = entity[1]
            if int(s) >= len_title:
                s = str(int(s) + 1)
                e = str(int(s) + 1)

            values = abbreviation_dict[id]
            for pair in values:
                if name == pair[0]:
                    new_name = pair[1]
                    break

            if flag == False:
                s = str(int(s) + add)
            else:
                flag = False
        
            text = text[:int(s)] + new_name + text[int(s)+len(name):] 
            add += len(new_name) - len(name)
            e = str(int(s) + len(new_name))
            list_entities.append(s+'\t'+e+'\t' + new_name +'\t' + entity[-1])

        tmp = text + '\n' + '\n'.join(list_entities)
        new_corpus[id] = tmp
        tmp = ''
        list_entities = []
        
    return new_corpus


def main(args):
    
    corpus_file = args.corpus_file
    input_file = args.input_file
    output_dir = args.output_dir
    ab3p_output_file = args.ab3p_output_file

    # create output if output dir does not exsits
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read input_file
    with open(input_file, "r") as f:
        input_ids = f.read().split('\n')

    corpus = load_corpus_into_dictionary(corpus_file=corpus_file)
    corpus = delete_overlapping_mentions(corpus=corpus)

    abbreviations_dict = load_Ab3P_output(ab3p_output_file)
    corpus = replace_abbr_with_long_form(corpus=corpus, abbreviation_dict=abbreviations_dict)

    st = stanza.Pipeline(lang='en', processors='tokenize')

    sents_dict = {}

    for id, doc in tqdm(corpus.items()):
        if id in input_ids:
            lines = doc.split('\n')
            
            text = lines[0] + '\n' + lines[1]
            list_entity_annotations = [x.split('\t') for x in lines[2:]]

            doc = st(text)
            list_sents = []

            for sentence in doc.sentences:
                list_tokens = []
                exist_mention = False
                for token in sentence.tokens:
                    string = token.text
                    start_c = token.start_char

                    flag = False
                    for entity in list_entity_annotations:
                        if int(entity[0]) == start_c:
                            flag = True
                            list_tokens.append([string, 'B:'+ entity[-1]])
                            exist_mention = True
                            break
                        elif int(entity[0]) < start_c and int(entity[1]) > start_c:
                            exist_mention = True
                            flag = True
                            list_tokens.append([string, 'I:'+ entity[-1]])
                            break
                    if flag==False:
                        list_tokens.append([string, 'O'])
                if exist_mention == True:
                    list_sents.append(list_tokens)

            sents_dict[id] = list_sents
            
            output_file = os.path.join(output_dir, "{}.txt".format(id))

            with open(output_file, "w") as f:
                for sent in list_sents:
                    for word_tag in sent:
                        f.write('\t'.join(word_tag) + '\n')
                
                    f.write('\n')

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', type=str,
                    default="./data/raw/ST21pv/corpus_pubtator.txt",
                    help='path of corpus file')
    parser.add_argument('--input_file', type=str,
                    default="./data/raw/ST21pv/corpus_pubtator_pmids_trng.txt",
                    help='path of input file (train/test')                
    parser.add_argument('--output_dir', type=str,
                    default="./data/processed/st21pv/train", 
                    help='path of output directionary')
    parser.add_argument('--ab3p_output_file', type=str,
                    default="./data/externel/st21pv/ab3p_output.txt", 
                    help='path of Ab3P output file')

    args = parser.parse_args()
    main(args)

    