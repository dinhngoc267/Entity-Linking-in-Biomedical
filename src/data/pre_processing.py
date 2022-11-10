from collections import defaultdict
from tqdm import tqdm
import stanza
stanza.download('en')

def load_st21pv_corpus(file_path: str):
    with open(file_path, "r") as f:
        data = f.read().split('\n\n')
        data = data[:-1]
        corpus_st21pv = []

        for item in data:
            arr = item.split('\n')
            id = arr[0].split('|')[0]
            title = arr[0][11:]
            abstract = arr[1][11:]
            tmp = id + '\n' + title + '\n' + abstract + '\n'

            for row in arr[2:]:
                splits = row.split('\t')
                tmp += '\t'.join(splits[1:4] + [splits[5]]) + '\n'
            tmp = tmp[:-1]
            corpus_st21pv.append(tmp)
    return corpus_st21pv

def delete_overlapping_mentions(corpus_st21_pv):

    deleted_terms = []

    for item in corpus_st21_pv:
        arr = item.split('\n')
        id = arr[0]
        text = arr[1] + '\n' + arr[2]
        entities = arr[3:]

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


    new_corpus_st21pv = []
    term_deleted = []
    for item in corpus_st21_pv:
        arr = item.split('\n')
        id = arr[0]
        
        text = arr[1] + '\n' + arr[2]
        entities = arr[3:]
        new_entities = []

        for entity in entities:
            splits = entity.split('\t')
            s = splits[0]
            e = splits[1]
            tmp = [s,e, splits[2], splits[3],id]
            if tmp in deleted_terms:
                term_deleted.append(entity.split('\t') + [id])
                deleted_terms.remove(tmp)
            else: 
                new_entities.append(entity)

        tmp = id+ '\n' + text + '\n'
        for i in new_entities:
            tmp += i + '\n'
        tmp = tmp[:-1]
        new_corpus_st21pv.append(tmp)

    return new_corpus_st21pv

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

def replace_abbr_with_long_form(corpus_st21_pv, abbreviation_dict):

    corpus_st21pv_lf = []

    for item in corpus_st21_pv:
        arr = item.split('\n')
        id = arr[0]
        
        if id not in abbreviation_dict:
            corpus_st21pv_lf.append(item)   
            continue
    
        title = arr[1] + "."
        len_title = len(title)
        abstract = arr[2]
        entities = arr[3:]
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
            list_entities.append(s+'\t'+e+'\t' + new_name +'\t' + entity[-1] + '\n')

        tmp = id + '\n' + text + '\n'
        for i in list_entities:
            tmp += i
        tmp = tmp[:-1]
        corpus_st21pv_lf.append(tmp)
        tmp = ''
        list_entities = []
        
    return corpus_st21pv_lf


def convert_IOB2_format(file_corpus_path, file_Ab3P_output, output_dir:str):
    st = stanza.Pipeline(lang='en', processors='tokenize')
    
    st21pv_corpus = load_st21pv_corpus(file_corpus_path)
    st21pv_corpus = delete_overlapping_mentions(st21pv_corpus)
    abbreviation_dict = load_Ab3P_output(file_Ab3P_output)
    st21pv_corpus = replace_abbr_with_long_form(st21pv_corpus, abbreviation_dict)

    IOB2_tags = defaultdict(list)
    count = 0
    for item in tqdm(st21pv_corpus):
        arr = item.split('\n') 
        id = arr[0]
        text = arr[1] + '\n' + arr[2]
        entities = arr[3:]

        list_entities = []
        for entity in entities:
            splits = entity.split('\t')
            s = int(splits[0])
            e = int(splits[1])
            name = splits[2]
            cui = splits[3].split(':')[1]

            list_entities.append([s,e,name, cui])

        doc = st(text)

        list_sents = []


        for sentence in doc.sentences:
            list_tokens = []
            exist_mention = False
            for token in sentence.tokens:
                string = token.text
                start_c = token.start_char
                end_c = token.end_char

                flag = False
                for entity in list_entities:
                    if entity[0] == start_c:
                        count += 1
                        flag = True
                        list_tokens.append([string, 'B:'+ entity[-1]])
                        exist_mention = True
                        break
                    elif entity[0] < start_c and entity[1] > start_c:
                        exist_mention = True
                        flag = True
                        list_tokens.append([string, 'I:'+ entity[-1]])
                        break
            if flag==False:
                list_tokens.append([string, 'O'])
            if exist_mention == True:
                list_sents.append(list_tokens)
        IOB2_tags[id] = list_sents

    with open(output_dir + "/IOB2_tags.txt", 'w') as f:
        for key, values in IOB2_tags.items(): # sentences
            f.write(key + '\n')
            for sentence in values:
                for token in sentence:
                    f.write(token[0] + '\t' + token[1] + '\n')
                f.write('\n')  
            f.write('\n')

    with open(output_dir + "all_mention.txt", 'w') as f:
        for item in st21pv_corpus:
            rows = item.split('\n')
            entities = rows[3:]

            for entity in entities:
                entity = entity.split('\t')
                f.write(rows[0] + '||' + entity[2] + '||' + entity[-1].split(':')[-1] + '\n')
    return IOB2_tags