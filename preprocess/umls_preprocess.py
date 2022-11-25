from collections import defaultdict
from tqdm import tqdm
import argparse
import os


def load_umls_synonym(mrconso_file_path: str):
    """
    Return a dictionary which key is concept CUI and value is list of synonyms
    """
    with open(mrconso_file_path, "r") as f:
        data = f.read().split('\n')
        data = data[:-1]

        synonyms_dict = defaultdict(list)

        for row in tqdm(data):
            row = row.split('|')
            if row[1] == 'ENG':
                synonyms_dict[row[0]].append(row[-5])

        for key, value in synonyms_dict.items():
            synonyms_dict[key] = list(set(value))

    return synonyms_dict

def load_umls_semantic(mrsty_file_path:str) -> dict:
    """
    Return a dictionary which key is concept CUI and value is semantic type of that concept
    """

    with open(mrsty_file_path, "r") as f:
        data = f.read().split('\n')
        data = data[:-1]

        semantic_types = defaultdict(str)
        for row in tqdm(data):
            row = row.split('|')
            semantic_types[row[0]] = row[-4]

    return semantic_types

def main(args):

    mrconso_file = args.mrconso_file
    mrsty_file = args.mrsty_file
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    umls_synonym = load_umls_synonym(mrconso_file_path=mrconso_file)
    umls_semantic = load_umls_semantic(mrsty_file_path=mrsty_file)

    with open(output_dir + "dictionary.txt", "w") as f:
        data = []
        for cui, list_synonyms in umls_synonym.items():
            semantic_type = umls_semantic[cui]
            data.append(cui + '||' + semantic_type + '||' + '|'.join(list_synonyms))
        
        f.write('\n'.join(data))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mrconso_file', type=str,
                    default="./data/raw/UMLs/MRCONSO.RRF",
                    help='path of corpus file')
    parser.add_argument('--mrsty_file', type=str,
                    default="./data/raw/UMLs/MRSTY.RRF",
                    help='path of input file (train/test)')                
    parser.add_argument('--output_dir', type=str,
                    default="./data/processed/umls/", 
                    help='path of output directionary')

    args = parser.parse_args()
    main(args)

