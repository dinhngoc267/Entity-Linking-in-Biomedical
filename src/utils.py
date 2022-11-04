from collections import defaultdict

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

        print(f'There are {len(synonyms_dict)} concepts in UMLS')

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

