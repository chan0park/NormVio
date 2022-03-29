import os
import concurrent.futures
import pandas as pd
import json
from config import PRAW_CLIENT_ID, PRAW_CLIENT_SECRET

def check_or_create_dir(path_dir):
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    
def check_if_path_exists(path):
    assert os.path.exists(path), path+" does not exist!"

def import_tsv(path):
    check_if_path_exists(path)
    df = pd.read_csv(path, sep="\t" if not path.endswith(".csv") else ",", on_bad_lines='skip')
    return df

def import_json(path):
    check_if_path_exists(path)
    with open(path, "r") as file:
        data = json.loads(file.read())
    return data

def import_rule_lexicon(path):
    def remove_empty(lst):
        return [l for l in lst if l!=""]
    check_if_path_exists(path)
    if path.endswith(".json"):
        dict_lexicon = import_json(path)
    elif path.endswith(".tsv"):
        df_lexicon = import_tsv(path)
        dict_lexicon = {}
        for _, row in df_lexicon.iterrows():
            cat = row['cat24']
            lexicon = [w.strip() for w in row['lexicon'].split(",")]
            dict_lexicon[cat] = lexicon
    dict_lexicon = {cat:remove_empty(lst) for cat,lst in dict_lexicon.items()}
    return dict_lexicon

def import_jsonl(path, as_panda=False):
    data = []
    with open(path,"r") as file:
        for line in file:
            _data = json.loads(line)
            data.append(_data)
    if as_panda:
        data = pd.DataFrame(data)
    return data

def save_json(dic, path):
    with open(path, "w") as file:
        file.write(json.dumps(dic))
    
def save_df_to_jsonl(df, path, split=False, test_size=0.2):
    from sklearn.model_selection import train_test_split
    if split:
        train, dev = train_test_split(df, test_size=test_size)
        path_train, path_dev = path.replace(".json","_train.json"), path.replace(".json","_dev.json")
        train.to_json(path_train, orient="records", lines=True)
        dev.to_json(path_dev, orient="records", lines=True)
    else:
        df.to_json(path, orient="records", lines=True)

def get_praw_instance():
    import praw
    reddit = praw.Reddit(client_id=PRAW_CLIENT_ID, client_secret=PRAW_CLIENT_SECRET,
                                user_agent='python:kappa:v0.1')
    return reddit

def get_comment_data_from_id(reddit, comment_id):
    return reddit.comment(comment_id)
