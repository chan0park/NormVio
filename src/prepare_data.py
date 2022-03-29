import argparse
import os
import json
import pandas as pd
from copy import deepcopy
from os.path import join, exists
from tqdm import tqdm
from convokit import Corpus
from sklearn.model_selection import train_test_split
from config import SEP_STRING
from utils import save_df_to_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('--path_conversation', "-conv", type=str, required=True, help='conversation data path')
parser.add_argument('--path_comment', "-comment", type=str, required=True, help='labeled comment data path')
parser.add_argument('--path_save', "-save", type=str, required=True, help='where to save the data splits')
parser.add_argument('--keep_removed', "-keep", action="store_true", default=False, help='whether to delete the comments that were not restored')
parser.add_argument('--verbose', "-v", action='store_true', default=False, help='verbose')
args = parser.parse_args()


def load_comment2convid(meta, type):
    comment2convid = {}
    for conv_id, conv_meta in meta.items():
        comment_id = conv_meta['moderation_id']
        if type == "mod":
            comment2convid[comment_id] = conv_id
        else:
            if comment_id not in comment2convid:
                comment2convid[comment_id] = []
            comment2convid[comment_id].append(conv_id)
    return comment2convid

def import_corpus(path):
    corpus, meta, comment2convid = {}, {}, {}

    for type in ["mod","unmod"]:
        path_type = join(path,type)
        if exists(path_type):
            corpus[type] = Corpus(path_type)
            with open(join(path_type, "convo_meta.json"),"r") as file:
                meta[type] = json.loads(file.read())
            comment2convid[type] = load_comment2convid(meta[type], type)
    return corpus, meta, comment2convid

def get_utterances(conv):
    processed_uttr = []
    for uttr in conv.get_chronological_utterance_list():
        processed_uttr.append({"id":uttr.id,"tokens": uttr.text})
    return processed_uttr

def choose_unmod(conv_ids, corpus, target_num_uttr, num_unmod_uttr):
    conv_lens = [(conv_id, len(get_utterances(corpus.get_conversation(conv_id)))) for conv_id in conv_ids]
    convs = sorted(conv_lens, key=lambda t:abs(t[1]-target_num_uttr))
    convs = [t[0] for t in convs]
    return convs[:num_unmod_uttr]


def combine_data(corpus, meta, comment2convid, comments_labeled, num_unmod=2):
    data = {}
    for type in ["mod","unmod"]:
        data[type] = []
        for idx, row in tqdm(comments_labeled.iterrows(), total=len(comments_labeled)):
            comment_id, subreddit, rule_texts, cats = row['comment_id'], row['subreddit'], row['rule_texts'], row['cat']
            if comment_id in comment2convid[type]:
                bool_derail = (type=="mod")
                conv_ids = comment2convid[type][comment_id]
                if type =="mod":
                    conv_ids = [conv_ids]
                
                if len(conv_ids) > num_unmod:
                    moderated_uttr = get_utterances(corpus['mod'].get_conversation(comment2convid['mod'][comment_id]))
                    conv_ids = choose_unmod(conv_ids, corpus[type], len(moderated_uttr), num_unmod)

                for conv_id in conv_ids:
                    conv = corpus[type].get_conversation(conv_id)
                    
                    conv_meta = conv.meta
                    is_restored = conv_meta['is_restored'] if "is_restored" in conv_meta else None
                    
                    uttr = get_utterances(conv)
                    context, final_comment = uttr[:-1], uttr[-1]
                    data[type].append({"comment_id": comment_id, "conv_id":conv_id, "subreddit":subreddit, "context":context, "final_comment":final_comment, "bool_derail":bool_derail, "rule_texts":rule_texts, "cats":cats, "is_restored":is_restored})
        data[type] = pd.DataFrame(data[type])
    return data

def split_conversations(dfs, test_ratio=0.1, keep_removed=False):
    
    df = dfs["mod"]
    if not keep_removed:
        df = df[df.is_restored]
        print(f"filtered out not restored comments. ended up with {len(df)} derailed comments")
    test_size= round(len(df)*test_ratio)
    train, test = train_test_split(df, test_size=test_size, stratify=df[["is_restored"]])
    train, dev = train_test_split(train, test_size=test_size, stratify=train[["is_restored"]])
    df_split_mod = {"train":train, "dev":dev, "test":test}

    final_split = {}
    for split in ["train","dev","test"]:
        split_commentids = set(df_split_mod[split].comment_id.to_list())
        df_split_unmod = dfs["unmod"][dfs["unmod"].comment_id.isin(split_commentids)]
        final_split[split] = pd.concat([df_split_mod[split], df_split_unmod])
    return final_split

def post_process_data(df_split):
    new_data = {}
    for split in ["train","dev","test"]:
        df = df_split[split]
        new_data[split] = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            del row_dict['rule_texts']
            del row_dict['cats']
            rule_texts, cats = row['rule_texts'], row['cats']
            rule_texts, cats = rule_texts.split(SEP_STRING), cats.split(SEP_STRING)
            assert len(rule_texts) == len(cats)
            for rule_text, cat in zip(rule_texts, cats):
                if rule_text == "" or cat == "":
                    continue
                new_row_dict = deepcopy(row_dict)
                new_row_dict['rule_text'] = rule_text
                new_row_dict['cat']= cat.split(",")
                new_data[split].append(new_row_dict)
        new_data[split] = pd.DataFrame(new_data[split])
    return new_data

def save_splits(path, df_split):
    if not os.path.isdir(path):
        os.makedirs(path)
    for split in ["train","dev","test"]:
        path_split = join(path, split+".jsonl")
        save_df_to_jsonl(df_split[split], path_split)

assert os.path.isdir(args.path_conversation), f"{args.path_conversation} does not exist"
corpus, meta, comment2convid = import_corpus(args.path_conversation)
assert "unmod" in corpus

assert os.path.isfile(args.path_comment), f"{args.path_comment} does not exist"
comments_labeled = pd.read_csv(args.path_comment, sep="\t")

if args.verbose:
    print("scraped conversations:")
    for type in meta:
        print(f"# {type} conversation: {len(meta[type])}")

dfs_labeled = combine_data(corpus, meta, comment2convid, comments_labeled)
if args.verbose:
    print("after combining with cats:")
    for type in meta:
        print(f"# {type} conversation: {len(meta[type])}")


df_split = split_conversations(dfs_labeled, keep_removed=args.keep_removed)
if args.verbose:
    print("conversation data split")
    for split in ["train","dev","test"]:
        print(f"{split} # conversation: {len(df_split[split])}")


df_split = post_process_data(df_split)
if args.verbose:
    print("after splitting multiple rules into multiple rows:")
    for split in ["train","dev","test"]:
        print(f"{split} # training samples: {len(df_split[split])}")

save_splits(args.path_save, df_split)