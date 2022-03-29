import os
import sys
import json
import pandas as pd
from os.path import join
from config import SEP_STRING
from utils import import_jsonl, import_json, import_tsv
from tqdm import tqdm

def convert_df_rules(df_rules, exclude=set(["Behavior_Content_Format_Allowed"])):
    final_df_rules = []
    for _, row in tqdm(df_rules.iterrows(), total=len(df_rules)):
        converted_row = {"rule":row['sentence'], 'subreddit':row['subreddit'], 'idx':row['idx']}
        cats = []
        for cat, label in row.iteritems():
            if cat in exclude:
                continue
            if label not in [0,1]:
                continue
            if cat == "idx":
                continue
            if label == 1:
                cats.append(cat)
        converted_row['cats'] = cats
        final_df_rules.append(converted_row)
    return pd.DataFrame(final_df_rules)

def map_rule_nums_text_to_cats(df_rule_subreddit, rule_num_txt, exclude=["Behavior_Content_Format_Allowed"]):
    if df_rule_subreddit is None:
        return None
    rule_num_txt = str(rule_num_txt)
    rule_nums = rule_num_txt.split(SEP_STRING)
    cats  = []
    for num in rule_nums:
        _df_rule = df_rule_subreddit[df_rule_subreddit.idx == int(num)]
        if len(_df_rule) > 1:
            print(str(num)+"\n"+str(_df_rule))
        if len(_df_rule) != 1:
            cats.append("")
        else:
            _df_rule = _df_rule.iloc[0]
            _cats = [c for c in _df_rule['cats'] if c not in exclude]
            cats.append(",".join(_cats))
    return SEP_STRING.join(cats)

def map_cat24_to_cat10(lst_cats, cat_mapping):
    if type(lst_cats) == str:
        lst_cats = lst_cats.split(SEP_STRING)
    final_cat = []
    for rule_num_cat in lst_cats:
        cats = rule_num_cat.split(",")
        cat10 = []
        for c in cats:
            if c == "":
                cat10.append("")
                continue
            cat10.append(cat_mapping[c])
        final_cat.append(",".join(list(set(cat10))))
    return SEP_STRING.join(final_cat)

def print_one_sample(df, target_cols=['subreddit','rule_texts','comment','cats'], target_cats=None):
    if target_cats:
        if type(target_cats) != list:
            target_cats = [target_cats]
        df = df[df.cat.apply(lambda x: bool(set(x) & set(target_cats)))]
    for name, value in df.sample(1)[target_cols].iloc[0].iteritems():
        print(f"{name}: {value}")

assert len(sys.argv) > 1, f"Usage: python get_comments_labeled.py NUM_SUBREDDIT"
num_subreddit = sys.argv[1]
num_max_comment = sys.argv[2]

RULE_NAME = f"rules_{num_subreddit}"
# path_rule = f"data/training/rules/{RULE_NAME}.json"
path_rule = f"data/combined/{RULE_NAME}.json"
path_rule_out = f"data/combined/{RULE_NAME}_labeled.tsv"
path_res = "res/rule-classifiers"
path_rules_classified = "res/categorized-rules/"
path_comments = f"data/combined/mod-comments_{num_max_comment}_{num_subreddit}_mapped.tsv"
path_mapping = "data/cat-mappings/cat24_to_cat10.json"

df_rules =import_jsonl(path_rule, as_panda=True)

cats = [d for d in os.listdir(path_res) if os.path.isdir(join(path_res,d))]
for cat in tqdm(cats):
    path_rule_prediction = join(path_rules_classified, cat, f"{RULE_NAME}.txt")
    if os.path.exists(path_rule_prediction):
        df_pred = import_tsv(path_rule_prediction)
        assert len(df_rules) == len(df_pred)
        df_rules[cat] = df_pred['prediction'].apply(lambda x: 0 if x == "neutral" else 1)


df_rules_final = convert_df_rules(df_rules)
df_rules_subreddit = {subreddit: sub_df for subreddit, sub_df in df_rules_final.groupby('subreddit')}

def is_empty_cats(cat_string):
    cat_string = cat_string.split(SEP_STRING)
    return all([c=="" for c in cat_string])

df_comments = import_tsv(path_comments)
tqdm.pandas()
df_comments['cat24'] = df_comments.apply(lambda row: map_rule_nums_text_to_cats(df_rules_subreddit.get(row['subreddit'], None),row['rule_nums']), axis=1)
df_comments = df_comments.dropna()
df_comments = df_comments[df_comments.cat24.apply(lambda x:not is_empty_cats(x))]

cat_mapping = import_json(path_mapping)
df_comments['cat'] = df_comments.cat24.apply(lambda x:map_cat24_to_cat10(x, cat_mapping))


df_rules_final = df_rules_final.rename(columns={"cats":"cat24"})
df_rules_final['cats'] = df_rules_final.cat24.apply(lambda x:list(set([cat_mapping[c] for c in x])))
df_rules_final.to_csv(path_rule_out, sep="\t")

path_save = path_comments.replace(".tsv","_labeled.tsv")
df_comments.to_csv(path_save, sep="\t")
print(f"labeled {len(df_comments)} comments saved to {path_save}")