import sys
import praw
import pandas as pd
import os
import json
import random
import warnings
import concurrent.futures
from scraper import RedditScraper
from tqdm import tqdm
from config import NUM_PROCESS
warnings.filterwarnings("ignore")

def import_subreddits(path):
    with open(path,"r") as file:
        subreddits = json.loads(file.read())
    return list(subreddits.keys())

def write_rules(subreddit, rules, path):
    rows = []
    for idx, rule in enumerate(rules):
        rows.append({"subreddit": subreddit,"idx":idx+1, "rule": rule.short_name.replace("\n"," "), "description": rule.description.replace("\n"," "), "kind": rule.kind.replace("\n"," "), "violation_reason":rule.violation_reason.replace("\n"," ")})
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    return rows

def get_one_subreddit_rule(scraper, subreddit, path_save_rules):
    outfile = path_save_rules+f'{subreddit}.tsv'
    rules = scraper.get_community_rules(subreddit)
    if rules:
        rules = write_rules(subreddit, rules, outfile)        
    return rules

def main():
    path_save_combined = "data/combined/"
    path_save_rules = "data/community-rules/"
    for path in [path_save_combined, path_save_rules]:
        if not os.path.isdir(path):
            os.makedirs(path)

    assert len(sys.argv) >= 2
    max_rank = sys.argv[1]
    path_subreddit_list = f"data/subreddits/top{max_rank}.json"
    assert os.path.exists(path_subreddit_list)

    if len(sys.argv) > 2:
        subreddits = sys.argv[2:]
        print("using subreddits given as input arguments")
    else:
        print(f"importing ruls from {path_subreddit_list}")
        subreddits = import_subreddits(path_subreddit_list)
    print(f"getting rules for {len(subreddits)} subreddits")

    scraper = RedditScraper()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
        rule_rows = list(
            tqdm(executor.map(
                lambda subreddit: get_one_subreddit_rule(scraper, subreddit, path_save_rules),
                subreddits), total=len(subreddits), leave=True))
        
    all_rules = [r for rule in rule_rows if rule is not None for r in rule if rule_rows is not None]

    df = pd.DataFrame(all_rules)
    df.to_csv(path_save_combined+f'rules_{max_rank}.tsv', sep="\t", index=False)

    df_json = df[['violation_reason','subreddit','idx']]
    df_json.columns = ['sentence','subreddit','idx']
    df_json['labels'] = "neutral"
    df_json.to_json(path_save_combined+f'rules_{max_rank}.json',orient="records", lines=True)

if __name__ == "__main__":
    main()
