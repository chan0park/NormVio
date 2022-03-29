from multiprocessing import set_start_method
import concurrent.futures
import os
import sys
import argparse
import pandas as pd
import warnings
from tqdm import tqdm
from os.path import join, exists
from scraper import RedditScraper
from utils import save_df_to_jsonl
from config import NUM_PROCESS

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', "-data", type=str, required=True, help='redacted data path')
parser.add_argument('--path_save', "-save", type=str, required=False, default=None, help='where to save the restored data splits')
parser.add_argument('--verbose', "-v", action='store_true', default=False, help='verbose')
args = parser.parse_args()

if not args.path_save:
    if args.path_data.endswith("/"):
        args.path_data = args.path_data[:-1]
    args.path_save = args.path_data.replace("_redacted","")+"_restored"

assert args.path_data != args.path_save
assert all([exists(join(args.path_data, f"{split}.jsonl")) for split in ['train','test','dev']])
if not exists(args.path_save):
    os.makedirs(args.path_save)

if args.verbose:
    print(f"restoring data in {args.path_data} and saving to {args.path_save}")

def extract_comment_id_from_id(id):
    return id.split("_")[0].split("~")[0]

def restore_context(data):
    restored = []
    num_row, num_restored = 0, 0
    cids_by_subreddit = {}
    sids_by_subreddit = {}
    for _, row in data.iterrows():
        num_row += 1
        is_completely_restored = True
        subreddit = row['subreddit']
        if subreddit not in cids_by_subreddit:
            cids_by_subreddit[subreddit] = set()
            sids_by_subreddit[subreddit] = set()
        context = row['redacted_context']
        ids = [extract_comment_id_from_id(c['id']) for c in context]
        submission_ids = [c for c in ids if len(c) == 6]
        comment_ids = [c for c in ids if len(c)==7]
        sids_by_subreddit[subreddit].update(submission_ids)
        cids_by_subreddit[subreddit].update(comment_ids)
    
    def get_one_subreddit_comments(subreddit, ids, kind="comment"):
        try:
            if kind=="comment":
                return scraper.fetch_psaw_from_ids(ids, subreddit, kind="comment")
            elif kind == "submission":
                return {sid:scraper.fetch_psaw_from_id(sid, subreddit, kind="submission") for sid in ids}
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            return {}

    fetched = {'submission':{}, 'comment':{}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
            fetched_by_subreddit = list(tqdm(executor.map(
                    lambda subreddit: get_one_subreddit_comments(subreddit, cids_by_subreddit[subreddit], "comment"),
                    list(cids_by_subreddit.keys())), total=len(cids_by_subreddit)))
        for _fetched in fetched_by_subreddit:
            fetched['comment'].update(_fetched)

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
            fetched_by_subreddit = list(tqdm(executor.map(
                    lambda subreddit: get_one_subreddit_comments(subreddit, sids_by_subreddit[subreddit], "submission"),
                    list(sids_by_subreddit.keys())), total=len(sids_by_subreddit)))
        for _fetched in fetched_by_subreddit:
            fetched['submission'].update(_fetched)
    
    for _, row in data.iterrows():
        is_completely_restored=True
        context = row['redacted_context']
        ids = [extract_comment_id_from_id(c['id']) for c in context]
        for comment,id in zip(context, ids):
            kind = "comment" if len(id) == 7 else "submission"
            if id not in fetched[kind]:
                is_completely_restored = False
                break
            _comment = fetched[kind][id]
            if _comment is None:
                is_completely_restored = False
                break
            if kind == "comment":
                text = _comment.body
            else:
                try:
                    text = _comment.selftext
                    if text == "":
                        text = _comment.title
                except:
                    text = _comment.title
            if text in ('[removed]',['[deleted]']):
                is_completely_restored = False
                break
            comment['tokens'] = text
        if is_completely_restored is False:
            restored.append(None)
        elif len(context) == 1 and context[0]['tokens'] == "":
            restored.append(None)
        else:
            restored.append(context)
            num_restored += 1
    return restored

def restore_final(data):
    restored, cids_by_subreddit = [], {}
    num_row, num_restored = 0, 0
    for _, row in data.iterrows():
        subreddit = row['subreddit']
        if subreddit not in cids_by_subreddit:
            cids_by_subreddit[subreddit] = []
        final_comment = row['redacted_final_comment']
        cid = extract_comment_id_from_id(final_comment['id'])
        cids_by_subreddit[subreddit].append(cid)
    
    def get_one_subreddit_comments(subreddit, subreddit_cids):
        try:
            _fetched = scraper.fetch_psaw_from_ids(subreddit_cids, subreddit=subreddit, kind="comment")
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            return {}
        return _fetched
        
    fetched_comments = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
            fetched = list(tqdm(executor.map(
                    lambda subreddit: get_one_subreddit_comments(subreddit, cids_by_subreddit[subreddit]),
                    list(cids_by_subreddit.keys())), total=len(cids_by_subreddit)))
        for _fetched in fetched:
            fetched_comments.update(_fetched)

    for _, row in data.iterrows():
        cid = extract_comment_id_from_id(row['redacted_final_comment']['id'])
        if cid not in fetched_comments:
            restored.append(None)
            continue
        _comment = fetched_comments[cid]
        if _comment is None:
            restored.append(None)
            continue
        text = _comment.body
        if text in ('[removed]',['[deleted]']):
            restored.append(None)
            continue
        final_comment['tokens'] = text
        restored.append(final_comment)
    return restored    

scraper = RedditScraper()
for split in ['test','dev','train']:
    data = pd.read_json(join(args.path_data, f"{split}.jsonl"),lines=True)
    assert all([('redacted_'+colname in data.columns) for colname in ['context','final_comment']])
    # data = data.sample(20)
    original_size = len(data)
    if args.verbose:
        print(f"[{split}] restoring final comments")
    data['final_comment'] = restore_final(data)
    data = data.dropna(subset=['final_comment'])
    
    if args.verbose:
        print(f"[{split}] restoring context comments")
    data['context'] = restore_context(data)
    # data = data.drospna(subset=['context']) # uncomment this line to remove examples without context
    
    del data['redacted_context']
    del data['redacted_final_comment']
    
    path_save_split = join(args.path_save, f"{split}.jsonl")
    save_df_to_jsonl(data, path_save_split)
    
    final_size = len(data)
    if args.verbose:
        print(f"{final_size} out of {original_size} conversations were restored and saved to {path_save_split}")