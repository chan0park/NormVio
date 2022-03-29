import sys
import pandas as pd
import os
import json
import random
import concurrent.futures
from tqdm import tqdm
from scraper import RedditScraper
from multiprocessing import Pool
from config import SAVE_EVERY, NUM_PROCESS

import warnings
warnings.filterwarnings("ignore")


def import_subreddits(path):
    with open(path, "r") as file:
        subreddits = json.loads(file.read())
    return list(subreddits.keys())


def get_all_subreddits(subreddit_candidate_file):
    df = pd.read_csv(subreddit_candidate_file, sep='\t')
    return list(df['Subreddit'])


def get_mod_comments(subreddit, per_mod_comments_limit, verbose=False):
    if verbose:
        print(f"Getting mods of {subreddit}...", end="")
    mods = k.get_moderators(subreddit)
    if verbose:
        print(f"{len(mods)} mods found")
        print("Getting comments for mods...", end="")
    mod_comments = k.get_mod_comments(mods, subreddit, per_mod_comments_limit)

    if verbose:
        num_mod_comment = sum([len(thread) for thread in mod_comments])
        print(f"loaded {num_mod_comment} mod comments. Now processing.")
    final_mod_comments = []
    for one_mod_comments in mod_comments:
        one_mod_comments = [comment for comment in one_mod_comments if comment.parent_id.startswith("t1")]
        mod_comment_parent_ids = [comment.parent_id for comment in one_mod_comments]
        comments_parents = k.r.info(mod_comment_parent_ids)
        for comment, parent in zip(one_mod_comments, comments_parents):
            if parent.body == "[removed]":
                comment_text = str(comment.body)
                comment_text = " ".join(comment_text.split())
                try:
                    final_mod_comments.append(
                        (comment_text, comment.id, comment.author))
                except:
                    final_mod_comments.append(
                        (comment_text, comment.id, comment.author.name))
    mod_comments = final_mod_comments
    if verbose:
        print(f"{len(mod_comments)} comments found.")
        print(f"10 examples are: ")
        print("\n".join([str(t) for t in random.sample(
            mod_comments, min(10, len(mod_comments)))]))
    return mods, mod_comments


def write_mods(mods, outfile):
    mod_names = [mod.name for mod in mods]
    with open(outfile, 'w') as f:
        f.write("\n".join(mod_names))


def write_mod_comments(subreddit, mod_comments, outfile, write=True):
    rows = []
    for comment, comment_id, user in mod_comments:
        comment = comment.replace("\n", " ").replace("\t", " ")
        rows.append({"subreddit": subreddit, "moderator": user,
                     "comment": comment, "comment_id": comment_id})
    if write:
        if len(rows) > 0:
            df = pd.DataFrame(rows)
            df.to_csv(outfile, sep="\t", index=False)
    return rows


def process_one_subreddit(subreddit, path_save_dir, path_save_moderators, per_mod_comments_limit):
    outfile_comments = f'{path_save_dir}/{subreddit}.tsv'
    outfile_moderators = f'{path_save_moderators}/{subreddit}.txt'
    if os.path.exists(outfile_comments):
        with open(outfile_comments, "r") as file:
            lines = file.readlines()
        if len(lines) > 0:
            moderators, comments, comment_ids = [], [], []
            for line in lines:
                line = line.strip()
                try:
                    _, moderator, comment, comment_id = line.split("\t")
                    #Temporary fix
                    comment = comment.replace("\n", " ").replace("\t", " ")
                except KeyboardInterrupt:
                    sys.exit(0)
                except:
                    continue
                moderators.append(moderator)
                comments.append(comment)
                comment_ids.append(comment_id)
            mod_comments = list(zip(comments, comment_ids, moderators))
            rows = write_mod_comments(
                subreddit, mod_comments, outfile_comments, write=False)
    else:
        mods, mod_comments = get_mod_comments(
            subreddit, per_mod_comments_limit)
        write_mods(mods, outfile_moderators)
        rows = write_mod_comments(subreddit, mod_comments, outfile_comments)
    return rows


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


k = RedditScraper()


def main():
        # usage: python get_mod_comments.py NUM_SUBREDDIT MAX_COMMENT (OPTIONAL)REDDIT_NAME_FOR_TESTING
    path_save_comments = "data/moderator-comments"
    path_save_moderators = "data/moderators"
    if not os.path.isdir(path_save_comments):
        os.makedirs(path_save_comments)
    if not os.path.isdir(path_save_moderators):
        os.makedirs(path_save_moderators)

    assert len(sys.argv) >= 3
    num_subreddit = sys.argv[1]
    per_mod_comments_limit = int(sys.argv[2])
    path_subreddit_list = f"data/subreddits/top{num_subreddit}.json"
    assert os.path.exists(path_subreddit_list)

    subreddits = import_subreddits(path_subreddit_list)
    print(f"getting comments for the following {len(subreddits)} subreddits (showing top 10): ", ", ".join(
        subreddits[:10]))

    path_save_combined = f'data/combined/mod-comments_{per_mod_comments_limit}_{num_subreddit}.tsv'
    path_save_dir = f'{path_save_comments}/{per_mod_comments_limit}'
    if not os.path.isdir(path_save_dir):
        os.mkdir(path_save_dir)

    all_rows = []
    if NUM_PROCESS > 1:
        num_batch = int(len(subreddits)/SAVE_EVERY)
        pbar = tqdm(chunks(subreddits, SAVE_EVERY), total=num_batch)
        for batch in pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
                data = list(
                    tqdm(executor.map(
                        lambda subreddit: process_one_subreddit(subreddit, path_save_dir, path_save_moderators, per_mod_comments_limit),
                        batch), total=len(batch), leave=False))
            
            for rows in data:
                if len(rows) > 0:
                    all_rows += rows
            df = pd.DataFrame(all_rows)
            df.to_csv(path_save_combined, sep="\t", index=False)
    else:
        pbar = tqdm(subreddits, total=len(subreddits))
        for subreddit in pbar:
            pbar.set_description(subreddit)
            rows = process_one_subreddit(subreddit, path_save_dir, path_save_moderators, per_mod_comments_limit)
            if len(rows) > 0:
                all_rows += rows
            df = pd.DataFrame(all_rows)
            df.to_csv(path_save_combined, sep="\t", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(path_save_combined, sep="\t", index=False)
    print(f"scraped mod comments saved to {path_save_combined}")


if __name__ == "__main__":
    main()
