import concurrent.futures
import os
import sys
import random
import pandas as pd
import json
import warnings
from os.path import join
from copy import deepcopy
from math import ceil
from itertools import islice, chain
from convokit import Speaker, Utterance, Corpus
from praw.models import Submission
from tqdm import tqdm
from shortuuid import uuid as new_id
from scraper import RedditScraper
from config import NUM_PROCESS, MAX_DEPTH

speakers = {}
modded_utterances_dic = {}
modded_submissions = {}
unmodded_utterances_dic = {}
conversations_to_remove = set()
unique_mod_utterances = set()

def utterance_from_comment(comment, has_attack, mod, extra_meta=None, unique_id='default'):
    utt_speaker = get_speaker(comment)
    if isinstance(comment, Submission):
        utt_id = comment.id + "~" + unique_id + ("_mod" if mod else "_clean")
        utt_root = comment.id + "~" + unique_id
        utt_reply_to = None
    else:
        utt_id = comment.id + "~" + unique_id
        utt_root = comment.link_id.split("_")[1] + "~" + unique_id
        utt_reply_to = comment.parent_id.split("_")[1] + "~" + unique_id

    if utt_reply_to == utt_root:
        utt_reply_to = utt_reply_to + ("_mod" if mod else "_clean")
    utt_root = utt_root + ("_mod" if mod else "_clean")
    utt_timestamp = comment.created_utc
    utt_is_header = utt_reply_to is None
    utt_text = comment.title if utt_is_header else comment.body
    subreddit = comment.subreddit.display_name if not isinstance(comment.subreddit, str) else comment.subreddit
    meta = {'is_section_header': utt_is_header, 'comment_has_personal_attack': has_attack, 'toxicity': has_attack,
            'subreddit': subreddit}
    if extra_meta:
        meta.update(extra_meta)
    utt_obj = Utterance(id=utt_id, root=utt_root, reply_to=utt_reply_to, timestamp=utt_timestamp,
                        speaker=utt_speaker, text=utt_text, user=utt_speaker,
                        meta=meta, conversation_id=utt_root)
    return utt_obj


def get_speaker(data):
    global speakers
    content = data.body if not isinstance(data, Submission) else data.title
    speaker_name = (data.author if isinstance(data.author, str) else data.author.name) if content not in (
        '[deleted]', '[removed]') and data.author else '[deleted]'
    speaker_id = speaker_name if speaker_name != '[deleted]' else new_id()
    if speaker_id not in speakers:
        speakers[speaker_id] = Speaker(id=speaker_id, name=speaker_name, meta={'name': speaker_name})
    utt_speaker = speakers[speaker_id]
    return utt_speaker

def get_all_parents(comment):
    comment_id = comment.id
    parent_ids = set([comment_id])
    parent = comment.parent()
    while not isinstance(parent, Submission):
        parent_ids.add(parent.id)
        parent = parent.parent()
    return parent_ids


def batch(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        try:
            yield chain([batchiter.__next__()], batchiter)
        except StopIteration:
            return []


class ThreadReconstructor:
    def __init__(self, path_out, num_max_comment=500, save_every=200, max_num_unmod=None):
        self.path_out_root = path_out
        self.path_out = {}
        self.corpus = {}
        self.meta = {}
        self.comment2convid = {}
        self.scraper = RedditScraper()
        self.traverser = CommentTraverser()
        
        self.num_max_comment = num_max_comment
        self.save_every = save_every
        self.max_num_unmod = max_num_unmod
        self.num_processed = 0
        self.num_restored = 0
        self.num_unmod_processed = 0
        self.num_unmod_retrieved = 0

        for type in ['mod','unmod']:
            self.path_out[type] = join(self.path_out_root, type)
            if not os.path.isdir(self.path_out[type]):
                os.mkdir(self.path_out[type])
            self.load_or_initialize_corpus(type)

    def load_comment2convid(self, meta, type):
        comment2convid = {}
        for conv_id, conv_meta in meta.items():
            comment_id = conv_meta['moderation_id']
            if type == "mod":
                comment2convid[comment_id] = conv_id
            elif type == "unmod":
                if not comment_id in comment2convid:
                    comment2convid[comment_id] = []
                comment2convid[comment_id].append(conv_id)
        return comment2convid
    
    def load_or_initialize_corpus(self, type="mod"):
        if os.path.exists(join(self.path_out[type], "utterances.jsonl")):
            self.corpus[type] = Corpus(self.path_out[type])
            with open(join(self.path_out[type], "convo_meta.json"),"r") as file:
                self.meta[type] = json.loads(file.read())
            self.comment2convid[type] = self.load_comment2convid(self.meta[type], type)
        else:
            self.corpus[type] = Corpus(utterances=[])
            self.meta[type] = {}
            self.comment2convid[type] = {}

    def check_exists(self, path):
        for filename in ["utterances.jsonl","convo_meta.json","processed_comment_ids.txt"]:
            if not os.path.exists(join(path, filename)):
                return False
        return True

    def get_already_reconstructed_comment_ids(self, path):
        with open(join(path, "processed_comment_ids.txt"),"r") as file:
            mod_comment_ids = [line.strip() for line in file if line.strip() != ""]
        return set(mod_comment_ids)

    def process_one_mod_thread(self, comment_data, subreddit, extra_meta=None, fetched_data=None):
        removed_comment = comment_data.parent()
        id_removed = removed_comment.id
        removed_comment_data = {'id': id_removed, 'meta': comment_data}
        if isinstance(removed_comment, Submission):
            return None
        modded_utterances_dic = {}
        
        if comment_data.id in fetched_data:
            _comment = fetched_data[comment_data.id]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _comment = self.scraper.fetch_psaw_from_id(id_removed, subreddit)
        if _comment is None:
            return None

        bool_comment_restored = False
        well_fetched_deleted_comment = None
        if not _comment.body in ('[removed]', '[deleted]'):
            bool_comment_restored = True
            well_fetched_deleted_comment = _comment
        
        thread_comment = removed_comment.parent()
        thread_data = [removed_comment_data]
        extra_meta['is_restored'] = bool_comment_restored
        utt = utterance_from_comment(well_fetched_deleted_comment if bool_comment_restored else removed_comment, 1, True, unique_id=id_removed, extra_meta=extra_meta)
        modded_utterances_dic[id_removed + "~" + id_removed] = utt
        
        utts_to_add = []
        while not isinstance(thread_comment, Submission):
            if thread_comment.body == '[removed]' or thread_comment.body == '[deleted]':
                # Removed for some random reason in same thread, so we chuck this thread
                return None
            thread_data.append({'id': thread_comment.id})
            utt_obj = utterance_from_comment(thread_comment, 0, mod=True, unique_id=id_removed)
            thread_comment = thread_comment.parent()
            utts_to_add.append(utt_obj)

        if len(utts_to_add) > MAX_DEPTH:
            return None
        for utt_obj in utts_to_add:
            modded_utterances_dic[utt_obj.id] = utt_obj

        # Last one is submission
        submission = thread_comment
        submission_id = submission.id + "~" + id_removed + "_mod"
        utt_obj = utterance_from_comment(submission, has_attack=0, mod=True, unique_id=id_removed)
        modded_utterances_dic[submission_id] = utt_obj
        return (id_removed, thread_data, modded_utterances_dic, bool_comment_restored)

        
    def fetch_one_unmod_thread(self, comment_id):
        moded_comment = self.get_comment_data_from_id(comment_id)
        unmoded_conversations = self.traverser.get_unmoded(moded_comment, comment_id)

        row_infos = {}
        for conv in unmoded_conversations:
            if len(conv) < 2:
                continue
            try:
                thread_ids = [{'id':u.id} for u in conv]
                thread_ids_str = "-".join([u.id for u in conv])
            except:
                print(comment_id)
                print(conv)
                raise
            last_comment_id = thread_ids[-1]['id']
            utterances_dic = {u.id:utterance_from_comment(u, has_attack=0, mod=False, unique_id=last_comment_id) for u in conv}
            row_infos[thread_ids_str] = {"thread_ids": thread_ids, "uttr": utterances_dic, "mod_comment_id":comment_id}
            # row_infos.append(row_info)
        row_infos = list(row_infos.values())
        if len(row_infos) == 0:
            return None, comment_id
        if self.max_num_unmod:
            if len(row_infos) > self.max_num_unmod:
                moded_conv = self.corpus['mod'].get_conversation(self.comment2convid['mod'][comment_id])
                moded_num_uttr = len(moded_conv.get_chronological_utterance_list())
                row_infos = self.choose_unmod(row_infos, moded_num_uttr, self.max_num_unmod)
        return row_infos, comment_id

    def choose_unmod(self, unmoded, num_mod_uttr, num_unmod):
        conv_lens = [(unmod, len(unmod['uttr'])) for unmod in unmoded]
        sorted_conv = sorted(conv_lens, key = lambda t: abs(t[1]-num_mod_uttr))
        unmoded, _ = zip(*sorted_conv)
        return unmoded[:num_unmod]

    def get_comment_data_from_id(self, comment_id):
        return self.scraper.r.comment(comment_id)

    def fetch_one_mod_thread(self, inp, fetched_removed_comments):
        row = inp[1]
        subreddit, mod_name, comment_text, comment_id = row['subreddit'], row['moderator'], row['comment'], row['comment_id']
        label = row['label'] if 'label' in row else None
        row_info = {"subreddit":subreddit, "moderator":mod_name, "mod_comment":comment_text, "rule_nums":row["rule_nums"], "label":label, "rule_texts": row["rule_texts"],"mod_comment_id": comment_id}
        comment = self.get_comment_data_from_id(comment_id)
        
        res = self.process_one_mod_thread(comment, subreddit, row_info, fetched_removed_comments)
        if res is None:
            return None, comment_id

        removed_id, thread_ids, uttr, bool_comment_restored = res
        row_info.update({"removed_comment_id":removed_id, "thread_ids": thread_ids, "uttr": uttr, 'bool_comment_restored': bool_comment_restored})
        return row_info, comment_id

    def reconstruct_neg_examples(self):
        to_be_processed = list(self.comment2convid['mod'].keys())
        if self.check_exists(self.path_out["unmod"]):
            existing_ids = self.get_already_reconstructed_comment_ids(self.path_out["unmod"])
            len_original = deepcopy(len(to_be_processed))
            to_be_processed = [id for id in to_be_processed if id not in existing_ids]
            print(f"found {len(existing_ids)} comments saved already. Processing ({len_original}->) {len(to_be_processed)} comments...")
        if len(to_be_processed) == 0:
            return

        num_chunk = ceil(len(to_be_processed)/self.save_every)
        pbar = tqdm(batch(to_be_processed, self.save_every), total=num_chunk)
        for chunk in pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
                data = list(
                    tqdm(executor.map(
                        lambda comment_data: self.fetch_one_unmod_thread(comment_data),
                        chunk), total=self.save_every, leave=False))

            data, processed_ids = zip(*data)
            data = [d for d in data if d is not None]
            data = [conv for d in data for conv in d]
            self.save_data(data, processed_ids, "unmod")
            
            self.num_unmod_retrieved += len(data)
            self.num_unmod_processed += len(processed_ids)
            pbar.set_description(f"{self.num_unmod_retrieved}/{self.num_unmod_processed}({int(self.num_unmod_retrieved/self.num_unmod_processed*100)}%)")


    def reconstruct(self, mapped_comments):
        self.reconstruct_from_comments(mapped_comments)
        self.save_data(type="mod")
        self.reconstruct_neg_examples()
        self.save_data(type="unmod")

    def fetch_removed_comments(self, comments):                
        subreddit, subreddit_comments = comments
        mod_comment_ids = subreddit_comments['comment_id'].tolist()
        removed_comments = {id:self.get_comment_data_from_id(id).parent() for id in mod_comment_ids}
        removed_comments_sub = {id:v for id,v in removed_comments.items() if len(id) != 7}
        removed_comments_not_sub = {r.id:id for id,r in removed_comments.items() if len(id) ==7}
        removed_comments_ids = [r for r in removed_comments_not_sub.keys()]
        
        
        res = {k:None for k in removed_comments_sub}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fetched_removed_comments = self.scraper.fetch_psaw_from_ids(removed_comments_ids, subreddit=subreddit, kind="comment")
        fetched_to_mod_comment = {removed_comments_not_sub[k]:v for k,v in fetched_removed_comments.items()}
        res.update(fetched_to_mod_comment)
        return res
        

    def reconstruct_from_comments(self, comments):
        if self.check_exists(self.path_out["mod"]):
            existing_ids = self.get_already_reconstructed_comment_ids(self.path_out["mod"])
            len_original = deepcopy(len(comments))
            comments = comments[~comments.comment_id.isin(existing_ids)]
            print(f"found {len(existing_ids)} comments saved already. Processing ({len_original}->) {len(comments)} comments...")
        if len(comments) == 0:
            return
        
        # fetch comments from psaw first
        comments_by_subreddit = comments.groupby(['subreddit'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
            fetched_comments = list(
                tqdm(executor.map(
                    lambda comment_data: self.fetch_removed_comments(comment_data),
                    comments_by_subreddit), total=len(comments_by_subreddit), leave=False))
        fetched_removed_comments = {id:comment for subreddit_comments in fetched_comments for id, comment in subreddit_comments.items()}
        
        num_chunk = ceil(len(comments)/self.save_every)
        pbar = tqdm(batch(comments.iterrows(), self.save_every), total=num_chunk)
        for chunk in pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
                data = list(
                    tqdm(executor.map(
                        lambda comment_data: self.fetch_one_mod_thread(comment_data, fetched_removed_comments),
                        chunk), total=self.save_every, leave=False))
            
            _processed_id = [comment_id for d, comment_id in data]
            self.num_processed += len(_processed_id)

            data = [d for d, _ in data if d is not None]
            self.num_restored += len(data)

            pbar.set_description(f"{self.num_restored}/{self.num_processed}({int(self.num_restored/self.num_processed*100)}%)")
            self.save_data(data, _processed_id, "mod")
    
    def update_corpus(self, data, type):
        data_utterances = [u for x in data for u in x['uttr'].values()]
        self.corpus[type] = self.corpus[type].add_utterances(data_utterances)
        for thread in data:
            conv_id = list(thread['uttr'].values())[0].conversation_id
            # assert conv_id not in self.meta, f"{conv_id}, {self.meta[conv_id]}"
            attack = 1 if conv_id.split("_")[1] == "mod" or type=="mod" else 0
            mod_comment_id = thread['mod_comment_id']
            _meta = {'split': 'reddit', 'annotation_year': 2021, 'conversation_has_personal_attack': attack, 'moderation_id':mod_comment_id}
            
            if 'bool_comment_restored' in thread:
                _meta['is_restored'] = thread['bool_comment_restored']
            
            self.meta[type][conv_id] = _meta
            if type == "mod":
                self.comment2convid[type][mod_comment_id] = conv_id
            elif type =="unmod":
                if mod_comment_id not in self.comment2convid[type]:
                    self.comment2convid[type][mod_comment_id] = []
                self.comment2convid[type][mod_comment_id].append(conv_id)
            conv = self.corpus[type].get_conversation(conv_id)
            conv.meta.update(_meta)
    
    def save_processed_id(self, ids, type):
        with open(join(self.path_out[type], "processed_comment_ids.txt"),"a") as file:
            file.write("\n".join(ids)+"\n")
    
    def save_data(self, data=None, processed_id=None, type="mod"):
        if data:
            self.update_corpus(data, type)
        assert len(self.meta[type]) == len(list(self.corpus[type].get_conversation_ids()))
        self.corpus[type].dump(self.path_out[type], base_path="./", force_version=1)
        with open(join(self.path_out[type],"convo_meta.json"), 'w') as f:
            json.dump(self.meta[type], f)
        if processed_id:
            self.save_processed_id(processed_id, type)

class CommentTraverser():
    def __init__(self, max_depth=MAX_DEPTH):
        self.unmoded_conversations = {}
        self.max_depth = max_depth

    def traverse(self, moded_comment, moded_comment_id):
        try:
            submission = moded_comment.submission
        except:
            self.unmoded_conversations[moded_comment_id] = []
            return
        parent_ids = get_all_parents(moded_comment)

        removed_content = set(['[removed]','[deleted]'])
        level_comments = []
        for level_comment in submission.comments:
            if level_comment.id in parent_ids:
                continue
            try:
                body = level_comment.body
            except:
                continue
            if body in removed_content:
                continue
            level_comments.append(level_comment)
        self.unmoded_conversations[moded_comment_id] = []
        for first_level_comment in level_comments:
            self.travese_recursively(first_level_comment, moded_comment_id, [submission])
    
    def travese_recursively(self, comment, moded_comment_id, comment_history=[], depth=1, max_num=3):
        if depth > self.max_depth:
            self.unmoded_conversations[moded_comment_id].append(comment_history)
            return 
             
        sub_comments = []
        for sub_comment in comment.replies:
            try:
                body = sub_comment.body 
            except:
                continue
            sub_comments.append(sub_comment)

        if len(sub_comments) == 0:
            self.unmoded_conversations[moded_comment_id].append(comment_history)
            return

        sub_comments = random.sample(sub_comments, min(max_num, len(sub_comments)))
        for sub_comment in sub_comments:
            new_history = comment_history + [sub_comment]
            self.travese_recursively(sub_comment, moded_comment_id, new_history, depth+1)
    
    def get_unmoded(self, moded_comment, moded_comment_id):
        self.traverse(moded_comment, moded_comment_id)
        return self.unmoded_conversations[moded_comment_id]


if __name__ == "__main__":
    num_subreddit = sys.argv[1]
    num_max_comment = sys.argv[2]

    path_mapped_comments = f"data/combined/mod-comments_{num_max_comment}_{num_subreddit}_mapped.tsv"
    mapped_comments = pd.read_csv(path_mapped_comments, sep="\t")
    path_out = f"data/conversations/max{num_max_comment}_subs{num_subreddit}/"
    if len(sys.argv) == 5:
        start, end = int(sys.argv[3]), int(sys.argv[4])
        mapped_comments = mapped_comments.iloc[start:end]
        path_out = f"data/conversations/max{num_max_comment}_subs{num_subreddit}_{start}_{end}/"
    
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    print(f"saving fetched conversations to {path_out}...")

    scraper = ThreadReconstructor(path_out, num_max_comment, save_every=150, max_num_unmod=2)
    scraper.reconstruct(mapped_comments)
