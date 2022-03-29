import os
import sys
import re
import concurrent.futures
from tqdm import tqdm
from config import SEP_STRING, NUM_PROCESS
from utils import import_tsv, import_json

alpha_val = {chr(ord_num):ord_num-64 for ord_num in range(ord('A'), ord('Z')+1)}
rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

def convert_string_to_int(string):
    def roman_to_int(s):
        int_val = 0
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
            else:
                int_val += rom_val[s[i]]
        return int_val
    if string.isnumeric():
        return int(string)
    try:
        return roman_to_int(string)
    except:
        if string in alpha_val:
            return alpha_val(string)
        else:
            return -1

rule_pattern = re.compile(r"[Rr][Uu][Ll][Ee][ \-\+\_]+[#\*]*([\dA-Z]+)")
def find_rule_numbers(rule):
    rule_numbers = rule_pattern.findall(rule)
    if len(rule_numbers) == 0:
        return None
    rule_numbers = set(rule_numbers)
    rule_numbers = [convert_string_to_int(num) for num in rule_numbers]
    rule_numbers = [num for num in rule_numbers if num > 0] # exclude rule 0
    return rule_numbers

def find_rule_nums_from_rule_string(comment, df_subreddit_rules):
    rule_numbers = []
    comment = comment.lower()
    for idx, row in df_subreddit_rules.iterrows():
        rule, rule_idx, reason = row['rule'], row['idx'], row['violation_reason']
        try:
            if rule.lower() in comment:
                rule_numbers.append(rule_idx)
            elif reason.lower() in comment:
                rule_numbers.append(rule_idx)
        except:
            continue
    if rule_numbers == []:
        return None
    return rule_numbers

def cats_from_rule_numbers(rule_nums, df_subreddit_rules):
    cats = []
    for num in rule_nums:
        cat = df_subreddit_rules[df_subreddit_rules['idx']==num]['label'].to_string(index=False).strip()
        cats.append(cat)
    return cats
    

def match_rule_pattern(comment, df_rules_subreddit):
    bool_pattern, bool_string = False, False
    bool_labeled = 'label' in df_rules_subreddit.columns

    rule_numbers = find_rule_numbers(comment)
    cats = None
    if rule_numbers is not None:
        bool_pattern = True
    if rule_numbers is None:
        rule_numbers = find_rule_nums_from_rule_string(comment, df_rules_subreddit)
        if rule_numbers is not None:
            bool_string = True
    if rule_numbers is None:
        return None, None, False, False
    
    if bool_labeled:
        cats = cats_from_rule_numbers(rule_numbers, df_rules_subreddit)
    return rule_numbers, cats, bool_pattern, bool_string

def get_rule_text(df_rules_subreddit, rule_nums):
    rule_texts = []
    for rule_num in rule_nums:
        text = df_rules_subreddit[df_rules_subreddit.idx==rule_num].rule.to_string(index=False).strip()
        rule_texts.append(text)
    return rule_texts

def categorize_one_row():
    pass

def process_one_row(row, field, df_rules_by_subreddit):
    idx, row = row
    comment, subreddit = row[field], row['subreddit']
    df_rules_subreddit = df_rules_by_subreddit[subreddit]
    try:
        _rule_nums, _cats, bool_pattern, bool_string = match_rule_pattern(comment, df_rules_subreddit)
        num_error = 0
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        num_error = 1
        _rule_nums,_cats = None, None

    num_comment_by_pattern, num_comment_by_string = 0, 0
    if _rule_nums is not None and len(_rule_nums)>0:
        if bool_pattern:
            num_comment_by_pattern = 1
        elif bool_string:
            num_comment_by_string = 1
        _num_rule = len(_rule_nums)
        _rule_texts = SEP_STRING.join(get_rule_text(df_rules_subreddit, _rule_nums))
        _rule_nums = SEP_STRING.join([str(n) for n in _rule_nums])
    else:
        _num_rule = None
        _rule_texts = None
    return [_rule_nums, _cats, _rule_texts, _num_rule, num_error, num_comment_by_pattern, num_comment_by_string]    

def categorize_comments(df, df_rules, field="comment", rule_lexicon=None, use_mapping=False):
    bool_rules_labeled = 'label' in df_rules.columns

    subreddits = list(df['subreddit'].unique())
    df_rules_by_subreddit = {subreddit:df_rules[df_rules['subreddit'] == subreddit] for subreddit in subreddits}

    num_comment_by_pattern, num_comment_by_string, num_error = 0, 0, 0

    rule_nums, cats = [], []
    rule_texts, num_rules = [], []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
        data = list(tqdm(executor.map(
                lambda row: process_one_row(row, field, df_rules_by_subreddit),list(df.iterrows())), total=len(df)))

    rule_nums, cats, rule_texts, num_rules, num_error, num_comment_by_pattern, num_comment_by_string = zip(*data)
    num_error, num_comment_by_pattern, num_comment_by_string = sum(num_error), sum(num_comment_by_pattern), sum(num_comment_by_string)

    df['rule_nums'] = rule_nums
    df['rule_texts'] = rule_texts
    df['num_rules'] = num_rules

    if bool_rules_labeled:
        cats = [SEP_STRING.join(_cats) if _cats is not None else None for _cats in cats]
        df['label'] = cats

    print(f"{num_comment_by_pattern} ({num_comment_by_pattern/(len(df)-num_error)*100:.1f}%) comments matched by 'Rule X' pattern")
    print(f"{num_comment_by_string} ({num_comment_by_string/(len(df)-num_error)*100:.1f}%) comments matched by Rule string")
    df_unmatched = df[df.isna().any(axis=1)]
    df_matched = df.dropna()
    return df_matched, df_unmatched


assert len(sys.argv)>2, "Usage: python match_comments_to_rules.py NUM_SUBREDDIT MAX_MOD_COMMENT [PATH_LEXICON]"
num_subreddit = sys.argv[1]
num_max_comment = sys.argv[2]

path_comments = f"data/combined/mod-comments_{num_max_comment}_{num_subreddit}.tsv"
path_subreddit = f"data/subreddits/top{num_subreddit}.json"
if os.path.exists(f"data/combined/final_rules_{num_subreddit}.tsv"):
    path_rules = f"data/combined/final_rules_{num_subreddit}.tsv"
else:
    path_rules = f"data/combined/rules_{num_subreddit}.tsv"

df_comments = import_tsv(path_comments)
df_rules = import_tsv(path_rules)
list_subreddits = df_comments.subreddit.unique()
subreddit_info = import_json(path_subreddit)
subreddit_info = {k:v[0] for k,v in subreddit_info.items()}

print(f"# of subreddits: {len(list_subreddits)}")
print(f"# of comments: {len(df_comments)}")

df_comments_categorized, df_comments_uncategorized = categorize_comments(df_comments, df_rules, field="comment", rule_lexicon=None, use_mapping=False)
print(f"{len(df_comments_categorized)}/{len(df_comments)} ({len(df_comments_categorized)/len(df_comments)*100:.1f}%) comments are categorized")

path_save = path_comments.replace(".tsv","_mapped.tsv")
df_comments_categorized.to_csv(path_save,sep="\t")
print(f"rule mapped comments saved to {path_save}")
