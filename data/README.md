# Directory Structure

The results of data collection are saved here in `data/`. You should be able to see two directories:

- `cat-mapping`: While (Fiesler et al., 2018) categorized Reddit community rules into 24 groups, we remapped them into slightly more coarse 10 groups, which are described in our paper. The `cat24_to_cat10.json` file in this directory shows the mapping we used.
- `processed`: This is where the processed datasets are placed. Such processed data can be used to train and run our norm violation detection models. We provide a redacted version of our original dataset as `max500_subs100000_redacted`, which is a set of norm violation conversations collected from the top 100000 subreddits. *max500* indicates that we discovered such conversations by looking at the (max) 500 past comments left by community moderators. You can run the reconstruction code in the main README file to generate `max500_subs100000`.

If you run  `scripts/main.sh` and collect a dataset from scratch, it will generate new directories such as:

- `subreddits`: stores the list of top X subreddits.
- `community-rules`: stores the list of community rules.
- `moderators`: for each subreddit, stores the list of Reddit handles of its moderators
- `moderator-comments`: for each subreddit, using the list of moderators, collect their past comments in the subreddit.
- `conversations`: stores collected conversations (original submission+past comments+final comment). There are two subdirectories: 1) *mod*, which stores the conversations that were moderated for violating rules, and 2) *unmod*,  which stores the conversations that were *not* moderated but randomly selected from the same posts in *mod* (serve as negative examples during model training).

# Processed Data Format

Under `processed/max500_subs500_redacted/` you will have three different files: `train.jsonl`, `dev.jsonl`, `test.jsonl`.
Each line in each file is a JSON object that contains information about one conversation. There are several fields in each object:

- `comment_id`: the id of the moderator's comment.
- `conv_id`: the id of conversation. `_mod` indicates that the conversation has derailed (i.e. violated community rules), and `_clean` suggests that the conversation was not moderated.
- `subreddit`: name of the subreddit the conversation was posted.
- `bool_derail`: whether the conversation derailed or not. (same as `_mod` vs. `_clean`)
- `rule_texts`: the description of community rule `final_comment` violated. For unmoderated conversations, this is a randomly chosen community rule.
- `cats`: the rule category of `rule_texts`.
- `is_restored`: whether the final comment was restored or not.

The following fields will be restored when you run the reconstruction code:

- `final_comment`: the text of the final comment (which actually violated one of the community rules).
- `context`: the original Reddit post and the previous comments posted in prior to `final_comment`.
- (optional) `mod_comment`: when you reconstruct the data, you can set `--mod_original_comment` to also scrape the original comment left by moderators at the time of moderation. Note that this field was not used in the paper but can be useful for developing generation models.
