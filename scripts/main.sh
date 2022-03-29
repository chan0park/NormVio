#!/bin/bash
NUM_SUBREDDIT="500"  # number of subreddit you scrape data from
MAX_COMMENT_PER_MOD="500"  # number of the most recent comment you collect for each moderator

# --------------------------[data preparation]----------------------------------------
# Constructing NormVio (fetching moderated conversations)

# (1) get a list of subreddits (sorted by number of subscribers)
python src/get_top_subreddits.py "$NUM_SUBREDDIT"

# (2) scrape rules of subreddits in the list from (1)
python src/get_subreddit_rules.py "$NUM_SUBREDDIT"

# (3) scrape moderator comments
if [[ $OSTYPE == 'darwin'* ]]; then
    # Mac OS: single process
    python src/get_mod_comments.py "$NUM_SUBREDDIT" "$MAX_COMMENT_PER_MOD" 1
else
    python src/get_mod_comments.py "$NUM_SUBREDDIT" "$MAX_COMMENT_PER_MOD"
fi

# (4) map comments to rule number & rule classes
python src/match_comments_to_rules.py "$NUM_SUBREDDIT" "$MAX_COMMENT_PER_MOD"

# (5) fetch the entire conversation thread + context
python src/get_thread_and_deleted_comment.py "$NUM_SUBREDDIT" "$MAX_COMMENT_PER_MOD"


# -------------------------[prepare training data for norm violation detection models]-----------------------------------------
# If you only need moderated dataset, you can skip step 6,7, and 8.
# step 6-8 outputs violation categories of moderator comments, and are needed if you want to train norm violation detection models

# (6) train a bert classifier (input: rule --> output: violation category)
# instead of training by yourself, you can download the trained models from https://drive.google.com/file/d/1gY904yZFbNQyLDw1M8XkhaSNr_s96VOV and locate them under res/rule-classifier/
if [ -d "res/rule-classifier/" ] 
then
    python scripts/train_bert_rule_classifier.sh
fi

# (7) using trained rule classifier, first categorize the rules and then use rule-comment mapping for mod comments
# _categorized_labeled
bash scripts/make_rule_prediction.sh "$NUM_SUBREDDIT"
python src/get_comments_labeled.py "$NUM_SUBREDDIT" "$MAX_COMMENT_PER_MOD"

# (8) prepare train/test/val data split. 
NAME=max"$MAX_COMMENT_PER_MOD"_subs"$NUM_SUBREDDIT"

python src/prepare_data.py -v --path_conversation data/conversations/"$NAME"/ \
    --path_comment data/combined/mod-comments_"$MAX_COMMENT_PER_MOD"_"$NUM_SUBREDDIT"_mapped_labeled.tsv \
    -save data/processed/"$NAME"/ 

# -----------------------[model training]-------------------------------------------
# (9) train norm violation detection models
# mkdir -p logs/violation-detection

# python src/train_derailment_model_binary.py -conv data/processed/"$NAME"/ \
#     -rule data/combined/rules_"$NUM_SUBREDDIT"_labeled.tsv \
#     -save saved_models/binary | tee logs/violation-detection/binary.txt

# python src/train_derailment_model_binary.py -conv data/processed/"$NAME"/ \
#     -rule data/combined/rules_"$NUM_SUBREDDIT"_labeled.tsv -sub subreddit \
#     -save saved_models/binary_subreddit | tee logs/violation-detection/binary_subreddit.txt

# python src/train_derailment_model_multi.py -conv data/processed/"$NAME"/ \
#     -rule data/combined/rules_"$NUM_SUBREDDIT"_labeled.tsv -multi \
#     -save saved_models/multi | tee logs/violation-detection/multi.txt

# python src/train_derailment_model_multi.py -conv data/processed/max500_subs100000/ \
#     -rule data/combined/rules_"$NUM_SUBREDDIT"_labeled.tsv -sub subreddit -multi \
#     -save saved_models/multi_subreddit | tee "$BASE"/logs/violation-detection/multi_subreddit.txt
