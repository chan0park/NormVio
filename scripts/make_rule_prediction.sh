NUM_SUBREDDIT=$1
PATH_RULE="data/combined/rules_$NUM_SUBREDDIT.json"
PATH_SAVE_PREDICTION="res/categorized-rules"
TRAINING_DATA_PATH="data/training/rule-classification"
PATH_MODEL="res/rule-classifiers"
cats="$TRAINING_DATA_PATH"/*

for cat in $cats
do
    # cat="res/rule-classifier/Trolling"
    catname=$(basename "$cat")
    mkdir -p "$PATH_SAVE_PREDICTION"/"$catname"
    echo ["$catname"] making predictions of "$PATH_RULE"

    python src/train_rule_classifier.py --model_name_or_path "$PATH_MODEL"/"$catname" \
    --train_file "$TRAINING_DATA_PATH"/"$catname"/train_dev.json \
    --test_file $PATH_RULE --do_predict \
    --label_names labels \
    --per_device_eval_batch_size 32 \
    --output_dir "$PATH_SAVE_PREDICTION"/"$catname" \
    --max_seq_length 128 

    rm "$PATH_SAVE_PREDICTION"/"$catname"/*.json
done