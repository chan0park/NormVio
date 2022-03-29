TRAINING_DATA_PATH="data/training/rule-classification"
PATH_SAVE_MODEL="res/rule-classifier"
PATH_SAVE_LOG="logs/rule-classifier"
cats="$TRAINING_DATA_PATH"/*

mkdir -p $PATH_SAVE_MODEL
mkdir -p $PATH_SAVE_LOG

for cat in $cats
do
    catname=$(basename "$cat")
    path_log="$PATH_SAVE_LOG"/bert_"$catname".txt
    echo training "$catname"
    python src/train_rule_classifier.py --model_name_or_path bert-base-cased \
    --train_file "$cat"/train_dev.json \
    --gradient_accumulation_steps 2  --save_total_limit 1 \
    --validation_file "$cat"/test.json \
    --logging_strategy steps --logging_steps 20 \
    --metric_for_best_model eval_f1 --load_best_model_at_end --evaluation_strategy steps --eval_steps 20 \
    --label_names labels \
    --do_train --do_eval \
    --max_seq_length 128  \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 20 \
    --output_dir "$PATH_SAVE_MODEL"/"$catname" \
    --overwrite_output_dir | tee $path_log
done