import os
import sys
import json
import random
import argparse
import unicodedata
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import sigmoid
from tqdm import tqdm
from time import time
from datetime import datetime
from torch import optim
from os.path import join, exists
from multiprocessing import Pool
from convokit import Corpus
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
from config import BERT_TYPE, BERT_CACHE, HIDDEN_SIZE, MAX_LENGTH, ENC_NUM_LAYER, DROPOUT_PROB, PAD_token, SOS_token, EOS_token, UNK_token
from config import BATCH_SIZE, VALID_BATCH_SIZE, CLIP, TF_RATIO, LR, DEC_LR, EPOCHS, EARLY_STOPPING
from models import EncoderBERT, ContextEncoderRNN, SingleTargetClf, Predictor
from utils import import_jsonl, import_json, import_tsv, save_json

NUM_PROCESS = 22
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def encode(text, max_length=MAX_LENGTH):
    # simplify the problem space by considering only ASCII data
    cleaned_text = unicodeToAscii(text)

    # if the resulting string is empty, nothing else to do
    if not cleaned_text.strip():
        return []

    return tokenizer.encode(cleaned_text, add_special_tokens=True, truncation=True, max_length=max_length)


# Given a ConvoKit conversation, preprocess each utterance's text by tokenizing and truncating.
# Returns the processed dialog entry where text has been replaced with a list of
# tokens, each no longer than MAX_LENGTH - 1 (to leave space for the EOS token)
def processDialog(dialog):
    processed = []
    for utterance in dialog.get_chronological_utterance_list():
        # skip the section header, which does not contain conversational content
        if utterance.meta['is_section_header']:
            continue
        tokens = utterance.text + " | "
        processed.append(
            {"tokens": tokens, "is_attack": int(utterance.meta['comment_has_personal_attack']), "id": utterance.id})

    return processed


# Load context-reply pairs from the Corpus, optionally filtering to only conversations
# from the specified split (train, val, or test).
# Each conversation, which has N comments (not including the section header) will
# get converted into N-1 comment-reply pairs, one pair for each reply
# (the first comment does not reply to anything).
# Each comment-reply pair is a tuple consisting of the conversational context
# (that is, all comments prior to the reply), the reply itself, the label (that
# is, whether the reply contained a derailment event), and the comment ID of the
# reply (for later use in re-joining with the ConvoKit corpus).
# The function returns a list of such pairs.

def get_wrong_rule_idx(correct_rule_idx, num_rule, num_example=1):
    # total = len(rule_config)
    possible_rule_idxs = [i for i in range(num_rule) if i not in correct_rule_idx]
    selected = random.sample(possible_rule_idxs, k=num_example)
    if num_example == 1:
        return selected[0]
    return selected

subreddit_types={}
with open("data/subreddits/top100000_nsfw.json","r") as file:
    subreddit_types['nsfw'] = set([k for k in json.load(file)])
with open("data/subreddits/top100000_sfw.json","r") as file:
    subreddit_types['sfw'] = set([k for k in json.load(file)])

def get_subreddit_type(subreddit_name):
    if subreddit_name in subreddit_types['nsfw']:
        return 'nsfw'
    elif subreddit_name in subreddit_types['sfw']:
        return 'sfw'
    else:
        print(f"we will assume sfw as we don't have nsfw/sfw info for r/{subreddit_name}")
        return 'sfw'
    return True

def get_input_text(tokens, rule_text, subreddit_name, append_rule, append_subreddit):
    input_text = tokens
    if append_rule or (append_subreddit is not None):
        input_text += " [SEP]"
    if subreddit_name == "subreddit":
        input_text += f" r/{subreddit_name}"
    elif subreddit_name == "nsfw":
        input_text += " r/" + get_subreddit_type(subreddit_name)
    if append_rule:
        input_text += " "+rule_text
    return input_text

def get_wrong_rule_txt(df_rules, subreddit, wrong_rule_idx):
    pass

def get_wrong_rules(cat_idxs, df_rules, cat_idx_mapping, num=1):
    df_rules_wrong = df_rules[~df_rules.cats.apply(lambda x: any([cat_idx_mapping.get(c,-1) in cat_idxs for c in x]))]
    df_rules_wrong = df_rules_wrong[df_rules_wrong.cats.apply(lambda x: len(x)>0)]
    if len(df_rules_wrong) >= num:
        selected_rules = df_rules_wrong.sample(num)
        wrong_rules = []
        for _, row in selected_rules.iterrows():
            wrong_rule_text = row['rule']
            wrong_rule_idx = random.choice([cat_idx_mapping[c] for c in row['cats']])
            wrong_rules.append((wrong_rule_idx, wrong_rule_text))
        return wrong_rules
    else:
        return None

def preproces_row_binary(inp):
    row, df_rules_subreddit, is_train, no_context, append_subreddit, min_context, train_class= inp
    append_rule = True

    pairs = []
    inputs_context = []
    context, final_comment = row['context'], row['final_comment']
    if len(context) >= min_context:
        rule_text, cats = row['rule_text'], row['cat']
        cat_idxs = [cat_idx_mapping[c] for c in cats]
        reply = encode(final_comment['tokens'])
        comment_id, conv_id, label, subreddit = row['comment_id'], row['conv_id'], row['bool_derail'], row['subreddit']
        if is_train:
            if label == True:
                # (train, derailment)
                for cat_idx in cat_idxs:
                    correct_rule_idx = cat_idx
                    if not no_context:
                        inputs_context = [encode(get_input_text(u["tokens"], rule_text, subreddit, append_rule, append_subreddit)) for u in context]
                    pairs.append((inputs_context, reply, 1, correct_rule_idx, conv_id + "_" + comment_id + "_" + str(correct_rule_idx), None))
                wrong_rule = get_wrong_rules(cat_idxs, df_rules_subreddit, cat_idx_mapping)
                if wrong_rule is not None:
                    wrong_rule_idx, wrong_rule_text = wrong_rule[0]
                    if not no_context:
                        inputs_context = [encode(get_input_text(u["tokens"], wrong_rule_text, subreddit, append_rule, append_subreddit)) for u in context]
                    pairs.append((inputs_context, reply, 0, wrong_rule_idx, conv_id + "_" + comment_id + "_" + str(wrong_rule_idx), None))
            else:
                # (train, non-derailment)
                correct_rule_idx = 0
                inputs_context = [encode(get_input_text(u["tokens"], rule_text, subreddit, append_rule, append_subreddit), max_length=MAX_LENGTH) for u in context]
                pairs.append((inputs_context, reply, 1, correct_rule_idx, conv_id + "_" + comment_id + "_" + str(correct_rule_idx), None))
                wrong_rules = get_wrong_rules([correct_rule_idx], df_rules_subreddit, cat_idx_mapping, num=2)
                if wrong_rules is not None:
                    for wrong_rule_idx, wrong_rule_text in wrong_rules:
                        if not no_context:
                            inputs_context = [encode(get_input_text(u["tokens"], wrong_rule_text, subreddit, append_rule, append_subreddit), max_length=MAX_LENGTH) for u in context]
                        pairs.append((inputs_context, reply, 0, wrong_rule_idx, conv_id + "_" + comment_id + "_" + str(wrong_rule_idx), None))
        else:
            context = context + [final_comment]
            if label == 1:
                # (test, derailment)
                for cat_idx in cat_idxs:
                    correct_rule_idx = cat_idx
                    wrong_rule = get_wrong_rules(cat_idxs, df_rules_subreddit, cat_idx_mapping)
                    for idx in range(1, len(context)):
                        reply = encode(get_input_text(context[idx]["tokens"], rule_text, subreddit, append_rule, append_subreddit))
                        comment_id = context[idx]["id"]
                        # gather as context all utterances preceding the reply
                        if not no_context:
                            inputs_context = [encode(get_input_text(u["tokens"], rule_text, subreddit, append_rule, append_subreddit)) for u in
                                context[:idx + 1]]
                        pairs.append(
                            (inputs_context, reply, 1, correct_rule_idx, conv_id + "_" + comment_id + "_" + str(correct_rule_idx), None))
                        if wrong_rule is None:
                            continue
                        wrong_rule_idx, wrong_rule_text = wrong_rule[0]
                        if not no_context:
                            inputs_context = [encode(get_input_text(u["tokens"], wrong_rule_text, subreddit, append_rule, append_subreddit)) for u in
                                context[:idx + 1]]
                        pairs.append((inputs_context, reply, 0, wrong_rule_idx,  conv_id + "_" + comment_id + "_" + str(wrong_rule_idx), None))
            else:
                # (test, non-derailment)
                correct_rule_idx = 0
                for idx in range(1, len(context)):
                    reply = encode(get_input_text(context[idx]["tokens"], rule_text, subreddit, append_rule, append_subreddit))
                    comment_id = context[idx]["id"]
                    # gather as context all utterances preceding the reply
                    if not no_context:
                        inputs_context = [encode(get_input_text(u["tokens"], rule_text, subreddit, append_rule, append_subreddit)) for u in
                                context[:idx + 1]]
                    pairs.append(
                        (inputs_context, reply, 1, correct_rule_idx, conv_id + "_" + comment_id + "_" + str(correct_rule_idx), None))
                wrong_rules = get_wrong_rules([correct_rule_idx], df_rules_subreddit, cat_idx_mapping, num=2)
                if wrong_rules is not None:
                    for wrong_idx, wrong_rule_text in wrong_rules:
                        for idx in range(1, len(context)):
                            reply = encode(get_input_text(context[idx]["tokens"], rule_text, subreddit, append_rule, append_subreddit))
                            comment_id = context[idx]["id"]
                            # gather as context all utterances preceding the reply
                            if not no_context:
                                inputs_context = [encode(get_input_text(context[idx]["tokens"], wrong_rule_text, subreddit, append_rule, append_subreddit)) for u in
                                        context[:idx + 1]]
                            pairs.append((inputs_context, reply, 0, wrong_idx, conv_id + "_" + comment_id + "_" + str(wrong_idx), None))
    return pairs


def preprocess_row_multi(inp):
    row, df_rules_subreddit, is_train, no_context, append_subreddit, min_context, train_class= inp
    append_rule = False

    inputs_context = []
    pairs = []
    context, final_comment = row['context'], row['final_comment']
    if len(context) >= min_context:
        rule_text, cats = row['rule_text'], row['cat']
        cat_idxs = [cat_idx_mapping[c] for c in cats]
        reply = encode(final_comment['tokens'])
        comment_id, conv_id, label, subreddit = row['comment_id'], row['conv_id'], row['bool_derail'], row['subreddit']
        if is_train:
            if label == True:
                # (train, derailment)
                if not no_context:
                    inputs_context = [encode(get_input_text(u["tokens"], rule_text, subreddit, append_rule, append_subreddit)) for u in context]
                pairs.append((inputs_context, reply, 1, cat_idxs, conv_id + "_" + comment_id + "_" + ",".join([str(c) for c in cat_idxs]), None))
            else:
                # (train, non-derailment)
                correct_rule_idx = 0
                if not no_context:
                    inputs_context = [encode(get_input_text(u["tokens"], rule_text, subreddit, append_rule, append_subreddit), max_length=MAX_LENGTH) for u in context]
                pairs.append((inputs_context, reply, 1, [correct_rule_idx], conv_id + "_" + comment_id + "_" + str(correct_rule_idx), None))
        else:
            context = context + [final_comment]
            if label == 1:
                # (test, derailment)
                for idx in range(1, len(context)):
                    reply = encode(get_input_text(context[idx]["tokens"], rule_text, subreddit, append_rule, append_subreddit))
                    comment_id = context[idx]["id"]
                    # gather as context all utterances preceding the reply
                    if not no_context:
                        inputs_context = [encode(get_input_text(u["tokens"], rule_text, subreddit, append_rule, append_subreddit)) for u in
                                context[:idx + 1]]
                    pairs.append(
                        (inputs_context, reply, 1, cat_idxs, conv_id + "_" + comment_id + "_" + ",".join([str(c) for c in cat_idxs]), None))
            else:
                # (test, non-derailment)
                correct_rule_idx = 0
                for idx in range(1, len(context)):
                    reply = encode(get_input_text(context[idx]["tokens"], rule_text, subreddit, append_rule, append_subreddit))
                    comment_id = context[idx]["id"]
                    # gather as context all utterances preceding the reply
                    if not no_context:
                        inputs_context = [encode(get_input_text(u["tokens"], rule_text, subreddit, append_rule, append_subreddit)) for u in
                                context[:idx + 1]]
                    pairs.append(
                        (inputs_context, reply, 1, [correct_rule_idx], conv_id + "_" + comment_id + "_" + str(correct_rule_idx), None))
    return pairs

def preprocess_data(df, df_rules, cat_idx_mapping, min_context=2, multi_class=False, no_context=False, append_subreddit=None, train_class=None, is_train=False):
    pairs = []
    rule_counts = {}
    num_rule = len(cat_idx_mapping)
    df_rules_subreddit = {subreddit: sub_df for subreddit, sub_df in df_rules.groupby('subreddit')}

    preprocess_func = preprocess_row_multi if multi_class else preproces_row_binary
    with Pool(NUM_PROCESS) as p:
        data = list(tqdm(p.imap(preprocess_func,[(row, df_rules_subreddit[row['subreddit']], is_train, no_context, append_subreddit, min_context, train_class) for row in df]), total=len(df)))
    pairs = [pair for pairs in data for pair in pairs]
    return pairs


def dialogBatch2UtteranceBatch(dialog_batch):
    utt_tuples = []  # will store tuples of (utterance, original position in batch, original position in dialog)
    for batch_idx in range(len(dialog_batch)):
        dialog = dialog_batch[batch_idx]
        for dialog_idx in range(len(dialog)):
            utterance = dialog[dialog_idx]
            utt_tuples.append((utterance, batch_idx, dialog_idx))
    # sort the utterances in descending order of length, to remain consistent with pytorch padding requirements
    utt_tuples.sort(key=lambda x: len(x[0]), reverse=True)
    # return the utterances, original batch indices, and original dialog indices as separate lists
    utt_batch = [u[0] for u in utt_tuples]
    batch_indices = [u[1] for u in utt_tuples]
    dialog_indices = [u[2] for u in utt_tuples]
    return utt_batch, batch_indices, dialog_indices


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l):
    indexes_batch = l
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l):
    indexes_batch = l
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch, already_sorted=False, use_binary_label=True):
    if not already_sorted:
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch, label_batch, id_batch, rule_batch = [], [], [], [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        if use_binary_label:
            label_batch.append(pair[2])
        else:
            label_batch.append(idxs2onehot(pair[3]))
        id_batch.append(pair[4])
        rule_batch.append(pair[5])

    dialog_lengths = torch.tensor([len(x) for x in input_batch])
    input_utterances, batch_indices, dialog_indices = dialogBatch2UtteranceBatch(input_batch)
    inp, utt_lengths = inputVar(input_utterances)
    rule, rule_lengths =  None, None
    output, mask, max_target_len = outputVar(output_batch)
    label_batch = torch.FloatTensor(label_batch) if label_batch[0] is not None else None
    return inp, dialog_lengths, utt_lengths, batch_indices, dialog_indices, label_batch, id_batch, output, mask, max_target_len, rule, rule_lengths



def makeContextEncoderInput(utt_encoder_hidden, dialog_lengths, batch_size, batch_indices, dialog_indices):
    """The utterance encoder takes in utterances in combined batches, with no knowledge of which ones go where in which conversation.
       Its output is therefore also unordered. We correct this by using the information computed during tensor conversion to regroup
       the utterance vectors into their proper conversational order."""
    # first, sum the forward and backward encoder states
    utt_encoder_summed = utt_encoder_hidden
    # we now have hidden state of shape [utterance_batch_size, hidden_size]
    # split it into a list of [hidden_size,] x utterance_batch_size
    last_states = [t.squeeze() for t in utt_encoder_summed.split(1, dim=0)]

    # create a placeholder list of tensors to group the states by source dialog
    states_dialog_batched = [[None for _ in range(dialog_lengths[i])] for i in range(batch_size)]

    # group the states by source dialog
    for hidden_state, batch_idx, dialog_idx in zip(last_states, batch_indices, dialog_indices):
        states_dialog_batched[batch_idx][dialog_idx] = hidden_state

    # stack each dialog into a tensor of shape [dialog_length, hidden_size]
    states_dialog_batched = [torch.stack(d) for d in states_dialog_batched]

    # finally, condense all the dialog tensors into a single zero-padded tensor
    # of shape [max_dialog_length, batch_size, hidden_size]
    return torch.nn.utils.rnn.pad_sequence(states_dialog_batched)

def idxs2onehot(lst_cats):
    onehot = [0] * NUM_CLASSES
    for cat in lst_cats:
        onehot[cat] = 1
    return onehot

def onehot2idxs(onehot):
    cats = []
    for idx, score in enumerate(onehot):
        if score == 1:
            cats.append(idx)
    return cats

def batchIterator(source_data, batch_size, shuffle=True, use_binary_label=True):
    cur_idx = 0
    if shuffle:
        random.shuffle(source_data)
    while True:
        if cur_idx >= len(source_data):
            cur_idx = 0
            if shuffle:
                random.shuffle(source_data)
        batch = source_data[cur_idx:(cur_idx + batch_size)]
        # the true batch size may be smaller than the given batch size if there is not enough data left
        true_batch_size = len(batch)
        # ensure that the dialogs in this batch are sorted by length, as expected by the padding module
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        # for analysis purposes, get the source dialogs and labels associated with this batch
        batch_dialogs = [x[0] for x in batch]
        if use_binary_label:
            batch_labels = [x[2] for x in batch]
        else:
            batch_labels = [idxs2onehot(x[3]) for x in batch]
        # convert batch to tensors
        batch_tensors = batch2TrainData(batch, already_sorted=True, use_binary_label=use_binary_label)
        yield (batch_tensors, batch_dialogs, batch_labels, true_batch_size)
        cur_idx += batch_size


def train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
          rule, rule_lengths,
          labels, use_binary_label, 
          # input/output arguments
          encoder, context_encoder, attack_clf,  # network arguments
          encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,  # optimization arguments
          batch_size, clip, max_length=MAX_LENGTH):  # misc arguments

    # Zero gradients
    encoder_optimizer.zero_grad()
    context_encoder_optimizer.zero_grad()
    attack_clf_optimizer.zero_grad()


    # Set device options
    input_variable = input_variable.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    labels = labels.to(device)
    # Forward pass through utterance encoder
    utt_encoder_hidden = encoder(input_variable, utt_lengths)

    # Convert utterance encoder final states to batched dialogs for use by context encoder
    context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices,
                                                    dialog_indices)

    # Forward pass through context encoder

    context_encoder_outputs, _ = context_encoder(context_encoder_input, dialog_lengths, hidden=None, rule=rule,
                                                 rule_lengths=rule_lengths)

    # Forward pass through classifier to get prediction logits
    logits = attack_clf(context_encoder_outputs, dialog_lengths)

    # Calculate loss
    # loss = F.binary_cross_entropy_with_logits(logits, labels)
    if use_binary_label:
        labels = labels.long()
    loss = criterion(logits, labels)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(attack_clf.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    context_encoder_optimizer.step()
    attack_clf_optimizer.step()

    return loss.item()


def evaluateBatch(encoder, context_encoder, predictor, input_batch, dialog_lengths,
                  dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, rule, rule_lengths, batch_size,
                  use_binary_label, device, thre=0.5,
                  max_length=MAX_LENGTH):
    # Set device options
    input_batch = input_batch.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    # Predict future attack using predictor
    scores = predictor(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
                       rule, rule_lengths,
                       batch_size, max_length)
    if len(scores.shape) == 1:
        scores = scores.unsqueeze(0)
    
    if use_binary_label:
        _, predictions = torch.max(scores, 1)
    else:
        scores = sigmoid(scores)
        predictions = [[idx for idx, s in enumerate(score) if s >= thre] for score in scores]
        predictions = [pred if len(pred)>0 else [0] for pred in predictions]

    return predictions, scores

def evaluateDataset(dataset, encoder, context_encoder, predictor, batch_size, use_binary_label):
    # create a batch iterator for the given data
    batch_iterator = batchIterator(dataset, batch_size, shuffle=False, use_binary_label=use_binary_label)
    # find out how many iterations we will need to cover the whole dataset
    n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
    output_df = {
        "id": [],
        "prediction": [],
        "score": [],
        'label': []
    }
    with torch.no_grad():
        for iteration in range(1, n_iters + 1):
            batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
            # Extract fields from batch
            input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, conv_ids, target_variable, mask, max_target_len, rule, rule_lengths = batch
            dialog_lengths_list = [len(x) for x in batch_dialogs]
            # run the model
            predictions, scores = evaluateBatch(encoder, context_encoder, predictor, input_variable,
                                                dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices,
                                                dialog_indices, rule, rule_lengths,
                                                true_batch_size, use_binary_label, device)

            # format the output as a dataframe (which we can later re-join with the corpus)
            for i in range(true_batch_size):
                conv_id = conv_ids[i]
                pred = predictions[i].item() if use_binary_label else predictions[i]
                label = labels[i].item() if use_binary_label else onehot2idxs(labels[i])
                score = -1  # scores[i].item()
                output_df["id"].append(conv_id)
                output_df["prediction"].append(pred)
                output_df["score"].append(score)
                output_df["label"].append(label)
            # if iteration % 1000 == 0:
            #     print(f"Iteration: {iteration}/{n_iters} ({iteration/n_iters*100:.1f}%)")

    return pd.DataFrame(output_df).set_index("id")

# Utility fn to calculate val F1, used during training to check for best model
def get_best_val_f1(val_pairs, convid_to_uttr, encoder, context_encoder, predictor, batch_size, use_binary_label):
    forecasts_df = evaluateDataset(val_pairs, encoder, context_encoder, predictor, batch_size, use_binary_label)
    forecasts_df.to_csv("temp/temp.tsv",sep="\t")

    predicted_corpus = {}
    for idx, row in forecasts_df.iterrows():
        try:
            conv_id, clean, utt_id, rule_id = idx.split('_')
        except:
            try:
                _, _, conv_id, clean, utt_id, rule_id = idx.split('_')
            except:
                print(idx)
                continue
        conv_id = conv_id + "_" + clean  # Because I messed up
        key = conv_id + "_" + rule_id
        if key not in predicted_corpus:
            label = int(row['label']) if use_binary_label else row['label']
            predicted_corpus[key] = {'conv_id': conv_id, 'rule_id': rule_id,'utts': [[utterance['id'], -1, utt_idx+1] for utt_idx, utterance in enumerate(convid_to_uttr[conv_id])],
                                     'label': label}
        preds = predicted_corpus[key]['utts']
        for idx in range(len(preds)):
            pred = preds[idx]
            if pred[0] == utt_id:
                pred[1] = row['prediction']

    conversational_forecasts_df = []
    for conv_rule_id in predicted_corpus:
        try:
            conv_id, clean, rule_id = conv_rule_id.split("_")
        except:
            try:
                _, _, conv_id, clean, rule_id = conv_rule_id.split('_')
            except:
                print(conv_rule_id)
                continue
        conv_id = conv_id+"_"+clean
        out = 0
        out_idx = set()
        earliest_idx = -1
        for utt_id, pred, utt_idx in predicted_corpus[conv_rule_id]['utts']:
            if use_binary_label:
                if pred == 1:
                    out = 1
                    if earliest_idx == -1:
                        earliest_idx = utt_idx
                    elif utt_idx < earliest_idx:
                        earliest_idx = utt_idx            
            else:
                if pred != [0] and pred != -1:
                    out = 1
                    out_idx.update(pred)
                    if earliest_idx == -1:
                        earliest_idx = utt_idx
                    elif utt_idx < earliest_idx:
                        earliest_idx = utt_idx            
        
        out_idx = [0] if out == 0 else list(out_idx)
        
        conversational_forecasts_df.append({'conv_id':conv_id, 'rule_id':rule_id,'conv_idx':earliest_idx ,'num_uttr':len(predicted_corpus[conv_rule_id]['utts']),'prediction': out if use_binary_label else out_idx, 'label': predicted_corpus[conv_rule_id]['label']})

    conversational_forecasts_df = pd.DataFrame(conversational_forecasts_df)
    conversational_forecasts_df.to_csv("temp/temp2.tsv",sep="\t")
    
    if use_binary_label:
        f1 = f1_score(conversational_forecasts_df.label, conversational_forecasts_df.prediction, average='macro')
    else:
        x, y = conversational_forecasts_df.prediction.tolist(), conversational_forecasts_df.label.tolist()
        m = MultiLabelBinarizer().fit(x+y)
        x,y = m.transform(x), m.transform(y)
        f1 = f1_score(y, x, average="macro")
    return f1, conversational_forecasts_df, predicted_corpus

def get_convid_to_uttr(split_data):
    _convid_to_uttr = {}
    for d in split_data:
        _convid_to_uttr[d['conv_id']] = d['context'] + [d['final_comment']]
    return _convid_to_uttr

def evaluate(iteration, loss, encoder, context_encoder, attack_clf, convid_to_uttr, path_save, best_f1, valid_batch_size, use_binary_label):
    print("Validating...", end="")
    # put the network components into evaluation mode
    encoder.eval()
    context_encoder.eval()
    attack_clf.eval()

    
    predictor = Predictor(encoder, context_encoder, attack_clf)
    start = time()
    dev_f1, forecasts_df, predicted_corpus = get_best_val_f1(processed_data['dev'], convid_to_uttr['dev'], encoder, context_encoder, predictor, valid_batch_size, use_binary_label)
    print(f" f1: {dev_f1 * 100:.1f} (took {(time()-start)/60:.1f}m)")

    # keep track of our best model so far
    if dev_f1 > best_f1:
        print("Validation accuracy better than current best; saving model...")
        best_f1 = dev_f1
        out_file_dev = f'dev_forecasts_df_{dev_f1*100:.1f}.tsv'
        forecasts_df.to_csv(join(path_save, out_file_dev), sep='\t')

        with open(join(path_save, f'predicted_corpus_{dev_f1 * 100:.1f}.json'), 'w') as f:
            json.dump(predicted_corpus, f)
        
        print("Generating predictions for test set...", end="")
        start = time()
        test_f1, forecasts_df, predicted_corpus = get_best_val_f1(processed_data['test'], convid_to_uttr['test'], encoder, context_encoder, predictor, valid_batch_size, use_binary_label)
        print(f" (took {(time()-start)/60:.1f}m)")
        out_file_test = f'test_forecasts_df_{test_f1*100:.1f}.tsv'
        forecasts_df.to_csv(join(path_save, out_file_test), sep='\t')

        config = {
            'step': iteration,
            'epoch': round(iteration/n_iter_per_epoch, 2),
            'loss': loss,
            'valid_f1': dev_f1,
            'test_f1': test_f1,
            'file_out_valid': out_file_dev,
            'file_out_test':out_file_test
        }
        save_json(config, join(path_save, "best_config.json"))
        torch.save(config.update({
            'en': encoder.state_dict(),
            'ctx': context_encoder.state_dict(),
            'atk_clf': attack_clf.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'ctx_opt': context_encoder_optimizer.state_dict(),
            'atk_clf_opt': attack_clf_optimizer.state_dict(),
        }), join(path_save,"finetuned_model.pt"))
    return best_f1, dev_f1

def trainIters(processed_data, original_data, path_save, encoder, context_encoder, attack_clf,
               encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer, use_binary_label, 
               n_iteration, batch_size, valid_batch_size, min_iter, early_stopping_patience, print_every, validate_every, clip):
    # create a batch iterator for training data
    batch_iterator = batchIterator(processed_data['train'], batch_size, shuffle=True, use_binary_label=use_binary_label)
    convid_to_uttr = {}
    convid_to_uttr['dev'] = get_convid_to_uttr(data['dev'])
    convid_to_uttr['test'] = get_convid_to_uttr(data['test'])

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0

    # Training loop
    print("Training...")
    # keep track of best validation accuracy - only save when we have a model that beats the current best
    best_f1 = 0
    es_counter = 0

    very_first = time()
    start_time = time()
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch, training_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, _, target_variable, mask, max_target_len, rule, rule_lengths = training_batch
        dialog_lengths_list = [len(x) for x in training_dialogs]

        # Run a training iteration with batch
        loss = train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
                     rule, rule_lengths,
                     labels, use_binary_label,  # input/output arguments
                     encoder, context_encoder, attack_clf,  # network arguments
                     encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,  # optimization arguments
                     true_batch_size, clip)  # misc arguments
        # loss = 0
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            
            took = time()-start_time
            took_total = time()-very_first
            remaining = took_total/iteration*(n_iteration-iteration)

            print(f"Iteration: {iteration}/{n_iteration} ({iteration/n_iteration*100:.1f}%)\tAverage loss: {print_loss_avg:.4f}\ttook {took:.0f}s (cum:{took_total:.0f}s, ETA:{remaining/60:.0f} min)")
            start_time = time()
            print_loss = 0
            # print("Memory used : ", torch.cuda.max_memory_allocated())

        # Evaluate on validation set
        if (iteration % validate_every == 0):
            best_f1, f1 = evaluate(iteration, loss, encoder, context_encoder, attack_clf, convid_to_uttr, path_save, best_f1, valid_batch_size, use_binary_label)
            if iteration > min_iter:
                if best_f1 != f1:
                    es_counter += 1
                    print(f"performance not improved. early stopping counter increased to {es_counter}")
                else:
                    es_counter = 0
                if es_counter >= early_stopping_patience:
                    # stop training
                    print(f"early stopping count reached its max patience ({early_stopping_patience}). Stopping the training. ")
                    break

            # put the network components back into training mode
            encoder.train()
            context_encoder.train()
            attack_clf.train()
    best_f1, _ = evaluate(iteration, loss, encoder, context_encoder, attack_clf, convid_to_uttr, path_save, best_f1, valid_batch_size, use_binary_label)
    print(f"final best f1 (dev set): {best_f1*100:.1f}")


rule_configs = {
    "rules": {
        "incivility": [
            "Incivility isn’t allowed on this sub. We want to encourage a respectful discussion.",
            "Remain civil towards other users, no expressions of ableism, homophobia, racism, sexism, transphobia, gendered slurs, ethnic slurs, slurs referring to disabilities and slurs against LGBT",
            "No trolling, hate speech, derogatory slurs, and personal attacks.",
            "Discrimination of any kind is not ok. No slurs or hate speech. Don't be a Jerk. Don't be Rude or Condescending. No trolling, personal attacks",
            "Stay respectful, polite, and friendly. No bigoted slurs, directed at other users. Don't insult people",
            "Personal attacks, insults, racial, homophobic, xenophobic, and sexist are not allowed",
        ],
        "politics": [
            "Purely political posts and comments will be removed",
            "No political debate",
            "Political discussion is not acceptable",
            "Comments cannot be inherently political, attempts to derail will be removed",
            "Off topic political, policy, and economic posts and comments will be removed",
            "Shaming campaigns, politician's take or political opinion pieces are not allowed",
            "No political opinions or hot takes or sensationalist controversies or tweets from president",
        ]
    }
}

def load_dataset(path, filter_removed=True):
    data = {}
    for split in ["train","test","dev"]:
        data[split]= import_jsonl(join(path, split+".jsonl"))
        if filter_removed:
            data[split] = {k:v for k,v in data[split] if v['final_comment'] != "[removed]"}
    return data

def sample_dataset(data, num_sample=200):
    for split in ["train","test","dev"]:
        data[split] = random.sample(data[split], num_sample)
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', "-conv", type=str, required=True, help='conversation data path')
parser.add_argument('--path_rule', "-rule", type=str, required=True, help='labeled rules data path')
parser.add_argument('--path_save', "-save", type=str, required=True, help='where to save the data splits')
parser.add_argument('--min_context', "-context", action='store', type=int, default=2, help='minimum number of utterances for context')
parser.add_argument('--epoch', "-ep", action='store', type=int, default=EPOCHS, help='num of training epoch')
parser.add_argument('--min_epoch', "-me", action='store', type=int, default=3, help='minimum epoch to train')
parser.add_argument('--early_stopping', "-es", action='store', type=int, default=EARLY_STOPPING, help='early stopping patience')

parser.add_argument('--batch_size', "-bs", action='store', type=int, default=BATCH_SIZE, help='num of training epoch')
parser.add_argument('--valid_batch_size', "-vbs", action='store', type=int, default=VALID_BATCH_SIZE, help='num of training epoch')
parser.add_argument('--print_every', "-print", action='store', type=int, default=200, help='minimum number of utterances for context')
parser.add_argument('--validate_every', "-validate", action='store', type=int, default=1000, help='minimum number of utterances for context')

parser.add_argument('--multi_class', "-multi", action='store_true', default=False, help='multi-class')
parser.add_argument('--no_context', "-noc", action='store_true', default=False, help='no context')
parser.add_argument('--append_subreddit', "-sub", action='store', default=None, choices=["subreddit","nsfw"], help='verbose')
parser.add_argument('--train_class', "-class", action='store', nargs='+', default=None, help='verbose')

parser.add_argument('--random_seed', "-seed", action='store', type=int, default=2021, help='random seed')
parser.add_argument('--verbose', "-v", action='store_true', default=False, help='verbose')
parser.add_argument('--sample', "-toy", action='store_true', default=False, help='sample dataset (toy experiment)')
args = parser.parse_args()
assert exists(args.path_data) and exists(args.path_rule)

set_random_seed(args.random_seed)

if args.path_save.endswith("/"):
    args.path_save = args.path_save[:-1]
timestamp = datetime.now().strftime("_%m%d_%H%M")
args.path_save = args.path_save + timestamp
if not os.path.isdir(args.path_save):
    os.mkdir(args.path_save)
print(f"saving trained models to {args.path_save}")

data = load_dataset(args.path_data)
df_rules = import_tsv(args.path_rule)
df_rules['cats'] = df_rules['cats'].apply(lambda x: eval(x))
cat_idx_mapping = import_json("data/mappings/cat10_to_idx.json")
cat_idx_mapping['neutral'] = 0

def filter_class(train_data, train_class):
    train_class = set(train_class)
    new_data = []
    new_data_comment_id = set()
    num_derail, num_non_derail = 0, 0
    for data in train_data:
        if data['bool_derail']:
            cats = [c for c in data['cat'] if c in train_class]
            if len(cats)>0:
                new_data.append(data)
                new_data_comment_id.add(data['comment_id'])
                num_derail += 1
    for data in train_data:
        if not data['bool_derail']:
            comment_id = data['comment_id']
            if comment_id in new_data_comment_id:
                new_data.append(data)
                num_non_derail += 1
    print(f"after filtering for {train_class}, {num_derail} derailments + {num_non_derail} non-derailments")
    return new_data

if args.train_class is not None:
    data['train'] = filter_class(data['train'], args.train_class)
    data['dev'] = filter_class(data['dev'], args.train_class)

if args.sample:
    data = sample_dataset(data, 300)

processed_data = {}
processed_data['train'] = preprocess_data(data['train'], df_rules, cat_idx_mapping, args.min_context, args.multi_class, args.no_context, append_subreddit=args.append_subreddit, train_class=args.train_class, is_train=True)
processed_data['dev'] = preprocess_data(data['dev'], df_rules, cat_idx_mapping, args.min_context, args.multi_class, args.no_context, append_subreddit=args.append_subreddit, is_train=False)
processed_data['test'] = preprocess_data(data['test'], df_rules, cat_idx_mapping, args.min_context, args.multi_class, args.no_context, append_subreddit=args.append_subreddit, is_train=False)

# train_pairs = loadPairs(corpus, 'train', is_train=True)
# length = len(train_pairs)
# val_pairs = loadPairs(corpus, 'val', is_train=False)
# print(test_convo_ids)

NUM_CLASSES = len(cat_idx_mapping) if args.multi_class else 2
criterion = nn.BCEWithLogitsLoss() if args.multi_class else nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1]).to(device))

# Define model
encoder = EncoderBERT(device)
context_encoder = ContextEncoderRNN(HIDDEN_SIZE, ENC_NUM_LAYER, DROPOUT_PROB, device)
attack_clf = SingleTargetClf(HIDDEN_SIZE, NUM_CLASSES, DROPOUT_PROB, device)

print('Models built and ready to go!')

# Put dropout layers in train mode
encoder.train()
context_encoder.train()
attack_clf.train()

# Initialize optimizers
print('Building optimizers...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
context_encoder_optimizer = optim.Adam(context_encoder.parameters(), lr=LR)
attack_clf_optimizer = optim.Adam(attack_clf.parameters(), lr=LR)

# Compute the number of training iterations we will need in order to achieve the number of epochs specified in the settings at the start of the notebook
n_iter_per_epoch = len(processed_data['train']) // args.batch_size + int(len(processed_data['train']) % args.batch_size == 1)
n_iteration = n_iter_per_epoch * args.epoch
min_iter = args.min_epoch * n_iter_per_epoch
print("Will train for {} iterations".format(n_iteration))

# Run training iterations, validating after every epoch
training_started = time()
trainIters(processed_data, data, args.path_save, encoder, context_encoder, attack_clf,
           encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer, (not args.multi_class), 
           n_iteration, args.batch_size, args.valid_batch_size, min_iter, args.early_stopping, args.print_every, args.validate_every, CLIP)
training_ended = time()
print(f"training for {args.epoch} took {(time()-training_started)/60:.0f}min")
