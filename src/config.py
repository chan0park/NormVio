# Reddit API related (replace them with your info)
PRAW_CLIENT_ID = ''
PRAW_CLIENT_SECRET = ''
PRAW_USERNAME=""
PRAW_PW=""

# default values
SEP_STRING = " ||| "

# Data Collection
NUM_PROCESS = 24
MAX_DEPTH = 10
MOD_COMMENT_LIMIT = 500
SAVE_EVERY= 100

# Derailment Model-related
BERT_TYPE = 'DeepPavlov/bert-base-cased-conversational'
BERT_CACHE = "~/.cache/huggingface/transformers/"
HIDDEN_SIZE = 768
MAX_LENGTH = 120
ENC_NUM_LAYER = 2
DEC_NUM_LAYER = 2
DROPOUT_PROB = 0.1

# Configure training/optimization
BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EARLY_STOPPING = 5
CLIP = 50.0
TF_RATIO = 1.0
LR = 1e-5
DEC_LR = 5.0
EPOCHS = 50

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown word token