This repository contains the code for the EMNLP 2021 Findings paper: [Detecting Community Sensitive Norm Violations in Online Conversations](https://aclanthology.org/2021.findings-emnlp.288.pdf). Please cite this paper if you use this dataset:

```
@inproceedings{park-etal-2021-detecting-community,
    title = "Detecting Community Sensitive Norm Violations in Online Conversations",
    author = "Park, Chan Young  and
      Mendelsohn, Julia  and
      Radhakrishnan, Karthik  and
      Jain, Kinjal  and
      Kanakagiri, Tushar  and
      Jurgens, David  and
      Tsvetkov, Yulia",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.288",
    doi = "10.18653/v1/2021.findings-emnlp.288",
    pages = "3386--3397",
}
```

# Dependencies & Pre-requisite

* Refer to requirements.txt for package requirements
* Mainly, you need the following python packages: *pandas*, *praw*, *psaw*, *convokit*, *tqdm*, *beautifulsoup4*, *scikit-learn* for data collection, and *torch*, *transformers*, *datasets* for model training/inference.
* You need your own Reddit account and a pair of API ID & Secret key. If you don't have one you can obtain by registering an app [here](https://www.reddit.com/prefs/apps/).


# Data & Trained Models:
You can find three items in this Google Drive folder: [drive.google.com/drive/folders/1IB823a-xc0WT9903ECbmBrEdCR8cppFp](https://drive.google.com/drive/folders/1IB823a-xc0WT9903ECbmBrEdCR8cppFp)

1. `max500_subs100000_redacted.zip`: NormVio Dataset that has texts redacted. We instead provide Reddit IDs of comments and posts. 

2. `training_data_for_rule_classifiers.zip`: Reddit community rules and their categories. Provided by the authors of [Reddit Rules! Characterizing an Ecosystem of Governance (ICWSM 2018)](https://ojs.aaai.org/index.php/ICWSM/article/view/15033). Please cite the paper if you plan to use this data.

3. `rule-classifiers.zip`: Trained Bert-based classifiers used in our paper. We trained one model per category, and each model outputs a binary decision on whether a given rule belongs to the target rule category.

# Reconstructing the Original Dataset Used in the Paper
Due to privacy concerns, we only release a redacted version of NormVio, which has texts of comments and posts removed. The released data contains a collection of posts and comments and their Reddit IDs. You can reconstruct the original data using the provided IDs. We also provide a script that will check if the comments or posts are still in the PushShift data dump and recover. You can simply follow the steps listed below:
- download the redacted zip file from the google drive folder above and then unzip it. Place it under `data/processed/`
- set up an environment (for example, make a new virtualenv environment and run `pip install -r requirements.txt`)
- first, go to src/config.py and make sure you fill out your Reddit API info into these entries: `PRAW_CLIENT_ID`, `PRAW_CLIENT_SECRET`, `PRAW_USERNAME`, `PRAW_PW`
- Run `python src/restore_from_redacted.py --path_data DOWNLOADED_REDACTED_DATA --path_save SAVE_PATH -v`. Note: you can also set `--mod_original_comment` to reconstruct the original comments from moderators left at the time of moderation.

Reconstruction should take about 20-30 hours, so I recommend running it on a server or when you know you will have a stable internet connection for a while. 

# Building a Dataset from Scratch
If you want to build a dataset from scratch, do the following steps:
- You need to download rule classifier training examples from the google drive folder. Download the rule classification data from the provided link and check if the path to the data is correctly set as `data/training/rule-classification`
- (optional) You could also download the trained rule classifiers so that you don't have to train it (download `rule-classifiers.zip` and place it under "res/rule-classifier/")
- Read through `scripts/main.sh` to see exactly what is being done in each step, and also tweak two parameters (NUM_SUBREDDIT, MAX_COMMENT_PER_MOD) as you want, which decides how many subreddits you want to scrape from and how many comments per moderator you want to look back (I recommend setting this value as somewhere near 500 as it can really slow down the scraping if you set this to a  large number).
- Finally, run `bash scripts/main.sh`
- **WARNING**: The data collection code is taking way longer than it used to be, so I don't recommend using this code at this moment. I plan to look into this issue and try to fix it.

# Training Norm Violation Detection Models
*To Be added*


# Contact
The main maintainer of this repository is @chan0park. Please feel free to reach out using the email listed in the paper.