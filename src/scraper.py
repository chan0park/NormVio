import sys
import time
import praw
from tqdm import tqdm
from psaw import PushshiftAPI
from config import PRAW_CLIENT_ID, PRAW_CLIENT_SECRET, PRAW_USERNAME, PRAW_PW, NUM_PROCESS

def wrap_error_with_blank(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except:
            result = []
            print("Error with function - ", func, " with args ", args, kwargs)
        return result

    return wrapper

class RedditScraper:
    def __init__(self):
        if PRAW_CLIENT_ID and PRAW_CLIENT_SECRET and PRAW_USERNAME and PRAW_PW:
            try:
                self.r = praw.Reddit(client_id=PRAW_CLIENT_ID, client_secret=PRAW_CLIENT_SECRET,
                                    user_agent='Get Comments', username=PRAW_USERNAME, password=PRAW_PW)
                if self.r.user.me() is None:
                    print("Reddit API Authentication failed")
                    self.r = None
                    self.s = None
                else:
                    self.s = PushshiftAPI(self.r)
                #     raise BaseException("Reddit API Authentication failed")
            except:
                self.r = None
                self.s = None
        self.s_without_r = PushshiftAPI()

    @wrap_error_with_blank
    def get_moderators(self, subreddit):
        try:
            moderators = [moderator for moderator in self.r.subreddit(
                subreddit).moderator()]
        except:
            time.sleep(10)
            try:
                moderators = [moderator for moderator in self.r.subreddit(
                    subreddit).moderator()]
            except:
                moderators = []
                print("Error with r/", subreddit)
        return moderators
    
    @wrap_error_with_blank
    def get_mod_comments(self, mods, subreddit, limit=100):
        mod_comments = []
        for mod in tqdm(mods, leave=False):
            _mod_comments = self.get_comments(mod, limit=limit, subreddit=subreddit)
            mod_comments += [_mod_comments]
        # mod_comments = [self.get_comments(
        #     mod, limit=limit, subreddit=subreddit) for mod in mods]
        mod_comments = [[comment for comment in thread] for thread in mod_comments]
        return mod_comments

    @wrap_error_with_blank
    def get_comments(self, user, limit=100, subreddit=None):
        try:
            comments = list(self.s._search(kind="comment", author=user, subreddit=subreddit, limit=limit))
        except:
            try:
                comments = [comment.replace("\n"," ") for comment in user.comments.new(limit=limit)]
                if subreddit is not None:
                    comments = [comment for comment in comments if comment.subreddit_name_prefixed.replace("r/", "")==subreddit]
            except:
                comments = []
                print("Error with u/", user.name)
        return comments

    @wrap_error_with_blank
    def get_community_rules(self, subreddit):
            try:
                rules = [rule for rule in self.r.subreddit(subreddit).rules]
            except:
                rules = None
                print("Rule Scraping Error with ", subreddit)
            return rules
    
    @wrap_error_with_blank
    def get_submissions(self, subreddit, limit=100, time_filter='month'):
        return [x for x in self.r.subreddit(subreddit).top(limit=limit, time_filter=time_filter)]

    def fetch_psaw_from_id(self, id_to_be_fetched, subreddit, kind="comment"):
        gen = self.s_without_r._search(kind=kind, ids=id_to_be_fetched, subreddit=subreddit)
        for i in range(125):
            try:
                x = next(gen)
                if x.id == id_to_be_fetched:
                    return x
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                return None
        return None

    def fetch_psaw_from_ids(self, ids_to_be_fetched, subreddit=None, kind="comment"):
        def _get_psaw_from_ids(ids, subreddit, kind):
            ids_set = set(ids)
            fetched = []
            gen = self.s_without_r._search(kind=kind, ids=','.join(ids), subreddit=subreddit)
            for _ in range(125):
                try:
                    c = next(gen)
                    if c.id in ids_set:
                        fetched.append(c)
                        ids_set.remove(c.id)
                        if len(ids_set) == 0:
                            return fetched
                except KeyboardInterrupt:
                    sys.exit(0)
                except:
                    return fetched
                    
        # Each ID is 8 (7+1 for comma) chars, max chars is 1000, 1000/8 = 125
        n = 125
        ids_to_be_fetched = list(ids_to_be_fetched)
        chunks = [ids_to_be_fetched[i:i + n] for i in range(0, len(ids_to_be_fetched), n)]
        
        if len(chunks)>1:
            psaws = []
            for chunk in tqdm(chunks, leave=False):
                psaws.append(_get_psaw_from_ids(chunk, subreddit, kind))
        else:
            psaws = [_get_psaw_from_ids(chunk, subreddit, kind) for chunk in chunks]
        psaws = [comment for sublist in psaws for comment in sublist]
        psaws = {x.id: x for x in psaws}
        return psaws