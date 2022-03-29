import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


assert len(sys.argv) == 2, "usage: python get_top_subreddits.py {max_rank}"
max_rank = int(sys.argv[1])
base_url="https://frontpagemetrics.com/top/offset/"

path_save = "data/subreddits/"
path_out = path_save+f"top{max_rank}.json"
if not os.path.isdir(path_save):
    os.makedirs(path_save)

def process_row(row):
    row_data = row.find_all("td")
    rank, name, subscribers = row_data
    rank, name, subscribers = rank.text.strip(), name.text.strip(), subscribers.text.strip()
    
    rank = int(rank.replace(",",""))
    name = name.replace("/r/","")
    subscribers = int(subscribers.replace(",",""))
    return rank, name, subscribers

data = {}
for page_rank in tqdm(range(0, max_rank, 100), total=int(max_rank/100)):
    url=base_url+str(page_rank)
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, features="html.parser")

    table = soup.find("table", attrs={"class": "table-bordered"})
    rows = table.find_all("tr")[1:] # exclude the heading
    if len(rows) != 100:
        print("less than 100 rows. Either reached the end of the list or error")
        print(url)

    if max_rank < 100:
        rows = rows[:max_rank]

    for row in rows:
        rank, name, subscribers = process_row(row)
        data[name] = (rank, subscribers)

with open(path_out, "w") as file:
    json.dump(data, file)
print(f"list of top {max_rank} subreddits are saved to {path_out}")