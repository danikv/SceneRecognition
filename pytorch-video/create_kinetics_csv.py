import argparse
import urllib
import urllib.request
import json
import pandas as pd

parser = argparse.ArgumentParser(description='train a lstm over a videos dataset using grid search')
parser.add_argument('--dataset', help='the dataset csv file')
parser.add_argument('--output', help='the dataset csv file')


args = parser.parse_args()
dataset = args.dataset
output = args.output

json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.URLopener().retrieve(json_url, json_filename)
except: urllib.request.urlretrieve(json_url, json_filename)

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

kinetics_classnames_to_id = {}
for k,v in kinetics_classnames.items():
    kinetics_classnames_to_id[str(k).replace('"', "")] = v

csv = pd.read_csv(dataset)
new_csv = []

for i, row in csv.iterrows():
    video = {}
    video['path'] = row['youtube_id']
    video['label'] = kinetics_classnames_to_id[row['label']]
    new_csv.append(video)

dataset = pd.DataFrame(new_csv)
dataset.to_csv(output, index=False)