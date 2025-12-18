# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import os
import zipfile
import subprocess
from datetime import datetime

DATASET = 'ml-1m'
RAW_PATH = os.path.join('./', DATASET)

# # Download data if not exists
# if not os.path.exists(RAW_PATH):
#     subprocess.call('mkdir ' + RAW_PATH, shell=True)
# if not os.path.exists(os.path.join(RAW_PATH, DATASET + '.zip')):
#     print('Downloading data into ' + RAW_PATH)
#     subprocess.call(
#         'wget http://files.grouplens.org/datasets/movielens/{}.zip -O {}'
#         .format(DATASET, os.path.join(RAW_PATH, DATASET + '.zip')), shell=True)
#     print('Unzip files...')
#     with zipfile.ZipFile(os.path.join(RAW_PATH, DATASET + '.zip'), 'r') as f:
#         for file in f.namelist():
#             print("Extract %s" % (file))
#             f.extract(file, RAW_PATH)

# Read interaction data
interactions = []
user_freq, item_freq = dict(), dict()
file = os.path.join(RAW_PATH, "ratings.dat")
with open(file) as F:
    for line in F:
        line = line.strip().split("::")
        uid, iid, rating, time = line[0], line[1], float(line[2]), float(line[3])
        if rating >= 4:
            label = 1
        else:
            label = 0
        interactions.append([uid, time, iid, label])
        if int(label) == 1:
            user_freq[uid] = user_freq.get(uid, 0) + 1
            item_freq[iid] = item_freq.get(iid, 0) + 1

# 5-core filtering
while True:
    select_uid, select_iid = [], []
    for u in user_freq:
        if user_freq[u] >= 5:
            select_uid.append(u)
    for i in item_freq:
        if item_freq[i] >= 5:
            select_iid.append(i)
    print("User: %d/%d, Item: %d/%d" % (len(select_uid), len(user_freq), len(select_iid), len(item_freq)))

    if len(select_uid) == len(user_freq) and len(select_iid) == len(item_freq):
        break

    select_uid = set(select_uid)
    select_iid = set(select_iid)
    user_freq, item_freq = dict(), dict()
    interactions_5core = []
    for line in interactions:
        uid, iid, label = line[0], line[2], line[-1]
        if uid in select_uid and iid in select_iid:
            interactions_5core.append(line)
            if int(label) == 1:
                user_freq[uid] = user_freq.get(uid, 0) + 1
                item_freq[iid] = item_freq.get(iid, 0) + 1
    interactions = interactions_5core

print("Selected Interactions: %d, Users: %d, Items: %d" % (len(interactions), len(select_uid), len(select_iid)))

# Construct DataFrame
interaction_df = pd.DataFrame(interactions, columns=["user_id", "time", "item_id", "label"])
interaction_df['time'] = interaction_df['time'].apply(lambda x: datetime.fromtimestamp(x))
min_date = interaction_df.time.min()
interaction_df['day'] = (interaction_df.time - min_date).apply(lambda x: x.days)

# Prepare data for Top-k Recommendation Task
interaction_pos = interaction_df.loc[interaction_df.label == 1].copy()
interaction_pos.rename(columns={'user_id': 'original_user_id', 'item_id': 'original_item_id'}, inplace=True)

# Split training, validation, and test sets.
split_time1 = int(interaction_pos.day.max() * 0.8)
train = interaction_pos.loc[interaction_pos.day <= split_time1].copy()
val_test = interaction_pos.loc[(interaction_pos.day > split_time1)].copy()
val_test.sort_values(by='time', inplace=True)
split_time2 = int(interaction_pos.day.max() * 0.9)
val = val_test.loc[val_test.day <= split_time2].copy()
test = val_test.loc[val_test.day > split_time2].copy()

# Delete user&item in validation&test sets that not exist in training set
train_u, train_i = set(train.original_user_id.unique()), set(train.original_item_id.unique())
val_sel = val.loc[(val.original_user_id.isin(train_u)) & (val.original_item_id.isin(train_i))].copy()
test_sel = test.loc[(test.original_user_id.isin(train_u)) & (test.original_item_id.isin(train_i))].copy()

# Assign ids for users and items
all_df = pd.concat([train, val_sel, test_sel], axis=0)
user2newid = dict(zip(sorted(all_df.original_user_id.unique()), range(1, all_df.original_user_id.nunique() + 1)))
item2newid = dict(zip(sorted(all_df.original_item_id.unique()), range(1, all_df.original_item_id.nunique() + 1)))

for df in [train, val_sel, test_sel, all_df]:
    df['user_id'] = df.original_user_id.apply(lambda x: user2newid[x])
    df['item_id'] = df.original_item_id.apply(lambda x: item2newid[x])

# Generate negative items
def generate_negative(data_df, all_items, clicked_item_set, random_seed, neg_item_num=99):
    np.random.seed(random_seed)
    neg_items = np.random.choice(all_items, (len(data_df), neg_item_num))
    for i, uid in enumerate(data_df['user_id'].values):
        user_clicked = clicked_item_set[uid]
        for j in range(len(neg_items[i])):
            while neg_items[i][j] in user_clicked or neg_items[i][j] in neg_items[i][:j]:
                neg_items[i][j] = np.random.choice(all_items, 1)
    return neg_items.tolist()

clicked_item_set = dict()
for user_id, seq_df in all_df.groupby('user_id'):
    clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())
all_items = all_df.item_id.unique()
val_sel['neg_items'] = generate_negative(val_sel, all_items, clicked_item_set, random_seed=1)
test_sel['neg_items'] = generate_negative(test_sel, all_items, clicked_item_set, random_seed=2)

# Save data
select_columns = ['user_id', 'item_id', 'time']
train[select_columns].to_csv('train.csv', sep='\t', index=False)
val_sel[select_columns + ['neg_items']].to_csv('dev.csv', sep='\t', index=False)
test_sel[select_columns + ['neg_items']].to_csv('test.csv', sep='\t', index=False)

print('Data preparation finished.')
