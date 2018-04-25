
# coding: utf-8

# In[1]:

import pandas as pd
import pickle
import numpy as np


# In[2]:

all_data = pd.read_csv("userid-timestamp-artid-artname-traid-traname.tsv", sep='\t', error_bad_lines=False)


# In[3]:

all_data.columns = "userid-timestamp-artid-artname-traid-traname".split("-")
all_data = all_data.dropna(axis=0,how='any')


# In[4]:

users = all_data['userid'].drop_duplicates()


# In[5]:

songs = all_data['traid'].drop_duplicates()


# In[6]:

seq = 0
user_id = {}
for i, u in users.iteritems():
    user_id[u] = seq
    seq+=1


# In[7]:

song_id = {}
seq = 0
for i, u in songs.iteritems():
    song_id[u] = seq
    seq += 1


# In[ ]:




# In[11]:

all_data['uid'] = all_data['userid'].apply(lambda x: user_id[x])
all_data['sid'] = all_data['traid'].apply(lambda x: song_id[x])


# In[12]:

t = all_data[['uid', 'sid']].to_dict()
uid_sid = {}
for seq, sid in t['sid'].items():
    uid = t['uid'][seq]
    if uid not in uid_sid:
        uid_sid[uid] = {}
    if sid not in uid_sid[uid]:
        uid_sid[uid][sid] = 0
    uid_sid[uid][sid] += 1


# In[16]:

tmp = pd.DataFrame()
tmp[['s1', 's2']] = all_data[['traid', 'traname']]
tmp = tmp.drop_duplicates()
tmp = tmp.dropna(axis=0,how='any')
tmp.to_csv("traid_traname.csv")


# In[17]:

with open("uid_sid.data", "wb") as f:
    pickle.dump(uid_sid, f)
with open("sid_seq.data", "wb") as f:
    pickle.dump(song_id, f)


# In[18]:

artists = all_data['artid'].drop_duplicates()
seq = 0
artist_id = {}
for i, u in artists.iteritems():
    artist_id[u] = seq
    seq+=1


# In[19]:

all_data['aid'] = all_data['artid'].apply(lambda x: artist_id[x])


# In[20]:

all_data.head(5)


# In[24]:

sid_aid = all_data[['aid', 'sid']].set_index('sid')['aid'].to_dict()


# In[26]:

with open("sid_aid.data", "wb") as f:
    pickle.dump(sid_aid, f)


# In[27]:

aid_artname = all_data[['aid', 'artname']].set_index('aid')['artname'].to_dict()
with open("aid_artname.data", "wb") as f:
    pickle.dump(aid_artname, f)


# In[13]:

dd = [[0 for i in range(len(song_id))] for j in range(len(uid_sid))]


# In[14]:

len(dd), len(dd[0])


# In[15]:

for uid, sids in uid_sid.items():
    for sid, n in sids.items():
        dd[uid][sid] = n


# In[16]:

with open("dd1.data", "wb") as f:
    pickle.dump(dd, f)

