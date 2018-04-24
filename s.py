# coding: utf-8
from flask import Flask, request, Response, send_file, render_template, redirect
from functools import wraps
import simplejson as json
app = Flask(__name__, static_folder="./static", template_folder="./template")
import pandas as pd
import random
import numpy

traid_traname = pd.read_csv("traid_traname.csv")
traid_traname.reset_index(inplace=False)
traid_size = [i for i in range(traid_traname.shape[0])]

def json_response(func):
    @wraps(func)
    def json_pack():
        res = func()
        if isinstance(res, dict):
            rr = json.dumps(res, ensure_ascii=False, ignore_nan=True)
            resp = Response(response=rr,
                            status=200,
                            mimetype="application/json")
            return resp
        else:
            return res

    return json_pack


def random_recommend(info):
    size = int(info['num'])
    return {'recommend': traid_traname.loc[random.sample(traid_size, size)]['s2'].tolist()}


def get_user_vec_by_history(history) -> dict:  # save memory
    v = {}
    for h in history:
        v[tid_sid[h['s1']]] = int(h['num'])
    return v


import pickle
with open("uid_sid.data", "rb") as f:
    uid_sid = pickle.load(f)

tid_sid = {}
tid_sname = traid_traname.set_index('s1')['s2'].to_dict()
song_name_id = {}
sid_tid = {}
with open("sid_seq.data", "rb") as f:
    t = pickle.load(f)
    for tid, sid in t.items():
        if type(tid) is not str:
            continue
        tid_sid[tid] = sid
        sid_tid[sid] = tid
        song_name_id[sid] = tid_sname[tid]


def get_user_songs(users):
    res = []
    for u in users:
        res.extend(list(uid_sid[u].keys()))
    return list(set(res))


def distance(v1, v2):
    keys = []
    keys.extend(list(v1.keys()))
    keys.extend(list(v2.keys()))
    keys = list(set(keys))
    v11 = [v1[i] if i in v1 else 0 for i in keys]
    v22 = [v2[i] if i in v2 else 0 for i in keys]
    dist = numpy.linalg.norm(numpy.array(v11) - numpy.array(v22))
    return dist


def get_user_dis(user_vec, k=3):
    res = {}
    for u, v in uid_sid.items():
        res[u] = distance(user_vec, v)
    res = list(sorted(res.items(), key=lambda x: x[1]))[:k]
    return [x[0] for x in res]


def knn_recommend(info):
    gender, age, country, signup, num, history = info['gender'], info['age'], info['country'], info['signup'], info['num'], info['history']
    user_vec = get_user_vec_by_history(history)
    users= get_user_dis(user_vec)
    songs = [s for s in get_user_songs(users) if s not in user_vec]
    result = [song_name_id[s] for s in songs]
    return {'recommend': result[:min(len(result), int(num))]}


AL = {
    'random': random_recommend,
    'knn': knn_recommend,
}


@app.route("/api/recommend", methods=['POST'])
@json_response
def recommend():
    j = request.get_json()
    algorithm = j['algorithm']
    return AL[algorithm](j)


@app.route("/api/random_history", methods=['POST'])
@json_response
def random_user_history():
    i = random.randint(0, len(uid_sid))
    data = uid_sid[i]
    res = []
    for sid, num in data.items():
        res.append({'num': num, 's1': sid_tid[sid], 's2': song_name_id[sid]})
    return {"history": res}


@app.route("/api/candidate", methods=['POST'])
@json_response
def get_candidate():
    k = request.get_json()['key']
    result = traid_traname[traid_traname['s2'].str.startswith(k)].head(20)
    result = result.T.to_dict().values()
    result = list(result)
    for x in result:
        x['num'] = 0
    return {'candidates': result}


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()