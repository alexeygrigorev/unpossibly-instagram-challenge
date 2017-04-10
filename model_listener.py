# coding: utf-8

import json

import pickle
from sklearn.preprocessing import normalize

from PIL import Image
from io import BytesIO
import requests
import numpy as np

import requests
from time import time, sleep

auth = ('alexey.s.grigoriev@gmail.com', 'kxpeBSSU7KGElhAaec7brqApIpidk7ob')


print('loading models...')

with open('pca.bin', 'rb') as f:
    pca = pickle.load(f)

with open('et.bin', 'rb') as f:
    et = pickle.load(f)

with open('means_stds.bin', 'rb') as f:
    md, sd = pickle.load(f)

print('loading vgg...')

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras import backend as K

model = VGG16(weights='imagenet')

prev_layer = model.layers[-2]
dense = K.function(model.inputs, [prev_layer.output])


def read_url(pic_url, target_size=None):
    response = requests.get(pic_url)
    img = Image.open(BytesIO(response.content))
    if target_size is not None:
        img = img.resize(target_size)
    return img


def read_vgg_pca_features(pic_url):
    img = read_url(pic_url, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features, = dense([x])
    pca_features = pca.transform(features)
    pca_features = normalize(pca_features)
    return pca_features



live_api_url = 'http://challenges.instagram.unpossib.ly/api/live'

def get_latest_updates():
    resp = requests.get(live_api_url, auth=auth)
    return resp.json()

submission_url = 'http://challenges.instagram.unpossib.ly/api/submissions'

def submit(post_id, likes):
    data = {'post': post_id, 'likes': likes}
    response = requests.post(submission_url, json=data, auth=auth)
    resp_json = response.json()
    print(resp_json)

def process_account_update(acc):
    user = acc['username']
    if len(acc['posts']) > 0:
        print('new posts for user %s' % user)
    else:
        print('no new posts for user %s' % user)

    for post in acc['posts']:
        t0 = time()
        insta = post['instagram']
        post_id = insta['id']
        pic_url = insta['display_src']
        feat = read_vgg_pca_features(pic_url)
        pred, = et.predict(feat)

        likes = int(np.round(pred * sd[user] + md[user]))

        took = time() - t0
        print('predicted %d likes for post_id=%s, took %.3fs' % (likes, post_id, took))
        print('submitting...')
        submit(post_id, likes)
        print()

def process_update(rec):
    for acc in rec['accounts']:
        process_account_update(acc)


print('listening for updates...')

while True:
    try:
        updates = get_latest_updates()
        process_update(updates)
        print('sleeping for 3 minutes...')
        sleep(3 * 60)
    except Exception as e:
        print('got exception', str(e))
        print('ignoring it... and sleeping for 1 minute')
        sleep(1 * 60)
