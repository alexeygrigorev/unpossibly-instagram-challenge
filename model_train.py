# coding: utf-8

import json
import pandas as pd

import os
import numpy as np

import pickle

from glob import glob
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.ensemble import ExtraTreesRegressor

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras import backend as K


# extract features from images

model = VGG16(weights='imagenet')

prev_layer = model.layers[-2]
dense = K.function(model.inputs, [prev_layer.output])


imgs = glob('./photos/*.jpg')

res = {}

for img_path in tqdm(imgs):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features, = dense([x])
    res[img_path] = features[0]

keys = list(res.keys())
imgs = [res[k] for k in keys]
imgs = np.array(imgs)

codes = [k[9:-4] for k in keys]


pca = PCA(n_components=128, random_state=1)
imgs_pca = pca.fit_transform(imgs)
imgs_pca = normalize(imgs_pca)

with open('pca.bin', 'wb') as f:
    pickle.dump(pca, f)


# read the training data

def unwrap(d):
    res = {}
    res['updated'] = d['updated']

    ann = d['annotations']

    entities = {}
    for e in ann.get('webDetection', {}).get('webEntities', []):
        if 'description' not in e:
            continue
        if 'score' not in e:
            continue

        desc = e['description'].lower()
        score = e['score']
        entities[desc] = score
    res['ann_entities'] = entities

    insta = d['instagram']
    res['insta_code'] = insta['code']
    res['insta_dimensions_h'] = insta['dimensions']['height']
    res['insta_dimensions_w'] = insta['dimensions']['width']
    res['insta_caption'] = insta.get('caption', '').lower()
    res['insta_comments_disabled'] = insta['comments_disabled']
    res['insta_comments'] = insta['comments']['count']
    res['insta_date'] = insta['date']
    res['insta_likes'] = insta['likes']['count']
    res['insta_owner'] = insta['owner']['id']
    res['insta_thumbnail_src'] = insta['thumbnail_src']
    res['insta_is_video'] = insta['is_video']
    res['insta_id'] = insta['id']
    res['insta_display_src'] = insta['display_src']

    return res

with open('dataset.json') as f:
    data = json.load(f)


df_data = []

for u in data:
    df_user = pd.DataFrame([unwrap(d) for d in u['posts']])
    df_user['username'] = u['username']
    df_data.append(df_user)
df_data = pd.concat(df_data).reset_index(drop=1)


codes_idx = {c: i for (i, c) in enumerate(codes)}
df_data['img_idx'] = df_data.insta_code.apply(lambda c: codes_idx.get(c, -1))
df_data = df_data[df_data.img_idx != -1].reset_index(drop=1)


# normalization of target

means = df_data.groupby('username').insta_likes.mean()
stds = df_data.groupby('username').insta_likes.std()


with open('means_stds.bin', 'wb') as f:
    md = means.to_dict()
    sd = stds.to_dict()
    pickle.dump((md, sd), f)


means_series = means[df_data.username].reset_index(drop=1)
stds_series = stds[df_data.username].reset_index(drop=1)

y_norm = (df_data.insta_likes - means_series) / stds_series


# train a model

X = imgs_pca[df_data.img_idx.values]
y = y_norm.values

et_params = dict(
    n_estimators=200,
    criterion='mse',
    max_depth=30,
    min_samples_split=6,
    min_samples_leaf=6,
    max_features=50, 
    bootstrap=False, 
    n_jobs=-1,
    random_state=1
)

et = ExtraTreesRegressor(**et_params)
et.fit(X, y)

with open('et.bin', 'wb') as f:
    pickle.dump(et, f)
