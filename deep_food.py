import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Reshape, merge
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization as BN
import numpy as np
import os
from PIL import Image
import glob 
import pickle
import sys
import msgpack
import msgpack_numpy as m; m.patch()
import numpy as np
import json
import re
img_width, img_height = 150, 150
train_data_dir = './danbooru.imgs'
validation_data_dir = './imgs'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50
result_dir = 'results'

def loader(th = None):
  Xs = []
  Ys = [] 
  Rs = []
  files = glob.glob('cookpad/imgs/*')
  if '--mini' in sys.argv:
    files = files[:500]
  for gi, name in enumerate(files):
    if th is not None and gi > th:
      break
    last_name = name.split('/')[-1].split('_')[0]
    hash_name = last_name.split(".")[0]
    json_dir  = "cookpad/json/{hash_name}.json.minify".format(hash_name=hash_name)

    print(json_dir)
    try:
      o = json.loads(open(json_dir).read())
    except Exception as e:
      print(e)
      continue
    materials   = o['material']
    r = {}
    r["name"]      = last_name
    r["materials"] = materials
    y = [0.]*2048
    for material in materials:
      y[material] = 1.0
    img = Image.open('{name}'.format(name=name))
    img = img.convert('RGB')
    arr   = np.array(img)
    Ys.append( y   )
    Xs.append( arr )
    Rs.append( r   )
  Xs = np.array(Xs)
  return Ys, Xs, Rs

from keras.applications.resnet50 import ResNet50
def build_model():
  input_tensor = Input(shape=(224, 224, 3))
  resnet_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

  dense  = Flatten()( \
             Dense(2048, activation='relu')( \
               BN()( \
	         resnet_model.layers[-1].output ) ) )
  result = Activation('sigmoid')( \
	            Dense(2048, activation="linear")(\
                 dense) )
  
  model = Model(inputs=resnet_model.input, outputs=result)
  for layer in model.layers[:139]: # default 179
    #print(layer)
    if 'BatchNormalization' in str(layer):
      ...
    else:
      layer.trainable = False
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model

def train():
  print('load lexical dataset...')
  Ys, Xs, Rs = loader()
  print('build model...')
  model = build_model()
  for i in range(100):
    model.fit(np.array(Xs), np.array(Ys), batch_size=16, nb_epoch=1 )
    if i%1 == 0:
      model.save('models/model%05d.model'%i)

def eval():
  item_index = pickle.loads(open("cookpad/item_index.pkl", "rb").read())
  index_items = { index:item for item, index in item_index.items()}
  model = build_model()
  model = load_model(sorted(glob.glob('models/*.model'))[-1]) 
  Ys, Xs, Rs = loader(th=10)
  for i in range(len(Xs)):
    result = model.predict(np.array([Xs[i]]) )
    ares   = [(index_items[index], w) for index, w in enumerate(result.tolist()[0]) ]
    print(Rs[i])
    for en, (item, w) in enumerate(sorted(ares, key=lambda x:x[1]*-1)[:10]):
      print(en, item, w)

def pred():
  item_index = pickle.loads(open("cookpad/item_index.pkl", "rb").read())
  index_items = { index:item for item, index in item_index.items()}
  model = build_model()
  model = load_model(sorted(glob.glob('models/*.model'))[-1]) 
  target_size = (224,224)
  dir_path = "to_pred/*"
  max_size = len(glob.glob(dir_path))
  for i, name in enumerate(glob.glob(dir_path)):
    print(i, max_size, name)
    try:
      img = Image.open(name)
    except OSError as e:
      continue
    w, h = img.size
    if w > h :
      blank = Image.new('RGB', (w, w))
    if w <= h :
      blank = Image.new('RGB', (h, h))
    blank.paste(img, (0, 0) )
    blank = blank.resize( target_size )
    print(blank.size)
    Xs = np.array([np.asanyarray(blank)])
    print(Xs.shape)
    result = model.predict(Xs)
    ares   = [(index_items[index], w) for index, w in enumerate(result.tolist()[0]) ]
    for en, (item, w) in enumerate(sorted(ares, key=lambda x:x[1]*-1)[:10]):
      print(en, item, w)
     


if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
  if '--eval' in sys.argv:
    eval()
  if '--pred' in sys.argv:
    pred()
