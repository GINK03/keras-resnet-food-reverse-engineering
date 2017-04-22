import msgpack
import msgpack_numpy as m; m.patch()
import numpy as np
import glob
from PIL import Image
import sys
import glob
import MeCab
import json

def serialize():
  for gi, name in enumerate(glob.glob("cookpad/imgs/*")):
    print(gi, name)
    last_name = name.split('/')[-1]
    im  = Image.open(name)
    arr = np.asarray(im)
    serialized = msgpack.packb(arr, default=m.encode)
    open('serialized/{}.msg'.format(last_name), 'wb' ).write(serialized)

def food_freq_check():
  t = MeCab.Tagger("-Owakati")
  term_freq = {}
  for name in glob.glob("cookpad/json/*"):
    o = json.loads(open(name).read())
    for term in t.parse(o['title']).strip().split():
      if term_freq.get(term) is None: term_freq[term] = 0
      term_freq[term] += 1
  for term, freq in sorted(term_freq.items(), key=lambda x:x[1]*-1):
    print(term, freq)


if __name__ == '__main__':
  if '--serialize' in sys.argv:
    serialize()
  if '--food_freq' in sys.argv:
    food_freq_check()
