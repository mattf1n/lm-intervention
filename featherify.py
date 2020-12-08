import pandas as pd
import os, sys
from glob import glob
from tqdm import tqdm

PATH = sys.argv[1]

dtypes = {
  'word': 'category',
  'base_string': 'category',
  'alt_string1': 'category',
  'candidate1': 'category',
  'candidate2': 'category',
  'layer': 'int32',
  'neuron': 'int32',
}

files = list(filter(lambda x: x.endswith('.csv'), os.listdir(PATH)))
for f in tqdm(files):
  pd.read_csv(PATH + f, dtype=dtypes)\
      .to_feather(PATH + 'feathers/' + f.replace('csv','feather'))
