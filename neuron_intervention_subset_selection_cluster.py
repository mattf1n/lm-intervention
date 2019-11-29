# -*- coding: utf-8 -*-
from datetime import datetime

import torch
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import random
from functools import partial
from tqdm import tqdm # tqdm_notebook as tqdm
# from tqdm import tqdm_notebook
import math
import os
import sys
import pandas as pd
import statistics
import random
import pickle 
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import dask.dataframe as dd
import gc
from utils import batch, convert_results_to_pd
from experiment import Intervention, Model

np.random.seed(1)
torch.manual_seed(1)

def get_template_list():
    # Get list of all considered templates
    # "That" sentences are ours
    # "Because" sentences are a subset
    # from https://arxiv.org/pdf/1807.11714.pdf (Lu et al.)
    return ["The {} said that",
            "The {} yelled that",
            "The {} whispered that",
            "The {} wanted that",
            "The {} desired that",
            "The {} wished that",
            "The {} ate because",
            "The {} ran because",
            "The {} drove because.",
            "The {} slept because",
            "The {} cried because",
            "The {} laughed because",
            "The {} went home because",
            "The {} stayed up because",
            "The {} was fired because",
            "The {} was promoted because",
            "The {} yelled because"]

def get_intervention_types():
    return ['man_indirect',
            'woman_indirect']

def construct_interventions(base_sent, tokenizer, DEVICE, gender='female'):
  interventions = {}
  if gender == 'female':
    filename = 'professions_female_stereo2.json'
  else:
    filename = 'professions_male_stereo2.json'
  with open(filename, 'r') as f:
      all_word_count = 0
      used_word_count = 0
      for l in f:
          # there is only one line that eval's to an array
          for j in eval(l):
              all_word_count += 1
              biased_word = j[0]
              try: 
                interventions[biased_word] = Intervention(
                            tokenizer,
                            base_sent,
                            [biased_word, "man", "woman"],
                            ["he", "she"],
                            device=DEVICE)
                used_word_count += 1
              except:
                pass
                  # print("excepted {} due to tokenizer splitting.".format(
                  #     biased_word))

      print("Only used {}/{} neutral words due to tokenizer".format(
          used_word_count, all_word_count))
  return interventions

def compute_odds_ratio(df, gender='female'):

  # filter some stuff out
  # print(df.shape)
  df['profession'] = df['base_string'].apply(lambda s: s.split()[1])
  df['definitional'] = df['profession'].apply(get_stereotypicality)
  df = df[df['definitional'] < 0.75]
  # print(df.shape)

  if gender == 'female':
    odds_base = df['candidate1_base_prob'] / df['candidate2_base_prob']
    odds_intervention = df['candidate1_prob'] / df['candidate2_prob']
  else:
    odds_base = df['candidate2_base_prob'] / df['candidate1_base_prob']
    odds_intervention = df['candidate2_prob'] / df['candidate1_prob']

  odds_ratio = odds_intervention / odds_base
  df['odds_ratio'] = odds_ratio
  return df

def sort_odds_obj(df):
  df['odds_diff'] = df['odds_ratio'].apply(lambda x: x-1)

  df_sorted = df.sort_values(by=['odds_diff'], ascending=False)
  return df_sorted

def get_stereotypicality(vals):
        return abs(profession_stereotypicality[vals]['definitional'])

profession_stereotypicality = {}
with open("professions.json") as f:
    for l in f:
        for p in eval(l):
            profession_stereotypicality[p[0]] = {
                'stereotypicality': p[2],
                'definitional': p[1],
                'total': p[2]+p[1], 
                'max': max([p[2],p[1]], key=abs)}
# get global list 
def get_all_contrib(model_type, templates, tokenizer, out_dir='', setting='test'):

  # get marginal contrib to empty set
  female_df = get_intervention_results(templates, tokenizer, gender='female')
  male_df = get_intervention_results(templates, tokenizer, gender='male')
  gc.collect()

  # compute odds ratio differently for each gender
  female_df = compute_odds_ratio(female_df, gender='female')
  male_df = compute_odds_ratio(male_df, gender='male')
  
  # merge and average
  df = pd.concat([female_df, male_df])
  df = df.groupby(['layer','neuron'], as_index=False).mean()
  df_sorted = sort_odds_obj(df)
  layer_list = df_sorted['layer'].values
  neuron_list = df_sorted['neuron'].values

  marg_contrib = {}
  marg_contrib['layer'] = layer_list
  marg_contrib['neuron'] = neuron_list

  pickle.dump(marg_contrib, open(out_dir + "/marg_contrib_" + model_type + ".pickle", "wb" ))
  return layer_list, neuron_list

def get_intervention_results(templates, tokenizer, DEVICE='cuda', gender='female',
                             layers_to_adj=[], neurons_to_adj=[], intervention_loc='all',
                             df_layer=None, df_neuron=None):
  if gender == 'female':
    intervention_type = 'man_indirect'
  else:
    intervention_type = 'woman_indirect'
  df = []
  for template in templates:
    interventions = construct_interventions(template, tokenizer, DEVICE, gender)
    intervention_results = model.neuron_intervention_experiment(interventions, intervention_type, 
                                                                layers_to_adj=layers_to_adj, neurons_to_adj=neurons_to_adj,
                                                                intervention_loc=intervention_loc)
    df_template = convert_results_to_pd(interventions, intervention_results, df_layer, df_neuron)
    # print(template)
    # df_template.to_csv(template + '.csv')
    # calc odds ratio and odds-abs 
    df.append(df_template)
    gc.collect()
  return pd.concat(df)

def top_k_by_layer(model, model_type, tokenizer, templates, layer, layer_list, neuron_list, k=50, out_dir=''):
  layer_2_ind = np.where(layer_list == layer)[0]
  neuron_2 = neuron_list[layer_2_ind]
  
  odd_abs_list = []
  for i in range(k):
    print(i)
    temp_list = list(neuron_2[:i+1])

    neurons = [temp_list]

    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=len(temp_list)*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                          df_layer=layer, df_neuron=neurons[0])
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=len(temp_list)*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=layer, df_neuron=neurons[0])
    gc.collect()

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    odd_abs_list.append(df['odds_ratio'].mean()-1)
  
    pickle.dump(odd_abs_list, open(out_dir + "/topk_" + model_type + '_' + str(layer) + ".pickle", "wb" ) )

def top_k(model, model_type, tokenizer, templates, layer_list, neuron_list, k=50, out_dir=''):
  odd_abs_list = []

  for i in range(k):
    print(i)
    n_list = list(neuron_list[:i+1])
    l_list = list(layer_list[:i+1])

    neurons = [n_list]  
    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                          df_layer=l_list, df_neuron=neurons[0])
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=l_list, df_neuron=neurons[0])

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    odd_abs_list.append(df['odds_ratio'].mean()-1)

    pickle.dump(odd_abs_list, open(out_dir + "/topk_" + model_type + ".pickle", "wb" ))


def greedy_by_layer(model, model_type, tokenizer, templates, layer, k=50, out_dir=''):
  neurons = []
  odd_abs_list = []
  neurons = []

  for i in range(k):


    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                          df_layer=layer, df_neuron=None)
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                        df_layer=layer, df_neuron=None)

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')
    gc.collect()

    # merge and average
    df = pd.concat([female_df, male_df])
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    neurons.append(df_sorted.head(1)['neuron'].values[0])
    odd_abs_list.append(df_sorted['odds_diff'].values[0])

    greedy_res = {}
    greedy_res['neuron'] = neurons
    greedy_res['val'] = odd_abs_list

    pickle.dump(greedy_res, open(out_dir + "/greedy_" + model_type + "_" + str(layer) + ".pickle", "wb" ))

def greedy(model, model_type, tokenizer, templates, k=50, out_dir=''):
  neurons = []
  odd_abs_list = []
  layers = []

  greedy_filename = out_dir + "/greedy_" + model_type + ".pickle"

  if os.path.exists(greedy_filename):
    print('loading precomputed greedy values')
    res = pickle.load( open(greedy_filename, "rb" )) 
    odd_abs_list = res['val']
    layers = res['layer'] 
    neurons = res['neuron']
    k = k - len(odd_abs_list)
  else:
    neurons = []
    odd_abs_list = []
    layers = []

  for i in range(k):
    print(i)

    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=layers, neurons_to_adj=neurons, intervention_loc='all',
                                          df_layer=None, df_neuron=None)
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=layers, neurons_to_adj=neurons, intervention_loc='all',
                                        df_layer=None, df_neuron=None)

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')
    gc.collect()

    # merge and average
    df = pd.concat([female_df, male_df])
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    neurons.append(df_sorted.head(1)['neuron'].values[0])
    layers.append(df_sorted.head(1)['layer'].values[0])
    odd_abs_list.append(df_sorted['odds_diff'].values[0])

    # memory issue
    del df
    del female_df
    del male_df
    gc.collect()

    greedy_res = {}
    greedy_res['layer'] = layers
    greedy_res['neuron'] = neurons
    greedy_res['val'] = odd_abs_list

    pickle.dump(greedy_res, open(greedy_filename, "wb" ))

def dash(model, model_type, tokenizer, templates, k=50, out_dir=''):

  greedy_filename = out_dir + "/dash_" + model_type + ".pickle"

  df_list = []

  # total neuron list
  n_list = 13*list(range(768))
  l_list = []
  for i in range(-1, 12):
      l_list += 768*[i]
  n_list = np.array(n_list)
  l_list = np.array(l_list)

  for i in range(3):
    # randomly sample the number of elements
    set_of_indices = random.sample(list(range(9984)),k=k)
    neurons = n_list[set_of_indices]
    layers = l_list[set_of_indices]

    print(neurons)
    print(layers)

    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=layers, neurons_to_adj=neurons, intervention_loc='all',
                                          df_layer=None, df_neuron=None)
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=layers, neurons_to_adj=neurons, intervention_loc='all',
                                        df_layer=None, df_neuron=None)

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')
    gc.collect()

    # merge and average
    df = pd.concat([female_df, male_df])
    gc.collect()
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df = df.loc[:, ['layer','neuron', 'odds_ratio']]
    # df_sorted = sort_odds_obj(df)
    df.to_csv('round' + str(i) + '.csv')
    df_list.append(df)


  # merge and average
  df = pd.concat(df_list)
  df = df.groupby(['layer', 'neuron'], as_index=False).mean()
  df.to_csv('round_avg.csv')
  gc.collect()

  df_sorted = df.sort_values(by=['odds_ratio'], ascending=False)

  neurons_to_choose = np.array(df_sorted.head(500)['neuron'].values[0])
  layers_to_choose = np.array(df_sorted.head(500)['layer'].values[0])


  # memory issue
  del female_df
  del male_df
  gc.collect()

  dash_res = {}

  for i in range(10):
    dash_res[i] = {}
    
    # randomly select from this set 
    set_of_indices = random.sample(list(range(500)),k=k)
    neurons = neurons_to_choose[set_of_indices]
    layers = layers_to_choose[set_of_indices]


    dash_res[i]['layer'] = layers
    dash_res[i]['neuron'] = neurons

    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=layers, neurons_to_adj=neurons, intervention_loc='neuron',
                                          df_layer=None, df_neuron=None)
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=layers, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=None, df_neuron=None)

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')
  
    df = pd.concat([female_df, male_df])
    dash_res[i]['value'] = df['odds_ratio'].mean()-1
    print(dash_res)
  

    pickle.dump(dash_res, open(greedy_filename, "wb" ))

def random_greedy_by_layer(layer, k=50):
  neurons = []
  odd_abs_list = []
  neurons = []
  el_list = list(range(1,k+1))
  df = []
  for i in range(k):
    

    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                          df_layer=layer, df_neuron=None)
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                        df_layer=layer, df_neuron=None)

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    j = random.choice(el_list)
    neurons.append(df_sorted.head(j)['neuron'].values[-1])
    odd_abs_list.append(df_sorted.head(j)['odds_abs'].values[-1])

  pickle.dump(odd_abs_list, open("rand_greedy_" + str(layer) + ".pickle", "wb" ))
  pickle.dump(neurons, open("rand_greedy_neurons_" + str(layer) + ".pickle", "wb" ))

def test():

  odd_abs_list = []

  # get layer result
  for i in range(-1, 12):
    print(i)
    n_list = list(range(768))
    l_list = 768*[i]

    neurons = [n_list]  
    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                          df_layer=i, df_neuron=neurons[0])
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=i, df_neuron=neurons[0])

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    odd_abs_list.append(df['odds_ratio'].mean()-1)
    print('layer by layer')
    print(odd_abs_list)

  # n_list = 13*list(range(768))
  # l_list = []
  # for i in range(-1, 12):
  #     l_list += 768*[i]

  # neurons = [n_list]  
  # # get marginal contrib to empty set
  # female_df = get_intervention_results(templates, tokenizer, gender='female',
  #                                      layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
  #                                       df_layer=l_list, df_neuron=neurons[0])
  # male_df = get_intervention_results(templates, tokenizer, gender='male',
  #                                    layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
  #                                     df_layer=l_list, df_neuron=neurons[0])

  # # compute odds ratio differently for each gender
  # female_df = compute_odds_ratio(female_df, gender='female')
  # male_df = compute_odds_ratio(male_df, gender='male')

  # # merge and average
  # df = pd.concat([female_df, male_df])
  # # print('layer by layer')
  # # print(odd_abs_list)
  # print('total')
  # print(df['odds_ratio'].mean()-1)
  # pickle.dump(odd_abs_list, open("results/top_k.pickle", "wb" ))

def test2():
  layer_list, neuron_list = get_all_contrib('gpt2', templates, tokenizer, out_dir='result')

  # marg_contrib = pickle.load(open('result/marg_contrib_gpt2.pickle', "rb" )) 
  # layer_list = marg_contrib['layer']
  # neuron_list = marg_contrib['neuron']

  chunk_size = int(768/8)
  odd_obj_res = np.zeros((12,8))
  # odd_obj_res = pickle.load(open("result/topk_gpt2_chunk.pickle", "rb" ))
  for idx, layer in enumerate(list(range(-1, 11))):
    layer_2_ind = np.where(layer_list == layer)[0]
    neuron_2 = neuron_list[layer_2_ind]

    for i in range(8):
      if odd_obj_res[idx,i] == 0:
        chunk_size_layer = int(chunk_size*(i+1))
        print(chunk_size_layer)
        temp_list = list(neuron_2[:chunk_size_layer])
        print(len(temp_list))
        neurons = [np.sort(temp_list)]

        # get marginal contrib to empty set
        female_df = get_intervention_results(templates, tokenizer, gender='female',
                                             layers_to_adj=len(temp_list)*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                              df_layer=layer, df_neuron=neurons[0])
        male_df = get_intervention_results(templates, tokenizer, gender='male',
                                           layers_to_adj=len(temp_list)*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                            df_layer=layer, df_neuron=neurons[0])
       # compute odds ratio differently for each gender
        female_df = compute_odds_ratio(female_df, gender='female')
        male_df = compute_odds_ratio(male_df, gender='male')

        # merge and average
        df = pd.concat([female_df, male_df])
        odd_obj_res[idx,i] = df['odds_ratio'].mean()-1
        print(odd_obj_res)
        pickle.dump(odd_obj_res, open("result/topk_gpt2_chunk.pickle", "wb" ))
  
  odd_obj_tot = []
  
  for i in range(104):
    chunk_size_tot = int(chunk_size*(i+1))
    n_list = list(neuron_list[:chunk_size_tot])
    l_list = list(layer_list[:chunk_size_tot])

    neurons = [n_list]  
    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                          df_layer=l_list, df_neuron=neurons[0])
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=l_list, df_neuron=neurons[0])

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    odd_obj_tot.append(df['odds_ratio'].mean()-1)

    pickle.dump(odd_obj_tot, open("result/topk_gpt2_chunk_tot.pickle", "wb" ))


def test3():
  marg_contrib = pickle.load(open('result/marg_contrib_gpt2.pickle', "rb" )) 
  layer_list = marg_contrib['layer']
  neuron_list = marg_contrib['neuron']

  chunk_size = int(768/8)
  # odd_obj_res = np.zeros((11,8))
  odd_obj_res = pickle.load(open("result/topk_gpt2_chunk.pickle", "rb" ))
  layer = -1
  layer_2_ind = np.where(layer_list == layer)[0]
  neuron_2 = neuron_list[layer_2_ind]

  i = 7
  chunk_size_layer = int(chunk_size*(i+1))
  temp_list = list(neuron_2[:chunk_size_layer])
  neurons = [temp_list]

  # get marginal contrib to empty set
  female_df = get_intervention_results(templates, tokenizer, gender='female',
                                       layers_to_adj=len(temp_list)*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=-1, df_neuron=neurons[0])
  male_df = get_intervention_results(templates, tokenizer, gender='male',
                                     layers_to_adj=len(temp_list)*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                      df_layer=-1, df_neuron=neurons[0])
 # compute odds ratio differently for each gender
  female_df = compute_odds_ratio(female_df, gender='female')
  male_df = compute_odds_ratio(male_df, gender='male')

  # merge and average
  df = pd.concat([female_df, male_df])
  unsorted_res = df['odds_ratio'].mean()-1


  n_list = list(range(768))
  l_list = 768*[-1]

  neurons = [n_list]
  #   # get marginal contrib to empty set
  # female_df = get_intervention_results(templates, tokenizer, gender='female',
  #                                      layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
  #                                       df_layer=l_list, df_neuron=neurons[0])
  # male_df = get_intervention_results(templates, tokenizer, gender='male',
  #                                    layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
  #                                     df_layer=l_list, df_neuron=neurons[0])

  # # compute odds ratio differently for each gender
  # female_df = compute_odds_ratio(female_df, gender='female')
  # male_df = compute_odds_ratio(male_df, gender='male')

  # # merge and average
  # df = pd.concat([female_df, male_df])
  # sorted_res = df['odds_ratio'].mean()-1

  # get marginal contrib to empty set
  female_df = get_intervention_results(templates, tokenizer, gender='female',
                                       layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=-1, df_neuron=neurons[0])
  male_df = get_intervention_results(templates, tokenizer, gender='male',
                                     layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                      df_layer=-1, df_neuron=neurons[0])

  # compute odds ratio differently for each gender
  female_df = compute_odds_ratio(female_df, gender='female')
  male_df = compute_odds_ratio(male_df, gender='male')

  # merge and average
  df = pd.concat([female_df, male_df])
  sorted_res2 = df['odds_ratio'].mean()-1

  # print('unsorted')
  # print(unsorted_res)
  # print('sorted')
  # print(sorted_res)
  print(sorted_res2)

        # pickle.dump(odd_obj_res, open("result/topk_gpt2_chunk.pickle", "wb" ))
def test4():
  marg_contrib = pickle.load(open('result/marg_contrib_gpt2.pickle', "rb" )) 
  layer_list = marg_contrib['layer']
  neuron_list = marg_contrib['neuron']
  chunk_size_tot = 9984
  n_list = list(neuron_list[:chunk_size_tot])
  l_list = list(layer_list[:chunk_size_tot])

  neurons = [n_list]  
  # get marginal contrib to empty set
  female_df = get_intervention_results(templates, tokenizer, gender='female',
                                       layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=l_list, df_neuron=neurons[0])
  male_df = get_intervention_results(templates, tokenizer, gender='male',
                                     layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                      df_layer=l_list, df_neuron=neurons[0])

  # compute odds ratio differently for each gender
  female_df = compute_odds_ratio(female_df, gender='female')
  male_df = compute_odds_ratio(male_df, gender='male')

  # merge and average
  df = pd.concat([female_df, male_df])
  print(df['odds_ratio'].mean()-1)

def test5():
# get marginal contrib to empty set
  female_df = get_intervention_results(templates, tokenizer, gender='female',
                                       layers_to_adj=1, neurons_to_adj=1, intervention_loc='all',
                                        df_layer=None, df_neuron=None)
  male_df = get_intervention_results(templates, tokenizer, gender='male',
                                     layers_to_adj=1, neurons_to_adj=1, intervention_loc='all',
                                      df_layer=None, df_neuron=None)

  # compute odds ratio differently for each gender
  female_df.to_csv('gr_female.csv')
  male_df.to_csv('gr_male.csv')
  
  gc.collect()

  # merge and average
  df = pd.concat([female_df, male_df])

if __name__ == '__main__':
    # if not len(sys.argv) == 6:
    #   print("USAGE: python ", sys.argv[0], "<model> <algo> <k> <layer> <out_dir>")

    # model_type = sys.argv[1]
    # algo = sys.argv[2] # greedy or topk
    # k = int(sys.argv[3]) # int 
    # layer = int(sys.argv[4])
    # out_dir = sys.argv[5] # dir to write results

    model_type = 'gpt2'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = Model(device='cuda', gpt2_version=model_type)
    DEVICE = 'cuda'

    templates = get_template_list()
    templates = ["The {} said that"]

    dash(model, model_type, tokenizer, templates, k=250, out_dir='results')

    # if algo == 'topk':
    #   # load marg contrib
    #   marg_contrib_filename = out_dir + "/marg_contrib_" + model_type + ".pickle"
    #   if os.path.exists(marg_contrib_filename):
    #     print('Using precomputed marginal contrib')
    #     marg_contrib = pickle.load(open(marg_contrib_filename, "rb" )) 
    #     layer_list = marg_contrib['layer']
    #     neuron_list = marg_contrib['neuron']
    #   else:
    #     layer_list, neuron_list = get_all_contrib(model_type, templates, tokenizer, out_dir=out_dir)

    #   # run corresponding algo
    #   if layer == -2:
    #     top_k(model, model_type, tokenizer, templates, layer_list, neuron_list, 50, out_dir=out_dir)
    #     for i in range(-1,12):
    #       top_k_by_layer(model, model_type, tokenizer, templates, layer, layer_list, neuron_list, k, out_dir=out_dir)
    #   else:
    #     top_k_by_layer(model, model_type, tokenizer, templates, layer, layer_list, neuron_list, k, out_dir=out_dir)
    # elif (algo == 'greedy') and (layer == -2):
    #   greedy(model, model_type, tokenizer, templates, k, out_dir=out_dir)
    # elif (algo == 'greedy') and (layer != -2):
    #   greedy_by_layer(model, model_type, tokenizer, templates, layer, k, out_dir=out_dir)
