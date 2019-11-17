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
    filename = 'professions_female_stereo.json'
  else:
    filename = 'professions_male_stereo.json'
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
  if gender == 'female':
    odds_base = df['candidate2_base_prob'] / df['candidate1_base_prob']
    odds_intervention = df['candidate2_prob'] / df['candidate1_prob']
  else:
    odds_base = df['candidate1_base_prob'] / df['candidate2_base_prob']
    odds_intervention = df['candidate1_prob'] / df['candidate2_prob']

  odds_ratio = odds_intervention / odds_base
  df['odds_ratio'] = odds_ratio
  return df

def sort_odds_obj(df):
  df['odds_diff'] = df['odds_ratio'].apply(lambda x: x-1)
  df['odds_abs'] = df['odds_diff'].apply(lambda x: abs(x))

  df_sorted = df.sort_values(by=['odds_abs'], ascending=False)
  return df_sorted

# get global list 
def get_all_contrib(templates, tokenizer, setting='test'):

  # get marginal contrib to empty set
  female_df = get_intervention_results(templates, tokenizer, gender='female')
  male_df = get_intervention_results(templates, tokenizer, gender='male')

  # compute odds ratio differently for each gender
  female_df = compute_odds_ratio(female_df, gender='female')
  male_df = compute_odds_ratio(male_df, gender='male')
  
  # merge and average
  df = pd.concat([female_df, male_df])
  df = df.groupby(['layer','neuron'], as_index=False).mean()
  df_sorted = sort_odds_obj(df)
  layer_list = df_sorted['layer'].values
  neuron_list = df_sorted['neuron'].values
  pickle.dump(layer_list, open("marg_contrib_layer.pickle", "wb" ))
  pickle.dump(neuron_list, open("marg_contrib_neuron.pickle", "wb" ))
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
    # calc odds ratio and odds-abs 
    df.append(df_template)
  return pd.concat(df)

def top_k_by_layer(layer, layer_list, neuron_list, k=50):
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

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    odd_abs_list.append(abs(df['odds_ratio'].mean()-1))
  
  pickle.dump(odd_abs_list, open("top_k" + str(layer) + ".pickle", "wb" ) )

def top_k(layer_list, neuron_list, k=50):
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
    odd_abs_list.append(abs(df['odds_ratio'].mean()-1))
  pickle.dump(odd_abs_list, open("top_k.pickle", "wb" ))


def greedy_by_layer(layer, k=50):
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

    # merge and average
    df = pd.concat([female_df, male_df])
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    neurons.append(df_sorted.head(1)['neuron'].values[0])
    odd_abs_list.append(df_sorted['odds_abs'].values[0])
  pickle.dump(odd_abs_list, open("greedy_" + str(layer) + ".pickle", "wb" ))
  pickle.dump(neurons, open("greedy_neurons_" + str(layer) + ".pickle", "wb" ))

def greedy(k=50):
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

    # merge and average
    df = pd.concat([female_df, male_df])
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    neurons.append(df_sorted.head(1)['neuron'].values[0])
    layers.append(df_sorted.head(1)['layer'].values[0])
    odd_abs_list.append(df_sorted['odds_abs'].values[0])

    # memory issue
    del df
    del female_df
    del male_df
    gc.collect()

    pickle.dump(odd_abs_list, open("greedy.pickle", "wb" ))
    pickle.dump(neurons, open("greedy_neurons.pickle", "wb" ))
    pickle.dump(layers, open("greedy_layers.pickle", "wb" ))


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


if __name__ == '__main__':
    ap = ArgumentParser(description="Run adversarial training for CIFAR.")
    ap.add_argument('--algo', type=str, choices=['topk', 'greedy', 'random_greedy', 'test'], default='topk')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--layer', type=int, default=-1)


    args = ap.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = Model(device='cuda')
    DEVICE = 'cuda'

    templates = get_template_list()

    k = args.k
    layer = args.layer
    if (args.algo == 'topk') and (layer == -1):
      layer_list, neuron_list = get_all_contrib(templates, tokenizer)
      # layer_list = pickle.load( open( 'marg_contrib_layer.pickle', "rb" )) 
      # neuron_list = pickle.load( open( 'marg_contrib_neuron.pickle', "rb" )) 
      top_k(layer_list, neuron_list, k)
    elif (args.algo == 'topk') and (layer != -1):
      layer_list, neuron_list = get_all_contrib(templates, tokenizer)
      # layer_list = pickle.load( open( 'marg_contrib_layer.pickle', "rb" )) 
      # neuron_list = pickle.load( open( 'marg_contrib_neuron.pickle', "rb" ))
      top_k_by_layer(layer, layer_list, neuron_list, k)
    elif (args.algo == 'greedy') and (layer == -1):
      greedy(k)
    elif (args.algo == 'greedy') and (layer != -1):
      greedy_by_layer(layer, k)
    elif (args.algo == 'test'):
      test()
    else:
      random_greedy_by_layer(layer, k)
