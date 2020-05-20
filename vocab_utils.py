#!/usr/bin/env python
# coding: utf-8

import pandas as pd

PATH = 'vocab/'

simple = pd.read_csv(PATH + 'simple.txt', sep=' |\t', 
                     engine='python', 
                     names=['The','noun','verb','number',
                            'grammaticality','id'])

nounpp = pd.read_csv(PATH + 'nounpp.txt', sep=' |\t|_', 
                     engine='python', 
                     names=['The', 'noun', 'preposition', 'the',
                            'pp_noun', 'verb', 'n_number',
                            'pp_number', 'grammaticality', 'id'])


# Construct nouns
n_singular = nounpp['noun'][nounpp['n_number'] == 'singular']        .drop_duplicates().reset_index(drop=True)

n_plural = nounpp['noun'][nounpp['n_number'] == 'plural']        .drop_duplicates().reset_index(drop=True)

n_frame = {'n_singular':n_singular, 'n_plural':n_plural}

nouns = pd.DataFrame(n_frame)


# Construct verbs
v_singular = nounpp['verb'][nounpp['n_number'] == 'singular']        [nounpp['grammaticality'] == 'correct']        .drop_duplicates().reset_index(drop=True)

v_plural = nounpp['verb'][nounpp['n_number'] == 'singular']        [nounpp['grammaticality'] == 'wrong']        .drop_duplicates().reset_index(drop=True)

v_frame = {'v_singular':v_singular, 'v_plural':v_plural}

verbs = pd.DataFrame(v_frame)


# Construct prepositional nouns
ppn_singular = nounpp['pp_noun'][nounpp['pp_number'] == 'singular']        .drop_duplicates().sort_values().reset_index(drop=True)

ppn_plural = nounpp['pp_noun'][nounpp['pp_number'] == 'plural']        .drop_duplicates().sort_values().reset_index(drop=True)

ppn_frame = {'ppn_singular':ppn_singular, 'ppn_plural':ppn_plural}

ppns = pd.DataFrame(ppn_frame)


# Construct prepositions
prepositions = nounpp['preposition'].drop_duplicates()


def get_nouns():
    return [(s,p) for s, p in zip(nouns['n_singular'],
                                  nouns['n_plural'])]

def get_verbs():
    return [(s,p) for s, p in zip(verbs['v_singular'], 
                                  verbs['v_plural'])]

def get_preposition_nouns():
    return [(s,p) for s, p in zip(ppns['ppn_singular'], 
                                  ppns['ppn_plural'])]

def get_prepositions():
    return prepositions.tolist()


def make_template(noun, preposition, ppn):
    return ' '.join([noun, preposition, 'the', ppn])

