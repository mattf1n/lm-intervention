import os
import csv
from vocab_utils import get_nouns, get_verbs

PATH = "vocab/"

def get_nouns2():
    noun2_list = []
    with open(os.path.join(PATH, "noun2.txt"), "r") as noun2_file:
        for noun2 in noun2_file:
            noun2s, noun2p = noun2.split()
            noun2_list.append((noun2s, noun2p))
    return noun2_list

def get_verbs2():
    verb2_list = []
    with open(os.path.join(PATH, "verb2.txt"), "r") as verb2_file:
        for verb2 in verb2_file:
            verb2s, verb2p = verb2.split()
            verb2_list.append((verb2s, verb2p))
    return verb2_list

def generate_rc(noun1_number, noun2_number, nouns1, nouns2, verbs1, verbs2, _id, complementizer=True):
    if complementizer:
        template = "The {} that the {} {} {}"
    else:
        template = "The {} the {} {} {}"

    out_list = []

    for (noun1s, noun1p) in nouns1:
        for (noun2s, noun2p) in nouns2:
            for (verb1s, verb1p) in verbs1:
                for (verb2s, verb2p) in verbs2:
                    noun1 = noun1s if noun1_number == "singular" else noun1p
                    noun2 = noun2s if noun2_number == "singular" else noun2p
                    verb2 = verb2s if noun2_number == "singular" else verb2p
                    correct_verb = verb1s if noun1_number == "singular" else verb1p
                    incorrect_verb = verb1p if noun1_number == "singular" else verb1s
                    label = noun1_number + "_" + noun2_number
                    out_list.append([template.format(noun1, noun2, verb2, correct_verb), \
                            label, "correct", "id"+str(_id)])
                    out_list.append([template.format(noun1, noun2, verb2, incorrect_verb), \
                            label, "wrong", "id"+str(_id)])
                    _id += 1

    return out_list, _id

def generate_within_rc(noun1_number, noun2_number, nouns1, nouns2, verbs1, _id, complementizer=True):
    if complementizer:
        template = "The {} that the {} {}"
    else:
        template = "The {} the {} {}"

    out_list = []

    for (noun1s, noun1p) in nouns1:
        for (noun2s, noun2p) in nouns2:
            for (verb1s, verb1p) in verbs1:
                noun1 = noun1s if noun1_number == "singular" else noun1p
                noun2 = noun2s if noun2_number == "singular" else noun2p
                correct_verb = verb1s if noun2_number == "singular" else verb1p
                incorrect_verb = verb1p if noun2_number == "singular" else verb1s
                label = noun1_number + "_" + noun2_number
                out_list.append([template.format(noun1, noun2, correct_verb), \
                        label, "correct", "id"+str(_id)])
                out_list.append([template.format(noun1, noun2, incorrect_verb), \
                        label, "wrong", "id"+str(_id)])
                _id += 1

    return out_list, _id


nouns1 = get_nouns()
nouns2 = get_nouns2()
verbs1 = get_verbs()
verbs2 = get_verbs2()

out_list = []
out_list_no_that = []
_id = 1
_id_no_that = 1

for noun1_number in ("singular", "plural"):
    for noun2_number in ("singular", "plural"):
        this_list, _id = generate_within_rc(noun1_number, noun2_number, \
                nouns1, nouns2, verbs1, _id)
        this_list_no_that, _id_no_that = generate_within_rc(noun1_number, noun2_number, \
                nouns1, nouns2, verbs1, _id_no_that, complementizer=False)
        out_list.extend(this_list)
        out_list_no_that.extend(this_list_no_that)

with open(os.path.join(PATH, "within_rc.txt"), "w") as rc_file:
    writer = csv.writer(rc_file, delimiter="\t")
    for out in out_list:
        writer.writerow(out)
with open(os.path.join(PATH, "within_rc_no_that.txt"), "w") as rc_file:
    writer = csv.writer(rc_file, delimiter="\t")
    for out_no_that in out_list_no_that:
        writer.writerow(out_no_that)
