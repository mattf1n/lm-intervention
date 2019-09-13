import os
import csv
import inspect

def load_examples(path='winogender_data/'):
    bergsma_pct_female = {}
    bls_pct_female = {}
    with open(os.path.join(path, 'winogender_occupation_stats.tsv')) as f:
        next(f, None)  # skip the headers
        for row in csv.reader(f, delimiter='\t'):
            occupation = row[0]
            bergsma_pct_female[occupation] = row[1]
            bls_pct_female[occupation] = row[2]
    examples = []
    with open(os.path.join(path, 'winogender_templates_filtered.tsv')) as f:
        next(f, None)  # skip the headers
        row_pair = []
        for row in csv.reader(f, delimiter='\t'):
            row_pair.append(row)
            if len(row_pair) == 2:
                base_string0, substitutes0, candidate0, occupation0, answer0 = _parse_row(row_pair[0])
                base_string1, substitutes1, candidate1, occupation1, answer1 = _parse_row(row_pair[1])
                assert base_string0 == base_string1
                assert len(base_string0) > 0
                assert '$' not in base_string0
                assert substitutes0 == substitutes1
                assert len(candidate0) > 0 and len(candidate1) > 0 and candidate0 != candidate1
                if answer0 == 0:
                    occupation_continuation = candidate0
                    participant_continuation = candidate1
                else:
                    occupation_continuation = candidate1
                    participant_continuation = candidate0
                examples.append(WinogenderExample(base_string0, substitutes0, occupation_continuation,
                                                  participant_continuation, occupation0,
                                                  bergsma_pct_female[occupation0], bls_pct_female[occupation0]))
                row_pair = []
    return examples


def _parse_row(row):
    occupation, participant, answer, sentence = row
    pronoun_to_substitutes = {
        '$NOM_PRONOUN': ('she', 'he'),
        '$POSS_PRONOUN': ('her', 'his')
    }
    for pronoun_type, substitutes in pronoun_to_substitutes.items():
        if pronoun_type in sentence:
            context, candidate = sentence.split(pronoun_type)
            base_string = context.replace('$OCCUPATION', occupation)
            base_string = base_string.replace('$PARTICIPANT', participant)
            base_string = base_string + '{}'
            return base_string, substitutes, candidate.strip(), occupation, answer
    raise ValueError('Sentence does not contain pronoun type')


class WinogenderExample():

    def __init__(self, base_string, substitutes, occupation_continuation, participant_continuation, occupation,
                 bergsma_pct_female, bls_pct_female):
        self.base_string = base_string
        self.substitutes = substitutes
        self.occupation_continuation = occupation_continuation
        self.participant_continuation = participant_continuation
        self.occupation = occupation
        self.bergsma_pct_female = bergsma_pct_female
        self.bls_pct_female = bls_pct_female

    def __str__(self):
        return inspect.cleandoc(f"""
            base_string: {self.base_string}
            substitutes: {self.substitutes}
            occupation_continuation: {self.occupation_continuation}
            participant_continuation: {self.participant_continuation}
            occupation: {self.occupation}
            bergsma_pct_female: {self.bergsma_pct_female}
            bls_pct_female: {self.bls_pct_female}
        """)

if __name__ == "__main__":
    for ex in load_examples():
        print()
        print(ex)