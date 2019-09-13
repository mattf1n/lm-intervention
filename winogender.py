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
            bergsma_pct_female[occupation] = float(row[1])
            bls_pct_female[occupation] = float(row[2])
    examples = []
    with open(os.path.join(path, 'winogender_templates_filtered.tsv')) as f:
        next(f, None)  # skip the headers
        row_pair = []
        for row in csv.reader(f, delimiter='\t'):
            row_pair.append(row)
            if len(row_pair) == 2:
                base_string1, substitute_female1, substitute_male1, candidate1, occupation1, answer1 = _parse_row(row_pair[0])
                base_string2, substitute_female2, substitute_male2, candidate2, occupation2, answer2 = _parse_row(row_pair[1])
                assert base_string1 == base_string2
                assert len(base_string1) > 0
                assert '$' not in base_string1
                assert substitute_female1 == substitute_female2
                assert substitute_male1 == substitute_male2
                assert len(candidate1) > 0 and len(candidate2) > 0 and candidate1 != candidate2
                if answer1 == 0:
                    continuation_occupation = candidate1
                    continuation_participant = candidate2
                else:
                    continuation_occupation = candidate2
                    continuation_participant = candidate1
                examples.append(
                    WinogenderExample(base_string1,
                                      substitute_female1,
                                      substitute_male1,
                                      continuation_occupation,
                                      continuation_participant,
                                      occupation1,
                                      bergsma_pct_female[occupation1],
                                      bls_pct_female[occupation1])
                )
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
            substitute_female = substitutes[0]
            substitute_male = substitutes[1]
            return base_string, substitute_female, substitute_male, candidate.strip(), occupation, int(answer)
    raise ValueError('Sentence does not contain pronoun type')


class WinogenderExample():

    def __init__(self, base_string, substitute_female, substitute_male, continuation_occupation, continuation_participant, occupation,
                 bergsma_pct_female, bls_pct_female):
        self.base_string = base_string
        self.substitute_female = substitute_female
        self.substitute_male = substitute_male
        self.continuation_occupation = continuation_occupation
        self.continuation_participant = continuation_participant
        self.occupation = occupation
        self.bergsma_pct_female = bergsma_pct_female
        self.bls_pct_female = bls_pct_female

    def __str__(self):
        return inspect.cleandoc(f"""
            base_string: {self.base_string}
            substitute_female: {self.substitute_female}
            substitute_male: {self.substitute_male}
            continuation_occupation: {self.continuation_occupation}
            continuation_participant: {self.continuation_participant}
            occupation: {self.occupation}
            bergsma_pct_female: {self.bergsma_pct_female}
            bls_pct_female: {self.bls_pct_female}
        """)

if __name__ == "__main__":
    for ex in load_examples():
        print()
        print(ex)