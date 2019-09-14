import os
import csv
import inspect
import re


def load_dev_examples(path='winobias_data/'):
    return load_examples(path, 'dev')


def load_examples(path, split):
    print(f'Split: {split.upper()}')
    with open(os.path.join(path, 'female_occupations.txt')) as f:
        female_occupations = [row.strip() for row in f]
    with open(os.path.join(path, 'male_occupations.txt')) as f:
        male_occupations = [row.strip() for row in f]
    occupations = female_occupations + male_occupations

    with open(os.path.join(path, f'pro_stereotyped_type1.txt.{split}')) as f:
        examples = []
        row_pair = []
        skip_count = 0
        for row in f:
            row_pair.append(row)
            if len(row_pair) == 2:
                skip = False
                if row_pair[0].count('[') != 2 or row_pair[1].count('[') != 2: # Multiple pronouns
                    skip = True
                elif '[him]' in row_pair[0] + row_pair[1]: # Objective pronoun, almost always at end of sentence
                    skip = True
                else:
                    base_string1, substitutes1, continuation1, occupation1 = _parse_row(row_pair[0], occupations)
                    base_string2, substitutes2, continuation2, occupation2 = _parse_row(row_pair[1], occupations)
                    if base_string1 != base_string2 or substitutes1 != substitutes2:
                        skip = True
                if skip:
                    print('Skipping: ', row_pair)
                    skip_count += 1
                    row_pair = []
                    continue
                base_string = base_string1
                assert substitutes1 == substitutes2
                female_pronoun, male_pronoun = substitutes1
                assert len(continuation1) > 0 and len(continuation2) > 0 and continuation1 != continuation2
                assert len(occupation1) > 0 and len(occupation2) > 0 and occupation1 != occupation2
                if occupation1 in female_occupations:
                    female_occupation = occupation1
                    female_occupation_continuation = continuation1
                    male_occupation = occupation2
                    male_occupation_continuation = continuation2
                    assert occupation2 in male_occupations
                else:
                    male_occupation = occupation1
                    male_occupation_continuation = continuation1
                    female_occupation = occupation2
                    female_occupation_continuation = continuation2
                    assert occupation1 in male_occupations
                    assert occupation2 in female_occupations
                examples.append(WinobiasExample(base_string, female_pronoun, male_pronoun, female_occupation, male_occupation,
                 female_occupation_continuation, male_occupation_continuation))
                row_pair = []
        assert row_pair == []
    print(f'Loaded {len(examples)} pairs. Skipped {skip_count} pairs.')
    return examples


def _parse_row(row, occupations):
    _, sentence = row.strip().split(' ', 1)
    occupation = None
    for occ in occupations:
        if f'[the {occ.lower()}]' in sentence.lower():
            assert occupation is None
            occupation = occ
    assert occupation is not None

    pronoun_groups = [ # First element is female, second is male
        ('she', 'he'), # nominative
        ('her', 'his') # possessive
    ]

    num_matches = 0
    substitutes = None
    for pronouns in pronoun_groups:
        pattern = '|'.join(r'\[' + p + r'\]' for p in pronouns) # matches '[he]', '[she]', etc.
        pronoun_matches = re.findall(pattern, sentence)
        assert len(pronoun_matches) <= 1
        if pronoun_matches:
            num_matches += 1
            pronoun_match = pronoun_matches[0]
            context, continuation = sentence.split(pronoun_match)
            context = context.replace('[', '').replace(']', '')
            context = context.strip()
            assert '[' not in continuation
            continuation = continuation.strip()
            substitutes = pronouns
    assert num_matches == 1
    base_string = context + ' {}'

    return base_string, substitutes, continuation, occupation


class WinobiasExample():

    def __init__(self, base_string, female_pronoun, male_pronoun, female_occupation, male_occupation,
                 female_occupation_continuation, male_occupation_continuation):
        self.base_string = base_string
        self.female_pronoun = female_pronoun
        self.male_pronoun = male_pronoun
        self.female_occupation = female_occupation
        self.male_occupation = male_occupation
        self.female_occupation_continuation = female_occupation_continuation
        self.male_occupation_continuation = male_occupation_continuation

    def __str__(self):
        return inspect.cleandoc(f"""
            base_string: {self.base_string}
            female_pronoun: {self.female_pronoun}
            male_pronoun: {self.male_pronoun}
            female_occupation: {self.female_occupation}
            male_occupation: {self.male_occupation}
            female_occupation_continuation: {self.female_occupation_continuation}
            male_occupation_continuation: {self.male_occupation_continuation}
        """)

    def __repr__(self):
        return str(self).replace('\n', ' ')


if __name__ == "__main__":
    for ex in load_dev_examples():
        print()
        print(ex)