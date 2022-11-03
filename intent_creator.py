from glob import glob
import random
import os
import json

with open('./data/intents.json', 'r') as file:
    intent_mapping = json.load(file)

with open('./data/datasets.json', 'r') as file:
    dataset_mapping = json.load(file)

dataset_name = '20_news_group'
dataset = dataset_mapping[dataset_name]
data_type = dataset['data_type']

# creating intent file names
intents = intent_mapping[data_type]['intents']
files = []
for intent in intents:
    files.append(f'./raw_training_data/general_templates/{intent}.txt')
# print(files)
num_records = dataset['num_records']
labels = dataset['labels']
features = dataset.get('feature_names', None)
lower = dataset['lower']
upper = dataset['upper']

def make_label(labels, prefix=None):

    lab = random.choice(labels)
    if prefix is None:
        return f'[{lab}](label)'
    else:
        return f'[{lab}]({prefix}_label)'


def make_record(num_records):
    
    num = random.randint(0, num_records)
    return f'[instance {num}](record)'

def make_token(tokens):

    token = random.choice(tokens)
    return f'[{token}](token)'


def make_feature(features):

    feature = random.choice(features)
    return f'[{feature}](feature)'


def make_value(lower, upper):

    vals_1 = [round(random.random(), 2) for x in range(100)]
    vals_2 = list(range(100))
    vals = vals_1 + vals_2
    num = random.choice(vals)
    return f'[{num}](new_val)'


def question_maker(text):
    # text = text.lower()
    res = ''
    used_labels = []
    for word in text.split():
        to_add = word

        if word == 'INS':
            to_add = make_record(num_records)
        elif word == 'LAB':
            to_add = make_label(labels)
        elif word == 'FEAT':
            to_add = make_feature(features)
        elif word == 'TOKEN':
            to_add = make_token(features)
        elif word == 'ACT_LAB':
            temp = make_label(labels, prefix='act')
            while temp in used_labels:
                temp = make_label(labels, prefix='act')
            used_labels.append(temp)
            to_add = temp
        elif word == 'DES_LAB':
            temp = make_label(labels, prefix='des')
            while temp in used_labels:
                temp = make_label(labels, prefix='des')
            used_labels.append(temp)
            to_add = temp
        elif word == 'NEW_VAL':
            to_add = make_value(lower, upper)

        res += f' {to_add}'
    return res.strip().lower()


def make_intent(file):
    file_name = os.path.basename(file).split('.')[0]
    with open(file, 'r') as f:
        lines = f.readlines()
    if len(lines) > 0 :
        res = [f'- intent: {file_name}\n',
            '  examples: |\n']
        for line in lines:
            if line != '\n':
                res.append(f'    - {question_maker(line)}\n')

        return res


data = []
for file in files:
    data.append('\n')
    try:
        intent_data = make_intent(file)
        if intent_data is not None:
            for item in intent_data:
                data.append(item)
    except FileNotFoundError:
        pass

with open(f'./created_intents/{dataset_name}_intents.yaml', 'w') as file:
    for line in data:
        file.write(line)
print(f'Intent are created --> ./created_intents/{dataset_name}_intents.yaml')