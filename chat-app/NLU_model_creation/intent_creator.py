import random
import os
import json
from generate_dataset import create_dataset
import pandas as pd
import time
data_path = '../flask-server-revised/json_files'


class Intent(object):
    def __init__(self, dataset_name):
        with open(f'./{data_path}/intents.json', 'r') as file:
            intent_mapping = json.load(file)

        with open(f'./{data_path}/entities.json', 'r') as file:
            self.entity_mapping = json.load(file)

        with open(f'./{data_path}/datasets.json', 'r') as file:
            dataset_mapping = json.load(file)
        # dataset_name = '20_news_group'
        self.dataset_name = dataset_name
        dataset = dataset_mapping[self.dataset_name]
        data_type = dataset['data_type']

        # creating intent file names
        self.intents = intent_mapping[data_type]['intents']
        self.files = []
        for intent in self.intents:
            self.files.append(
                f'./raw_training_data/general_templates/{intent}.txt')
        # print(files)
        self.num_records = dataset['num_records']
        self.labels = dataset['labels']
        self.features = dataset.get('feature_names', None)
        self.entities = []
        
        # lower = dataset['lower']
        # upper = dataset['upper']

    def make_new_value(self):
        data = create_dataset(self.dataset_name)
        data.to_csv(f'./datasets/{self.dataset_name}.csv', index=False)
        target_column = data[self.selected_feature]
        return random.choice(target_column)

    def make_label(self, prefix=None):
        lab = random.choice(self.labels)

        if prefix is None:
            return f'[{lab}](label)'
        else:
            return f'[{lab}]({prefix}_label)'

    def make_record(self):

        num = random.randint(0, self.num_records)
        return f'[instance {num}](record)'

    def make_token(self):

        token = random.choice(self.features)
        return f'[{token}](token)'

    def make_feature(self):

        feature = random.choice(self.features)
        self.selected_feature = feature
        return f'[{feature}](feature)'

    def make_value(self):

        vals_1 = [round(random.random(), 2) for x in range(100)]
        vals_2 = list(range(100))
        vals = vals_1 + vals_2
        num = random.choice(vals)
        return f'[{num}](new_val)'

    def question_maker(self, text):
        # text = text.lower()
        used_labels = []
        res = ''
        for word in text.split():
            to_add = word

            if word == 'INS':
                to_add = self.make_record()
                self.entities.append('INS')
            elif word == 'LAB':
                to_add = self.make_label()
                self.entities.append('LAB')
            elif word == 'FEAT':
                to_add = self.make_feature()
                self.entities.append('FEAT')
            elif word == 'TOKEN':
                to_add = self.make_token()
                self.entities.append('TOKEN')
            elif word == 'NEW_VAL':
                to_add = self.make_new_value()
                self.entities.append('NEW_VAL')
            elif word == 'ACT_LAB':
                temp = self.make_label(prefix='act')
                self.entities.append('ACT_LAB')
                while temp in used_labels:
                    temp = self.make_label(prefix='act')
                used_labels.append(temp)
                to_add = temp
            elif word == 'DES_LAB':
                temp = self.make_label(prefix='des')
                self.entities.append('DES_LAB')
                while temp in used_labels:
                    temp = self.make_label(prefix='des')
                used_labels.append(temp)
                to_add = temp
            # elif word == 'NEW_VAL':
            #     to_add = make_value(lower, upper)
            # self.entities.append('NEW_VAL')

            res += f' {to_add}'
        return res.strip().lower()

    def get_all_entities(self, text):
        # text = text.lower()
        res = ''
        used_labels = []
        for word in text.split():
            to_add = word

            if word == 'INS':
                to_add = self.make_record()
            elif word == 'LAB':
                to_add = self.make_label()
            elif word == 'FEAT':
                to_add = self.make_feature()
            elif word == 'TOKEN':
                to_add = self.make_token()
            elif word == 'ACT_LAB':
                temp = self.make_label(prefix='act')
                while temp in used_labels:
                    temp = self.make_label(prefix='act')
                used_labels.append(temp)
                to_add = temp
            elif word == 'DES_LAB':
                temp = self.make_label(prefix='des')
                while temp in used_labels:
                    temp = self.make_label(prefix='des')
                used_labels.append(temp)
                to_add = temp
            # elif word == 'NEW_VAL':
            #     to_add = make_value(lower, upper)

            res += f' {to_add}'
        return res.strip().lower()

    def make_intent(self, file):
        
        file_name = os.path.basename(file).split('.')[0]
        print(f'---- creating : {file_name}')
        with open(file, 'r') as f:
            lines = f.readlines()
        if len(lines) > 0:
            res = [f'- intent: {file_name}\n',
                   '  examples: |\n']
            for line in lines:
                if line != '\n':
                    res.append(f'    - {self.question_maker(line)}\n')

            return res

    def create_intent(self):

        data = []
        for file in self.files:
            data.append('\n')
            try:
                intent_data = self.make_intent(file)
                if intent_data is not None:
                    for item in intent_data:
                        data.append(item)
            except FileNotFoundError:
                pass

        with open(f'./created_intents/{self.dataset_name}_intents.yaml', 'w') as file:
            for line in data:
                file.write(line)

        # preparing parts to be pasted in the domain_base.nlu
        base_intents = ['greet',
                        'goodbye',
                        'affirm',
                        'deny',
                        'mood_great',
                        'mood_unhappy',
                        'bot_challenge', ]
        used_entities = list(set(self.entities))
        used_entities = [self.entity_mapping[i] for i in used_entities]
        used_intent_entities = ['intents:\n']
        used_intent_entities += [f'  - {i}\n' for i in base_intents]
        used_intent_entities += [f'  - {i}\n' for i in self.intents]
        used_intent_entities += ['\n']
        used_intent_entities += ['entities:\n']
        used_intent_entities += [f'  - {i}\n' for i in set(used_entities)]

        with open(f'./intents_entities/intent_entity_{self.dataset_name}.txt', 'w') as file:
            for line in used_intent_entities:
                file.write(line)


# intent_obj = Intent('20_news_group')
# intent_obj.create_intent()
