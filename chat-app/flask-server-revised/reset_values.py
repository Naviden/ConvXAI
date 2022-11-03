import components
import json
import pickle

brain = {'dataset': None,
         'model': None,
         'profile': None,
         'intent': None,
         'entities':
         {'record': None,
          'label': None,
          'act_label': None,
          'des_label': None,
          'feature': None,
          'new_val': None,
          'token': None,
          'level': None}}

missing = []
filling = False
mind_change = False
json_path = './json_files'
pickle_path = './pickle_files'


def initialize():
    with open(f'{pickle_path}/missing.p', 'wb') as file:
        pickle.dump([], file)
    with open(f'{pickle_path}/filling.p', 'wb') as file:
        pickle.dump(False, file)
    with open(f'{pickle_path}/mind_change.p', 'wb') as file:
        pickle.dump(False, file)
    with open(f'{pickle_path}/can_start.p', 'wb') as file:
        pickle.dump(False, file)
    with open(f'{pickle_path}/_dataset_waiting.p', 'wb') as file:
        pickle.dump(False, file)
    with open(f'{pickle_path}/_model_waiting.p', 'wb') as file:
        pickle.dump(False, file)
    with open(f'{pickle_path}/_profile_waiting.p', 'wb') as file:
        pickle.dump(False, file)


def load_brain():
    with open(f'{json_path}/brain.json', 'r') as file:
        brain = json.load(file)
        return brain


def save_brain(data):
    with open(f'{json_path}/brain.json', 'w') as fp:
        json.dump(data, fp)


# save_brain(brain)
# brain = load_brain()

save_brain(brain)
initialize()
print('all files have been initialized')
