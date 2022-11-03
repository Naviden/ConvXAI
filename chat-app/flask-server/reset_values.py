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
          'token': None}}

missing = []
filling = False
mind_change = False


def initialize():
    with open('missing.p', 'wb') as file:
        pickle.dump([], file)
    with open('filling.p', 'wb') as file:
        pickle.dump(False, file)
    with open('mind_change.p', 'wb') as file:
        pickle.dump(False, file)
    with open('can_start.p', 'wb') as file:
        pickle.dump(False, file)
    with open('_dataset_waiting.p', 'wb') as file:
        pickle.dump(False, file)
    with open('_model_waiting.p', 'wb') as file:
        pickle.dump(False, file)
    with open('_profile_waiting.p', 'wb') as file:
        pickle.dump(False, file)


def load_brain():
    with open('brain.json', 'r') as file:
        brain = json.load(file)
        return brain


def save_brain(data):
    with open('brain.json', 'w') as fp:
        json.dump(data, fp)


# save_brain(brain)
# brain = load_brain()

save_brain(brain)
initialize()
print('all files have been initialized')
