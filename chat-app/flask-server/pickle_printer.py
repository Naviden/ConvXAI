import pickle
from glob import glob
import json
from pprint import pprint

files = glob('./*.p')

for file in files:
    name = file.split('/')[-1].split('.')[0]
    with open(file, 'rb') as file:
        data = pickle.load(file)
    print(f'{name}: {data}')

def load_brain():
    with open('brain.json', 'r') as file:
        brain = json.load(file)
        return brain

memory = load_brain()
pprint(memory)