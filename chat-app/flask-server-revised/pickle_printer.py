import pickle
import json
from pprint import pprint

file = '/Users/navid/Documents/rasa_practice/XAIBot_V1/chat-app/flask-server-revised/pickle_files/can_start.p'

with open(file, 'rb') as file:
    data = pickle.load(file)


# with open('brain.json', 'r') as file:
#     data = json.load(file)
    


pprint(data)