import pickle

from pprint import pprint


file = 'wine_artifacts'
path = f'/Users/navid/Documents/rasa_practice/XAIBot_V1/chat-app/flask-server-revised/pickle_files/{file}.p'
with open(path, 'rb') as file:
    data = pickle.load(file)
# pprint(data)
print(data['X_test'])