import pickle

with open('filling.p', 'rb') as file:
    data = pickle.load(file)
print(data)