from intent_creator import Intent
import json

data_path = '../flask-server-revised/json_files'

with open(f'./{data_path}/datasets.json', 'r') as file:
    dataset_mapping = json.load(file)

available_datasets = dataset_mapping.keys()

for dataset in available_datasets:
    
    print(f'CREATED --> {dataset}')
    intent_obj = Intent(dataset)
    intent_obj.create_intent()
    
# for dataset in available_datasets:
#     try:
#         print(f'CREATED --> {dataset}')
#         intent_obj = Intent(dataset)
#         intent_obj.create_intent()
#     except:
#         print(f'ERROR --> {dataset}')