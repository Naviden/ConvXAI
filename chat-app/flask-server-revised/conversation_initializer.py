import pickle
import random
from DST import DST
from components import dataset_mapping, model_mapping, exp_mapping, profile_mapping
from aux_functions import *
from pprint import pprint
import sys


def conversation_initializer(user_input):
    memory = load_brain()
    print(f'\n\n<<<<<<<<<>>>>>>>>>>>1-memory: {memory}')
    
    # global memory
    # memory = load_brain()
    # print('*'*40)
    # print('memory is loaded:')
    # pprint(memory)
    # print('*'*40)

    missing, filling, mind_change, can_start, _dataset_waiting, _model_waiting, _profile_waiting = load_memory_files()
    print(f'\n\n2-memory: {memory}')
    dataset_obj = load_component('datasets')
    datasets = [pretty(item[1]['name']) for item in dataset_obj.items()]
    model_obj = load_component('models')
    models = [pretty(item[1]['name']) for item in model_obj.items()]
    profile_obj = load_component('profiles')
    profiles = [pretty(item[1]['name']) for item in profile_obj.items()]
    if can_start:
        # pass inp to RespoAlgo
        print('Chatbot is asking DST to take forward the conversation')
        NLU_model = pretty(memory['dataset'])
        return DST(user_input, NLU_model)
    else:
        print(f'\n\n3-memory: {memory}')
        if memory["dataset"]:
            # dataset is in memory
            if memory["model"]:
                # model is in memory

                memory["profile"] = ugly(user_input)  # saving userprofile
                save_brain(memory)

                update_element('_profile_waiting', False)
                update_element('can_start', True)
                print('XX'* 40)

                model = memory["model"]
                dataset = memory["dataset"]

                ######## CHECK COMPATIBILITY ##############
                dataset_datatype = dataset_mapping[dataset]['data_type']
                model_datatype = model_mapping[model]['task']

                if not isinstance(dataset_datatype, list):
                    dataset_datatype = [dataset_datatype]

                shared_datatype = set(dataset_datatype) & set(model_datatype)
                print(
                    f'data_types supported by {dataset} dataset: {dataset_datatype}')
                print(
                    f'data_types supported by {model} model: {model_datatype}')

                # getting possible presentations for this profile
                profile = memory['profile']
                possible_presentations = set(
                    profile_mapping[profile]['presentation'])

                # if the blackbox exists
                
                # Create a list of explainers that we can use
                if len(shared_datatype) > 0:  # data and blackbox match together
                    explainers = exp_mapping
                    matched_explainers = []
                    for explainer in explainers.keys():
                        # available data types
                        av_dt = set(explainers[explainer]['data_type'])
                        explainer_presentation = set(
                            explainers[explainer]['presentation'])
                        if len(av_dt & shared_datatype) > 0 and len(possible_presentations & explainer_presentation) > 0:
                            matched_explainers.append(explainer)
                    with open('./pickle_files/available_explainers.p', 'wb') as file:
                        pickle.dump(matched_explainers, file)

                    if matched_explainers:
                        if nlu_model_exist(pretty(memory["dataset"])):
                            print(f'matched explainers --> {matched_explainers}')
                            #### MAKE BLACKBOX ####
                            
                            create_dataset(dataset)
                            save_model(dataset, model)
                            print('='*40)
                            print('all values have been initialized. The following data are being saved as brain:')
                            print(f'dataset --> {memory["dataset"]}')
                            print(f'model --> {memory["model"]}')
                            print(f'profile --> {memory["profile"]}')
                            print('='*40)
                            save_brain(memory)
                            
                            return {
                                "msg": f'Great! Now that I know what we\'re talking about you can ask me your questions :)',
                                "intention": "wants_answer",
                                "end_of_conversation": False,
                            }
                        else:
                            return {
                                "msg": f'Please first train an NLU model for {memory["dataset"]} dataset.',
                                "intention": "wants_answer",
                                "end_of_conversation": True,
                            }


                    else:
                        print(
                            f'state: There is no explainer for profile {profile}')
                        print(memory)
                        return {
                            "msg": f'Currently there is no explainer for your profile :(',
                            "intention": "wants_answer",
                            "end_of_conversation": True,
                        }

                else:  # no match between data and black box
                    print(f'state: data and black box cannot be matched')
                    print(f'memory -->{memory}')
                    return {
                        "msg": 'The model and the data you have chosen can\'t be used together :(',
                        "intention": "wants_answer",
                        "end_of_conversation": True,
                    }
                ###########################################

            else:
                update_element('_model_waiting', False)
                memory["model"] = ugly(user_input)
                save_brain(memory)
                update_element('_profile_waiting', True)
                print('\nmemory after model assigning:')
                print(memory)
                return {
                    "msg": "Which profile does describe you better?",
                    "options": profiles,
                    "intention": "wants_list",
                    "end_of_conversation": False,
                }

        # no dataset
        else:
            if _dataset_waiting:
                memory["dataset"] = ugly(user_input)
                save_brain(memory)
                update_element('_dataset_waiting', False)
                update_element('_model_waiting', True)
                print('\nmemory after dataset assigning:')
                print(memory)
                return {
                    "msg": "Which is the model you want to use?",
                    "options": models,
                    "intention": "wants_list",
                    "end_of_conversation": False,
                }
            else:
                update_element('_dataset_waiting', True)
                print('\nmemory before dataset assigning:')
                print(memory)
                greetings = ['Nice to meet you!', 'Hello there!', 'Greetings!']
                return {
                    "msg": f"{random.choice(greetings)}\nWhich is the dataset you need explanations for?",
                    "options": datasets,
                    "intention": "wants_list",
                    "end_of_conversation": False,
                }

if __name__ == '__main__':
    conversation_initializer(sys.argv[1])