import json
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_20newsgroups
import pickle
import sklearn.ensemble
import numpy as np
import sklearn
import lime
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from pprint import pprint
from rasa.nlu.model import Interpreter
import warnings
import random
from dialogue_policy import make_response
from sklearn.model_selection import train_test_split
from components import exp_intents as slots


NLU_model = '20newsgroup'

# slots = {'list_features': {'must': ['record'],
#                            'alts': {}},
#          'what_if_subs': {'must': ['record', 'feature', 'new_val'],
#                           'alts': {}},
#          'why_not': {'must': ['record', 'des_label'],
#                      'alts': {}},
#          'why_this': {'must': ['record', 'label'],
#                       'alts': {'label': ['des_label']}},
#          'most_important_feature': {'must': ['record'],
#                                     'alts': {}},
#          'greet': {'must': {},
#                    'alts': {}},
#          'goodbye': {'must': {},
#                      'alts': {}},
#          'affirm': {'must': {},
#                     'alts': {}},
#          'deny': {'must': {},
#                   'alts': {}},
#          'nlu_fallback': {'must': {},
#                           'alts': {}},
#          'chitchat_general': {'must': {},
#                               'alts': {}
#                               },
#          'help': {'must': {},
#                   'alts': {}
#                   },
#          'human_handoff': {'must': {},
#                            'alts': {}
#                            }, }


def load_memory_files():
    with open('missing.p', 'rb') as file:
        missing = pickle.load(file)
    with open('filling.p', 'rb') as file:
        filling = pickle.load(file)
    with open('mind_change.p', 'rb') as file:
        mind_change = pickle.load(file)
    with open('can_start.p', 'rb') as file:
        can_start = pickle.load(file)
    with open('_dataset_waiting.p', 'rb') as file:
        _dataset_waiting = pickle.load(file)
    with open('_model_waiting.p', 'rb') as file:
        _model_waiting = pickle.load(file)
    with open('_profile_waiting.p', 'rb') as file:
        _profile_waiting = pickle.load(file)
    return missing, filling, mind_change, can_start, _dataset_waiting, _model_waiting, _profile_waiting


def save_missing(data):
    with open('missing.p', 'wb') as file:
        pickle.dump(data, file)


def empty_missing():
    global missing
    with open('missing.p', 'wb') as file:
        missing = pickle.dump([], file)


def save_filling(data):
    with open('filling.p', 'wb') as file:
        pickle.dump(data, file)


def save_mind_change(data):
    with open('mind_change.p', 'wb') as file:
        pickle.dump(data, file)


def load_brain():
    with open('brain.json', 'r') as file:
        brain = json.load(file)
        return brain


def save_brain(data):
    with open('brain.json', 'w') as fp:
        json.dump(data, fp)


def empty_brain():
    current_brain = load_brain()

    data = {'dataset': current_brain['dataset'],
            'model': current_brain['model'],
            'profile': current_brain['profile'],
            'intent': None,
            'entities':
            {'record': None,
             'label': None,
             'act_label': None,
             'des_label': None,
             'feature': None,
             'new_val': None}}
    with open('brain.json', 'w') as fp:
        json.dump(data, fp)


# memory = load_brain()
# missing, filling, mind_change = load_memory_files()


def memory_filler(extracted_info):
    global memory
    memory['intent'] = extracted_info['intent']['name']
    if extracted_info['entities'] is not None:
        for item in extracted_info['entities'].items():
            memory['entities'][item[0]] = item[1]
    save_brain(memory)


def missing_filler(nlu_resp):
    # entities
    global missing
    missing = load_memory_files()[0]
    detected_entities = nlu_resp['entities']
    detected_intent = get_intent(nlu_resp)
    print(f'Detcted intent --> {detected_intent}')
    if detected_entities is not None:
        print(f'Detcted entities --> {list(detected_entities.keys())}')

    # get musts of intent
    musts = slots[detected_intent]['entities']['must']
    print(f'must have entities --> {musts}')
    # get missings
    if detected_entities is not None and len(musts) != 0:
        missing_things = set(musts) - set(list(detected_entities.keys()))
        print(f'missing_things --> {missing_things}')
        extra = set(list(detected_entities.keys())) - set(musts)
    elif detected_entities is None and len(musts) != 0:
        missing_things = set(musts)
        extra = []
    elif detected_entities is not None and len(musts) == 0:
        missing_things = []
        extra = set(list(detected_entities.keys()))
    else:
        missing_things = []
        extra = []

    # if missing

    print(f'here missing --> {missing}')
    if len(missing_things) > 0:
        for miss in missing_things:
            # if it's possible to have alt
            if len(slots[detected_intent]['entities']['alts']) > 0:
                # get alternatives for missing
                miss_alts = slots[detected_intent]['entities']['alts'].get(miss, '@@@@')
                if not any([ex for ex in extra if ex in miss_alts]):
                    missing.append(miss)
            else:
                missing.append(miss)
    with open('missing.p', 'wb') as fp:
        pickle.dump(missing, fp)


def missing_ask(missing):
    pretty_name = {'des_label': 'your desired label',
    'act_label': 'actual label'}
    return f'What is the value for {pretty_name.get(missing[0], missing[0])} ?'


def missing_ask_again(missing):
    pretty_name = {'des_label': 'your desired label',
    'act_label': 'actual label'}
    return f'So...can you tell me what is the value for {pretty_name.get(missing[0], missing[0])} ?'


def rasa_output(text):
    rasa_model_path = f"../../RASA_NLU/models/{NLU_model}/nlu"

    interpreter = Interpreter.load(rasa_model_path, skip_validation=True)
    message = str(text).strip()
    result = interpreter.parse(message)
    return result


def clean_output(data):
    intent = get_intent(data)
    print(f'Identified intent: {intent}')

    ent = data.get('entities', 'None')
    if ent != 'None':
        print('\nExtracted entities:')
        for e in ent:
            print(f"\t{e['entity']}: {e['value']}")


def preprocess(text):

    text = text.strip().lower()
    repeated_punct = re.compile(
        r'([?.,/#!$%^&*;:{}=_`~()-])[?.,/#!$%^&*;:{}=_`~()-]+')
    punct_end = re.compile(".*\W$")
    repeated_space = re.compile(r'(\s)\s+')
    text = repeated_punct.sub(r'\1', text)
    if bool(re.match(punct_end, text)):
        end = text[-1]
        text = f'{text[:-1]} {end}'

    text = repeated_space.sub(r'\1', text)
    return text_to_num(text)


# memory = load_brain()
# missing, filling, mind_change, can_start = load_memory_files()[:4]

def load_component(name):
    with open(f'../../data/{name}.json', 'r') as file:
        data = json.load(file)
        return data

def ugly(text):
    return text.replace(' ', '_').lower()
def pretty(text):
    return text.replace('_', ' ')

def create_dataset(dataset):
    if dataset == '20_news_group':
        categories = ['alt.atheism', 'soc.religion.christian']
        newsgroups_train = fetch_20newsgroups(
            subset='train', categories=categories)
        newsgroups_test = fetch_20newsgroups(
            subset='test', categories=categories)
        class_names = ['atheism', 'christian']

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            lowercase=False)
        X_train = vectorizer.fit_transform(newsgroups_train.data)
        y_train = newsgroups_train.target
        X_test = newsgroups_test.data
        y_test = newsgroups_test.target
        #print(f'{dataset} --> X_test is:\n{X_test}')

        obj =  {'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'vectorizer': vectorizer,
                'class_names': class_names,
                'feature_names': None}

    elif dataset == 'iris': 
        iris = sklearn.datasets.load_iris()

        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)
        feature_names = iris.feature_names

        # prediction = model.predict(X_test)

        obj =  {'X_train': train,
                'y_train': labels_train,
                'X_test': test,
                'y_test': labels_test,
                'class_names': iris.target_names,
                'feature_names': feature_names}
    else:
        obj = {}

    with open(f'{dataset}_artifacts.p', 'wb') as file:
        pickle.dump(obj, file)


def save_model(dataset, model):
    with open(f'{dataset}_artifacts.p', 'rb') as file:
        data = pickle.load(file)
    if model == 'random_forest':
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
        rf.fit(data['X_train'], data['y_train'])

        with open(f'{model}.p', 'wb') as file:
            pickle.dump(rf, file)
        

def chatbot(user_input):
    """
    The main function which gets the initial values (datset, model and profile)
    from user and when everything is ready passes the input to the RespoAlgo
    """
    def update_element(element, value):
        with open(f'{element}.p', 'wb') as file:
            pickle.dump(value, file)

    global memory
    memory = load_brain()

    missing, filling, mind_change, can_start, _dataset_waiting, _model_waiting, _profile_waiting = load_memory_files()

    dataset_obj = load_component('datasets')
    datasets = [pretty(item[1]['name']) for item in dataset_obj.items()]
    model_obj = load_component('models')
    models = [pretty(item[1]['name']) for item in model_obj.items()]
    profile_obj = load_component('profiles')
    profiles = [pretty(item[1]['name']) for item in profile_obj.items()]
    if can_start:
        # pass inp to RespoAlgo
        print('Asking chatbot is asking main for answer')
        return main(user_input)
    else:
        if memory["dataset"]:
            # dataset is in memory
            if memory["model"]:
                # model is in memory
                memory["profile"] = ugly(user_input)
                save_brain(memory)
                update_element('_profile_waiting', False)
                update_element('can_start', True)

                model = memory["model"]
                dataset = memory["dataset"]
                #### MAKE BLACKBOX ####
                print('='*40)
                print('Creating dataset')
                create_dataset(dataset)
                print('='*40)
                save_model(dataset, model)
                print(memory)
                return {
                    "msg": f'Great! Now that I know what we\'re talking about you can ask me your questions :)',
                    "intention": "wants_answer",
                    "end_of_converstaion": False,
                }
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
                    "end_of_converstaion": False,
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
                    "end_of_converstaion": False,
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
                    "end_of_converstaion": False,
                }


def main(user_input):
    

    memory = load_brain()
    missing, filling, mind_change, can_start = load_memory_files()[:4]
    print('starting values:')
    print('-' * 40)
    print(f'memory: {memory}')
    print(f'missing: {missing}')
    print(f'filling: {filling}')
    print(f'mind_change: {mind_change}')
    print(f'can_start: {can_start}')
    print('-' * 40)

    general_intents = ['greet',
                       'affirm',
                       'goodbye',
                       'deny',
                       'nlu_fallback',
                       'chitchat_general',
                       'help',
                       'human_handoff',
                       'thank_you']

    # filling FALSE
    if not filling:
        message = rasa_output(preprocess(user_input))
        extracted_info = simplify_rasa_output(message)
        print(f'extracted info: {extracted_info}')
        # COMPATIBILITY CHECK
        if extracted_info['intent']['name'] not in general_intents:
            rejection_response = {
                        "msg": 'hmmm...you can\'t ask such question about this dataset :(',
                        "intention": 'wants_answer',
                        "end_of_converstaion": False,
                    }
            dataset_obj = load_component('datasets')
            dataset = memory['dataset']
            dataset = dataset_obj[ugly(dataset)]
            ddt = dataset['data_type']  # dataset data type
            print(f'ddt: {ddt}')
            
            model_obj = load_component('models')
            model = memory['model']
            model = model_obj[ugly(model)]
            mdt = model['task']  # model data type
            print(f'mdt: {mdt}')

            
            if ddt not in mdt:
                return rejection_response

            else:
                intent_obj = load_component('intents')
                possible_intents = intent_obj[ddt]['intents']
                if extracted_info['intent']['name'] not in possible_intents:
                    return rejection_response


        memory_filler(extracted_info)
        print(f'memory --> {memory}')
        missing_filler(extracted_info)
        missing = load_memory_files()[0]
        print(f'missing --> {missing}')
        if not missing:
            print('Detected --> no missing')
            empty_missing()
            # response = fake_response(extracted_info)
            memory = load_brain()
            resp = make_response(memory)
            print(f'resp before anything is: {resp}')

            if len(resp) == 3:
                print('resp has 3 items!')
                if resp[1] == 'wants_plot':
                    print('main is responding with wants plot')
                    print(f'the response which goes to chatbot is: {resp}')
                    return {
                                "intention": "wants_plot",
                                "url" : resp[0],
                                "end_of_converstaion" : False,
                            }
                print('Detected --> response has 3 items')
                response, intent, end = make_response(memory)
                empty_brain()
                return {
                    "msg": response,
                    "intention": intent,
                    "end_of_converstaion": end,
                }
            else:
                print('XXXXXXXX-->3')
                print(f'resp is:\n{make_response(memory)}')
                response, option, intent, end = make_response(memory)
                # return {
                #     "msg": "here is the list:",
                #     "options": option,
                #     "intention": "wants_list",
                #     "end_of_converstaion": False,
                # }
                return {
                    "msg": "here is the list:",
                    "options": ['AAAAAA', 'BBBBBB', 'CCCCCC', 'DDDDDDDD'],
                    "intention": "wants_list",
                    "end_of_converstaion": False,
                }

        else:
            print('Detected --> Not filling')
            save_filling(True)
            print(f'filling --> {filling}')
            return {
                "msg": missing_ask(missing),
                "intention": "wants_answer",
                "end_of_converstaion": False,
            }
    # filling TRUE
    else:
        print('Detected --> Filling mode')
        # MIND CHANGE --> FALSE
        if not mind_change:
            print('Detected --> mind_change=False')
            message = rasa_output(preprocess(user_input))
            extracted_info = simplify_rasa_output(message)
            print(f'extracted info --> {extracted_info}')

            # if user gives a short answer, most probably RASA is going to f**k it up
            # in this case we should override the detected intention by RASA
            if len(user_input.split(' ')) <= 2:
                print('Detected --> user input too short')
                print(f"** Replacing '{extracted_info['intent']['name']}' with '{memory['intent']}' **")
                new_intent = None
                extracted_info['intent'] = memory['intent']
            else:
                print('Detected --> user input NOT too short')
                new_intent = get_intent(extracted_info)
            # IF NO INTENT DETECTED
            if new_intent is None:
                print('Detected --> no intent is identified')
                # save user input in memory
                temp = memory
                temp['entities'][missing[0]] = extracted_info['text']
                save_brain(temp)
                # remove user input entity from missing
                temp = missing
                temp.remove(missing[0])
                save_missing(temp)
                # if there is any missing yet
                if temp:
                    print('Detected --> missing value')
                    return {
                        "msg": missing_ask(missing),
                        "intention": "wants_answer",
                        "end_of_converstaion": False,
                    }
                # There was just one missing and user filled it
                # XXXXXXXXXXXXXXXXXXXXXXXXXXXX
                else:
                    print('Detected --> no missing value')
                    save_filling(False)
                    # response = fake_response(extracted_info)

                    memory = load_brain()
                    response, intent, end = make_response(memory)
                    empty_brain()
                    if intent == 'wants_plot':
                        print('main is responding with wants plot')
                        return {
                                    "intention": "wants_plot",
                                    "url" : response,
                                    "end_of_converstaion" : False,
                                }
                    return {
                        "msg": response,
                        "intention": intent,
                        "end_of_converstaion": end,
                    }
            # if a new intent is detected
            else:
                print(f'the new intent detected --> {new_intent}')
                save_mind_change(True)

                return {
                    "msg": 'Do you want to change your initial question?',
                    "intention": "wants_answer",
                    "end_of_converstaion": False,
                }
        # if the user changed her mind
        else:
            print('Detected --> mind_change=True')
            message = rasa_output(preprocess(user_input))
            extracted_info = simplify_rasa_output(message)
            new_intent = get_intent(extracted_info)
            if new_intent not in ('affirm', 'deny'):
                print('Detected --> An answer which is not yes/no')
                empty_brain()
                empty_missing()
                save_filling(False)
                save_mind_change(False)

                return {
                    "msg": 'I didn\'t get what you mean. let\'s start from the beginning',
                    "intention": "wants_answer",
                    "end_of_converstaion": False,
                }
            # if intent either positive or negative
            else:
                print('Detected --> yes/no answer')
                if new_intent == 'affirm':
                    empty_brain()
                    empty_missing()
                    save_filling(False)
                    save_mind_change(False)
                    response = random.choice(['Sorry I couldn\'nt answer :(',
                                              'Ok so let\'s start from the beggining!',
                                              'OK! let\'s start from the beginning!'])

                    return {
                        "msg": response,
                        "intention": "wants_answer",
                        "end_of_converstaion": False,
                    }

                # intent is deny --> user doesn't want to start from the beginning
                else:
                    print('Detected --> Deny intent')

                    return {
                        "msg": missing_ask_again(missing),
                        "intention": "wants_answer",
                        "end_of_converstaion": False,
                    }


def simplify_rasa_output(rasa_output):
    res = {}
    res['intent'] = rasa_output['intent']
    # res['intent'] = rasa_output['intent']['name']
    res['entities'] = None
    if len(rasa_output['entities']) > 0:
        res['entities'] = {}
        for item in rasa_output['entities']:
            res['entities'][item['entity']] = item['value']
    res['text'] = rasa_output['text']
    return res


def get_intent(message):
    best_score = message['intent']['confidence']
    if best_score <= 0.5:
        return 'nlu_fallback'
    else:
        return message['intent']['name']


def get_entities(message):
    ents = message.get('entities', None)
    if ents is not None:
        return {ent['entity']: ent['value'] for ent in ents}


def text_to_num(text):
    text = text.split()
    result = []
    for t in text:
        res = t
        temp = text2int(t)
        if temp is not None:
            res = temp
        result.append(res)
    return ' '.join(result)


def text2int(textnum):
    numwords = {}
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", ]

    tens = ["", "", "twenty", "thirty", "forty",
            "fifty", "sixty", "seventy", "eighty", "ninety"]

    scales = ["hundred", "thousand", "million", "billion", "trillion"]

    numwords["and"] = (1, 0)
    for idx, word in enumerate(units):
        numwords[word] = (1, idx)
    for idx, word in enumerate(tens):
        numwords[word] = (1, idx * 10)
    for idx, word in enumerate(scales):
        numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word in numwords:

            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0

            return str(result + current)


def extract_num(text):
    return int(text.split()[-1])
