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

# missing = []
# filling = False
# mind_change = False
NLU_model = '20newsgroup'

slots = {'A': {'must': ['record', 'label'],
               'alts': {'label': ['act_label', 'des_label']}},
         'B': {'must': ['record', 'feature'],
               'alts': {}},
         'C': {'must': ['new_val', 'record', 'des_label'],
               'alts': {'des_label': ['label']}}}

slots = {'list_features': {'must': ['record'],
                           'alts': {}},
         'what_if_subs': {'must': ['record', 'feature', 'new_val'],
                          'alts': {}},
         'why_not': {'must': ['record', 'des_label'],
                     'alts': {}},
         'why_this': {'must': ['record', 'label'],
                      'alts': {}},
         'most_important_feature': {'must': ['record'],
                                    'alts': {}},
         'greet': {'must': {},
                   'alts': {}},
         'goodbye': {'must': {},
                     'alts': {}},
         'affirm': {'must': {},
                    'alts': {}},
         'deny': {'must': {},
                  'alts': {}},
         }


def initialize_files():
    with open('missing.p', 'rb') as file:
        missing = pickle.load(file)
    with open('filling.p', 'rb') as file:
        filling = pickle.load(file)
    with open('mind_change.p', 'rb') as file:
        mind_change = pickle.load(file)
    return missing, filling, mind_change


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
    data = {'intent': None,
            'entities':
            {'record': None,
             'label': None,
             'act_label': None,
             'des_label': None,
             'feature': None,
             'new_val': None}}
    with open('brain.json', 'w') as fp:
        json.dump(data, fp)


memory = load_brain()
missing, filling, mind_change = initialize_files()


def memory_filler(extracted_info):
    global memory
    memory['intent'] = extracted_info['intent']
    if extracted_info['entities'] is not None :
        for item in extracted_info['entities'].items():
            memory['entities'][item[0]] = item[1]
    save_brain(memory)


def fake_response(extracted_inf):
    return 'A fancy and shiny response'


def missing_filler(nlu_resp):
    # entities
    detected_entities = nlu_resp['entities']
    detected_intent = get_intent(nlu_resp)

    # get musts of intent
    musts = slots[detected_intent]['must']
    # get missings
    if detected_entities is not None and len(musts) != 0:
        missing_things = set(musts) - set(detected_entities)
        extra = set(detected_entities) - set(musts)
    elif detected_entities is None and len(musts) != 0:
        missing_things = set(musts) 
        extra = []
    elif detected_entities is not None and len(musts) ==  0:
        missing_things = []
        extra = set(detected_entities)
    else:
        missing_things = []
        extra = []


    # if missing
    global missing
    if len(missing_things) > 0:
        # if it's possible to have alt
        if len(slots[detected_intent]['alts']) > 0:
            for miss in missing_things:
                # get alternatives for missing
                miss_alts = slots[detected_intent]['alts'].get(miss, '@@@@')
                if not any([ex for ex in extra if ex in miss_alts]):
                    missing.append(miss)
    with open('missing.p', 'wb') as fp:
        pickle.dump(missing, fp)


def missing_ask(missing):
    return f'What is the value for {missing[0]} ?'


def missing_ask_again(missing):
    return f'So...can you tell me what is the value for {missing[0]} ?'


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


def create_dataset(dataset):
    if dataset == '20newsgroup':
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

        return {'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'vectorizer': vectorizer,
                'class_names': class_names}

    elif dataset == 'iris':
        data = pd.read_csv('./blackbox_input/iris.csv')
        train, test = train_test_split(
            data, test_size=0.4, stratify=data['species'], random_state=42)
        class_names = ['setosa', 'versicolor', 'virginica']
        X_train = train[['sepal_length', 'sepal_width',
                         'petal_length', 'petal_width']]
        y_train = train.species
        X_test = test[['sepal_length', 'sepal_width',
                       'petal_length', 'petal_width']]
        y_test = test.species
        # prediction = model.predict(X_test)

        return {'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'class_names': class_names}


def blackbox(dataset, algorithm):
    if algorithm == 'random_forest':
        data = create_dataset(dataset)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        class_names = data['class_names']
        vectorizer = data.get('vectorizer', None)

        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
        rf.fit(X_train, y_train)
        c = make_pipeline(vectorizer, rf)
        predict_proba = c.predict_proba

        return {'X_test': X_test, 'predict_proba': predict_proba, 'class_names': class_names}


class LIME_explanations(object):
    def __init__(self, dataset, algorithm):
        bb_data = blackbox(dataset, algorithm)
        self.class_names = bb_data['class_names']
        self.X_test = bb_data['X_test']
        self.pred_proba = bb_data['predict_proba']

        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def predict(self, record):
        pass
        # get rf from the blackbox function
        # to get a new prediction you should pass the row as follows
        # in this case there are 4 features
        # vals = np.array([6.6, 3. , 4.4, 1.4]).reshape(1,-1)
        # predictions = rf.predict(vals)

    def explain_why(self, record):
        exp = self.explainer.explain_instance(
            self.X_test[record], self.pred_proba, num_features=6)
        exp = exp.as_list()

        base = f'The model predicted record {record}, mainly based on the presense of the following 6 words:'
        res = ''
        for e in exp:
            res += f'{e[0]}, '
        res = res[:-2]
        res0 = res.split(',')[:-1]
        res0 = ', '.join(res0)
        res1 = res.split(',')[-1].strip()
        res1 = f' and {res1}'
        res = res0 + res1
        exp = f'{base} {res}'

        return exp

    def most_important(self, record):
        exp = self.explainer.explain_instance(
            self.X_test[record], self.pred_proba, num_features=6)
        exp = exp.as_list()[0][0]

        exp = f'The most important feature leading to this decision is "{exp}"'

        return exp

def make_response(message):
    intent = get_intent(message)
    if intent in ('why_this', 'list_features'):# and check_components(message):
        lo = LIME_explanations('20newsgroup', 'random_forest')
        # entities = get_entities(message)
        entities = message['entities']
        record = extract_num(entities['record'])
        response = lo.explain_why(record)

    elif intent == 'most_important_feature' and check_components(message):
        lo = LIME_explanations('20newsgroup', 'random_forest')
        # entities = get_entities(message)
        entities = message['entities']
        record = extract_num(entities['record'])
        response = lo.most_important(record)

    elif intent == 'greet':
        response = random.choice(['Hi there!', 'Hello!', 'Hi!', 'Hey there!'])

    elif intent == 'goodbye':
        response = random.choice(
            ['Goodbye!', 'See you later!', 'Bye! see you later!'])

    elif intent == 'why_not':
        response = 'hmmm...seems you\'re looking for a contrastive explanantion...unfortunately at the moment I\'m not able to generate such an explanation :('

    elif intent == 'nlu_fallback' or intent == 'chitchat_general':
        resps = ['Sorry I didn\'t get what you said...could you rephrase it?',
        'Hmmm...can you repeat what you said ?',
        'Don\'t hate me but I didn\'t understand what you said :(',
        'Sorry but it seems I have no answer to that...can you rephrase it ?',
        'Sorry but I can\'t understand what you say']
        response = random.choice(resps)

    elif intent == 'help':
        response = """I\'m a chat-bot you can ask anything about the outcome of the blackbox model. Here are some examples:\n
                        - why the model classified instance number N as class X ?
                        - what are the most important features the model used for classifying the instance N ?
                        - what is the most important feature for classifying the instance N ?  """


    return response



def main(user_input):
    global memory
    global missing
    global filling
    global mind_change

    # filling FALSE
    if not filling:
        message = rasa_output(preprocess(user_input))
        extracted_info = simplify_rasa_output(message)
        # print(f'extracted info: {extracted_info}')
        memory_filler(extracted_info)
        # print(f'memory --> {memory}')
        missing_filler(extracted_info)
        print(f'filled missing --> {missing}')
        if not missing:
            empty_missing()
            # response = fake_response(extracted_info)
            response = make_response(extracted_info)
            empty_brain()
            return response
        else:
            save_filling(True)
            return missing_ask(missing)
    # filling TRUE
    else:
        # MIND CHANGE --> FALSE
        if not mind_change:
            message = rasa_output(preprocess(user_input))
            extracted_info = simplify_rasa_output(message)
            new_intent = extracted_info['intent']
            # IF NO INTENT DETECTED
            if new_intent is None:
                # save user input in memory
                temp = memory
                temp['entities'][missing[0]] = user_input['text']
                save_brain(temp)
                # remove user input entity from missing
                temp = missing
                temp.remove(missing[0])
                save_missing(temp)
                # if there is any missing yet
                if temp:
                    return missing_ask(missing)
                else:
                    save_filling(False)
                    # response = fake_response(extracted_info)
                    response = make_response(extracted_info)
                    empty_brain()
                    return response
            # if there a new intent is detected
            else:
                save_mind_change(True)
                return f'Do you want to change your initial question?'
        # if the user changed her mind
        else:
            message = rasa_output(preprocess(user_input))
            extracted_info = simplify_rasa_output(message)
            new_intent = extracted_info['intent']
            if new_intent not in ('affirm', 'deny'):
                empty_brain()
                empty_missing()
                save_filling(False)
                save_mind_change(False)
                return 'I didn\'t get what you mean. let\'s start from the beginning'
            # if intent either positive or negative
            else:
                if new_intent == 'affirm':
                    empty_brain()
                    empty_missing()
                    save_filling(False)
                    save_mind_change(False)
                    response = random.choice(['Sorry I couldn\'nt answer :(',
                                              'Ok so let\'s start from the beggining!',
                                              'OK! let\'s start from the beginning!'])
                    return response
                # intent is deny --> user doesn't want to start from the beginning
                else:
                    return missing_ask_again(missing)




def fake_nlu(inp):
    return inp

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
