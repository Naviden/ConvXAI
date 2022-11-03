import re
from os import path
from glob import glob
from sklearn.datasets import fetch_20newsgroups, load_breast_cancer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import json
import sklearn
import pandas as pd
import numpy as np
from scipy.stats import skew
from components import exp_intents as slots
from sklearn.pipeline import make_pipeline
from sklearn import svm, tree, neighbors
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import statsmodels.api as sm

def load_memory_files():
    with open('./pickle_files/missing.p', 'rb') as file:
        missing = pickle.load(file)
    with open('./pickle_files/filling.p', 'rb') as file:
        filling = pickle.load(file)
    with open('./pickle_files/mind_change.p', 'rb') as file:
        mind_change = pickle.load(file)
    with open('./pickle_files/can_start.p', 'rb') as file:
        can_start = pickle.load(file)
    with open('./pickle_files/_dataset_waiting.p', 'rb') as file:
        _dataset_waiting = pickle.load(file)
    with open('./pickle_files/_model_waiting.p', 'rb') as file:
        _model_waiting = pickle.load(file)
    with open('./pickle_files/_profile_waiting.p', 'rb') as file:
        _profile_waiting = pickle.load(file)
    return missing, filling, mind_change, can_start, _dataset_waiting, _model_waiting, _profile_waiting


def load_brain():
    with open('./json_files/brain.json', 'r') as file:
        brain = json.load(file)
        return brain


def save_brain(data):
    with open('./json_files/brain.json', 'w') as fp:
        json.dump(data, fp)


def load_component(name):
    with open(f'./json_files/{name}.json', 'r') as file:
        data = json.load(file)
        return data


def ugly(text):
    return text.replace(' ', '_').lower()


def pretty(text):
    return text.replace('_', ' ')


def nlu_dir_name(text):
    return text.replace('_', '').lower()


def create_dataset(dataset):
    pickle_files = glob('./pickle_files/*.p')
    pickle_files = [path.split('/')[-1].split('.')[0] for path in pickle_files]
    if dataset not in pickle_files:
        print('='*40)
        print('Preparing dataset')
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

            obj = {'X_train': X_train,
                   'y_train': y_train,
                   'X_test': X_test,
                   'y_test': y_test,
                   'vectorizer': vectorizer,
                   'class_names': class_names,
                   'feature_names': None}

        elif dataset == 'iris':
            iris = sklearn.datasets.load_iris(as_frame=True)

            train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
                iris.data, iris.target, train_size=0.80,  random_state=42)
            feature_names = iris.feature_names

            # prediction = model.predict(X_test)

            obj = {'X_train': train,
                   'y_train': labels_train,
                   'X_test': test,
                   'y_test': labels_test,
                   'class_names': iris.target_names,
                   'feature_names': feature_names}

        elif dataset == 'wine':
            wine = sklearn.datasets.load_wine(as_frame=True)

            train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
                wine.data, wine.target, train_size=0.80, random_state=42)
            feature_names = wine.feature_names

            # prediction = model.predict(X_test)

            obj = {'X_train': train,
                   'y_train': labels_train,
                   'X_test': test,
                   'y_test': labels_test,
                   'class_names': wine.target_names,
                   'feature_names': feature_names}

        elif dataset == 'abalone':

            df = pd.read_csv('abalone.csv')
            nf = df.select_dtypes(include=[np.number]).columns
            cf = df.select_dtypes(include=[np.object]).columns

            # sending all numericalfeatures and omitting nan values
            skew_list = skew(df[nf], nan_policy='omit')
            skew_list_df = pd.concat([pd.DataFrame(nf, columns=['Features']), pd.DataFrame(
                skew_list, columns=['Skewness'])], axis=1)
            mv_df = df.isnull().sum().sort_values(ascending=False)
            pmv_df = (mv_df/len(df)) * 100
            missing_df = pd.concat([mv_df, pmv_df], axis=1, keys=[
                                   'Missing Values', '% Missing'])

            df['Age'] = df['Class_number_of_rings'] + 1.5
            df['Sex'] = LabelEncoder().fit_transform(df['Sex'].tolist())

            Xtrain = df.drop(['Class_number_of_rings', 'Sex'], axis=1)
            Ytrain = df['Sex']

            train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
                Xtrain, Ytrain, train_size=0.80, random_state=42)

            obj = {'X_train': train,
                   'y_train': labels_train,
                   'X_test': test,
                   'y_test': labels_test,
                   'class_names': [0, 1, 2],
                   'feature_names': ['Length', 'Diameter', 'Height', 'Whole_weight',
                                     'Shucked_weight', 'Viscera_weight', 'Shell_weight',
                                     'Sex_0', 'Sex_1', 'Sex_2']}

        elif dataset == 'breast_cancer':
            data = load_breast_cancer()
            X = pd.DataFrame(data['data'], columns=data['feature_names'])
            y = pd.Series(data['target'])
            train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
                X, y, test_size=0.2, random_state=42)

            obj = {'X_train': train,
                   'y_train': labels_train,
                   'X_test': test,
                   'y_test': labels_test,
                   'class_names': ['0', '1'],
                   'feature_names': ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                                     'mean smoothness', 'mean compactness', 'mean concavity',
                                     'mean concave points', 'mean symmetry', 'mean fractal dimension',
                                     'radius error', 'texture error', 'perimeter error', 'area error',
                                     'smoothness error', 'compactness error', 'concavity error',
                                     'concave points error', 'symmetry error',
                                     'fractal dimension error', 'worst radius', 'worst texture',
                                     'worst perimeter', 'worst area', 'worst smoothness',
                                     'worst compactness', 'worst concavity', 'worst concave points',
                                     'worst symmetry', 'worst fractal dimension']}

        elif dataset == 'diabetes':
            diabetes = sklearn.datasets.load_diabetes(as_frame=True)

            train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
                diabetes.data, diabetes.target, train_size=0.80, random_state=42)
            feature_names = diabetes.feature_names

            # prediction = model.predict(X_test)

            obj = {
                'X_train': train,
                'y_train': labels_train,
                'X_test': test,
                'y_test': labels_test,
                'feature_names': ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
            }
        elif dataset == 'bike_sharing':
            bike_sharing = pd.read_csv('./datasets/bike_sharing.csv')
            X = bike_sharing.drop(['casual', 'registered', 'count'], axis=1)
            y = bike_sharing['registered']

            train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
                X, y, train_size=0.80,  random_state=42)
            feature_names = ['workingday', 'temp', 'humidity', 'windspeed', 'casual', 'registered',
                             'count', 'time', 'month', 'season_2', 'season_3', 'season_4',
                             'weather_2', 'weather_3', 'weather_4']

            # prediction = model.predict(X_test)

            obj = {'X_train': train,
                   'y_train': labels_train,
                   'X_test': test,
                   'y_test': labels_test,
                   'class_names': None,
                   'feature_names': feature_names}

        else:
            obj = {}

        with open(f'./pickle_files/{dataset}_artifacts.p', 'wb') as file:
            pickle.dump(obj, file)
        print('='*40)


def save_model(dataset, model):
    pickle_files = glob('./pickle_files/*.p')
    pickle_files = [path.split('/')[-1].split('.')[0] for path in pickle_files]
    # if model not in pickle_files:
    print('='*40)
    print('Creating model')
    with open(f'./pickle_files/{dataset}_artifacts.p', 'rb') as file:
        data = pickle.load(file)

    if model == 'random_forest':
        blackbox = sklearn.ensemble.RandomForestClassifier(random_state=42,
                                                           n_estimators=500)

    if model == 'svm':
        blackbox = svm.SVC(gamma=0.001, C=100.,
                           random_state=42, probability=True)

    if model == 'decision_tree':
        blackbox = tree.DecisionTreeClassifier()

    if model == 'naive_bayes':
        blackbox = GaussianNB()

    if model == 'xgboost':
        blackbox = xgb.XGBClassifier(objective='reg:logistic', colsample_bytree=0.3, learning_rate=0.1,
                                     max_depth=5, alpha=10, n_estimators=10)

    if model == 'knn':
        blackbox = neighbors.KNeighborsClassifier(3, weights='distance')

    if model == 'random_forest_regressor':
        blackbox = sklearn.ensemble.RandomForestRegressor(random_state=42)


     # fitting data
    if model == 'naive_bayes':
        blackbox.fit(data['X_train'].values, data['y_train'])


    if model == 'ols':
        blackbox = sm.OLS( data['y_train'], data['X_train']).fit()
   
    else:
        blackbox.fit(data['X_train'], data['y_train'])

    with open(f'./pickle_files/{model}.p', 'wb') as file:
        pickle.dump(blackbox, file)
    print('='*40)

def predict(model, dataset):
    with open(f'./pickle_files/{model}.p', 'wb') as file:
        trained_model = pickle.load(file)

    with open(f'./pickle_files/{dataset}_artifacts.p', 'rb') as file:
        data = pickle.load(file)
    

    pass

def save_missing(data):
    with open('./pickle_files/missing.p', 'wb') as file:
        pickle.dump(data, file)


def empty_missing():
    with open('./pickle_files/missing.p', 'wb') as file:
        pickle.dump([], file)


def save_filling(data):
    with open('./pickle_files/filling.p', 'wb') as file:
        pickle.dump(data, file)


def save_mind_change(data):
    with open('./pickle_files/mind_change.p', 'wb') as file:
        pickle.dump(data, file)


def memory_filler(extracted_info):
    #global memory
    memory = load_brain()
    memory['intent'] = extracted_info['intent']['name']
    if extracted_info['entities'] is not None:
        for item in extracted_info['entities'].items():
            memory['entities'][item[0]] = item[1]
    save_brain(memory)


def missing_filler(nlu_resp):
    # entities
    #global missing
    missing = load_memory_files()[0]
    detected_entities = nlu_resp['entities']
    detected_intent = get_intent(nlu_resp)
    print(f'Detected intent --> {detected_intent}')
    if detected_entities is not None:
        print(f'Detected entities --> {detected_entities}')

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

    if len(missing_things) > 0:
        for miss in missing_things:
            # if it's possible to have alt
            if len(slots[detected_intent]['entities']['alts']) > 0:
                # get alternatives for missing
                miss_alts = slots[detected_intent]['entities']['alts'].get(
                    miss, '@@@@')
                if not any([ex for ex in extra if ex in miss_alts]):
                    missing.append(miss)
            else:
                missing.append(miss)
    with open('./pickle_files/missing.p', 'wb') as fp:
        pickle.dump(missing, fp)


def missing_ask(missing):
    pretty_name = {'des_label': 'your desired label',
                   'act_label': 'actual label'}
    return f'What is {pretty_name.get(missing[0], missing[0])} ?'


def missing_ask_again(missing):
    pretty_name = {'des_label': 'your desired label',
                   'act_label': 'actual label'}
    return f'So...can you tell me what is {pretty_name.get(missing[0], missing[0])} ?'


def load_brain():
    with open('./json_files/brain.json', 'r') as file:
        brain = json.load(file)
        return brain


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
    with open('./json_files/brain.json', 'w') as fp:
        json.dump(data, fp)


def save_filling(data):
    with open('./pickle_files/filling.p', 'wb') as file:
        pickle.dump(data, file)


def extract_num(text):
    return int(text.split()[-1])


def update_element(element, value):
    with open(f'./pickle_files/{element}.p', 'wb') as file:
        pickle.dump(value, file)


def get_explainers():
    with open('./pickle_files/available_explainers.p', 'rb') as file:
        return pickle.load(file)


def simplify_NLU(inp):
    res = {}
    res['intent'] = inp['intent']
    res['entities'] = None
    if len(inp['entities']) > 0:
        res['entities'] = {}
        for item in inp['entities']:
            res['entities'][item['entity']] = item['value']
    res['text'] = inp['text']
    return res


def get_intent(message):
    best_score = message['intent']['confidence']
    if best_score <= 0.5:
        return 'nlu_fallback'
    else:
        return message['intent']['name']


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
    # convert: xxx? --> xxx ?
    text = re.sub('\w{1}\?$', f'{text[-2]} ?', text)
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


def nlu_model_exist(dataset):
    p = f"../../RASA_NLU/models/{dataset}/"
    return path.exists(p)


def blackbox(dataset, algorithm):
    """
    USELESS
    """

    data = create_dataset(dataset)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    class_names = data['class_names']
    vectorizer = data.get('vectorizer', None)
    if algorithm == 'random_forest':

        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
        rf.fit(X_train, y_train)
        c = make_pipeline(vectorizer, rf)
        predict_proba = c.predict_proba

        return {'X_test': X_test, 'predict_proba': predict_proba, 'class_names': class_names, 'rf': rf}

    if algorithm == 'svm':
        svm = sklearn.svm.SVC(kernel='rbf', probability=True)
        svm.fit(X_train, y_train)

        return {'X_test': X_test, 'predict_proba': svm.predict_proba, 'class_names': class_names, 'rf': svm}


def label_to_index(dataset, label):
    data_path = './json_files'
    with open(f'./{data_path}/datasets.json', 'r') as file:
        dataset_mapping = json.load(file)
    labels = dataset_mapping[dataset]['labels']
    idx = dataset_mapping[dataset]['label_index']
    dictt = {k: v for k, v in zip(labels, idx)}
    return dictt[label]


def index_to_label(dataset, ind):
    data_path = './json_files'
    with open(f'./{data_path}/datasets.json', 'r') as file:
        dataset_mapping = json.load(file)
    labels = dataset_mapping[dataset]['labels']
    idx = dataset_mapping[dataset]['label_index']
    dictt = {k: v for k, v in zip(idx, labels)}
    return dictt[ind]


#global memory
#memory = load_brain()
