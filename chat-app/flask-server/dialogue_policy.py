from lime.lime_text import LimeTextExplainer
import lime
from lime import lime_text
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.ensemble
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import random
from sklearn.pipeline import make_pipeline
import json
import pickle
import time
import matplotlib.pyplot as plt
import shap
import warnings
import re
import contrastive_explanation as ce


plt.switch_backend('Agg')


def extract_num(text):
    return int(text.split()[-1])


def get_intent(message):
    if isinstance(message['intent'], dict):
        best_score = message['intent']['confidence']
        if best_score <= 0.5:
            return 'nlu_fallback'
        else:
            return message['intent']['name']
    elif isinstance(message['intent'], str):
        return message['intent']


# def create_dataset(dataset):
#     if dataset == '20newsgroup':
#         categories = ['alt.atheism', 'soc.religion.christian']
#         newsgroups_train = fetch_20newsgroups(
#             subset='train', categories=categories)
#         newsgroups_test = fetch_20newsgroups(
#             subset='test', categories=categories)
#         class_names = ['atheism', 'christian']

#         vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
#             lowercase=False)
#         X_train = vectorizer.fit_transform(newsgroups_train.data)
#         y_train = newsgroups_train.target
#         X_test = newsgroups_test.data
#         y_test = newsgroups_test.target

#         return {'X_train': X_train,
#                 'y_train': y_train,
#                 'X_test': X_test,
#                 'y_test': y_test,
#                 'vectorizer': vectorizer,
#                 'class_names': class_names}

#     elif dataset == 'iris':
#         data = pd.read_csv('./blackbox_input/iris.csv')
#         train, test = train_test_split(
#             data, test_size=0.4, stratify=data['species'], random_state=42)
#         class_names = ['setosa', 'versicolor', 'virginica']
#         X_train = train[['sepal_length', 'sepal_width',
#                          'petal_length', 'petal_width']]
#         y_train = train.species
#         X_test = test[['sepal_length', 'sepal_width',
#                        'petal_length', 'petal_width']]
#         print(f'X_test is:\n{X_test}')
#         y_test = test.species
#         # prediction = model.predict(X_test)

#         return {'X_train': X_train,
#                 'y_train': y_train,
#                 'X_test': X_test,
#                 'y_test': y_test,
#                 'class_names': class_names}


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
        svm.fit(X_train, Y_train)

        return {'X_test': X_test, 'predict_proba': svm.predict_proba, 'class_names': class_names, 'rf': svm}

class FoilTree_explanations(object):
    def __init__(self, dataset, model):
        # bb_data = blackbox(dataset, model)
        with open(f'{dataset}_artifacts.p', 'rb') as file:
            bb_data = pickle.load(file)
        with open(f'{model}.p', 'rb') as file:
            self.model = pickle.load(file)
        
        self.X_test = bb_data['X_test']
        self.X_train = bb_data['X_train']
        self.class_names = bb_data['class_names']
        self.feature_names = bb_data['feature_names']

    def contrastive_explain(self, record, foil):
        sample = self.X_test[record]

        # Create a domain mapper (map the explanation to meaningful labels for explanation)
        dm = ce.domain_mappers.DomainMapperTabular(self.X_train,
                                                feature_names=self.feature_names,
                                                contrast_names=self.class_names)

        # Create the contrastive explanation object (default is a Foil Tree explanator)
        exp = ce.ContrastiveExplanation(dm)

        # Explain the instance (sample) for the given model
        exp = exp.explain_instance_domain(self.model.predict_proba, sample, foil=foil)

        return exp




class LIME_explanations(object):
    def __init__(self, dataset, model):
        # bb_data = blackbox(dataset, model)
        with open(f'{dataset}_artifacts.p', 'rb') as file:
            bb_data = pickle.load(file)
        with open(f'{model}.p', 'rb') as file:
            self.model = pickle.load(file)
        self.class_names = bb_data['class_names']
        self.X_test = bb_data['X_test']
        self.X_train = bb_data['X_train']
        self.y_test = bb_data['y_test']
        self.vectorizer = bb_data.get('vectorizer', None)
        self.feature_names = bb_data['feature_names']
        # self.pred_proba = bb_data['predict_proba']
        self.text_explainer = LimeTextExplainer(class_names=self.class_names)
        self.table_explainer = explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
         feature_names=self.feature_names,
          class_names=self.class_names,
          discretize_continuous=True)

    def predict(self, record):
        rf = blackbox(dataset, model)['rf']
        pass
        # get rf from the blackbox function
        # to get a new prediction you should pass the row as follows
        # in this case there are 4 features
        # vals = np.array([6.6, 3. , 4.4, 1.4]).reshape(1,-1)
        # predictions = rf.predict(vals)

    def explain_why_text_old(self, record):
        exp = self.text_explainer.explain_instance(
            self.X_test[record], self.pred_proba, num_features=6)
        exp = exp.as_list()

        base = f'The model predicted record {record}, mainly based on the presense of the following 6 words:\n\n'
        res = ''
        for e in exp:
            res += f'{e[0]}\n '

        return f'{base} {res[:-2]}'
    def explain_why_text(self, record):
        
        t0 = time.time()
        c = make_pipeline(self.vectorizer, self.model)
        predict_proba = c.predict_proba

        exp = self.text_explainer.explain_instance(
            self.X_test[record], predict_proba, num_features=6)
        exp_type = random.choice(['TEXT', 'GRAPHIC'])
        # HARD-CODING
        exp_type = 'TEXT'
        if exp_type == 'TEXT':
            print(f' >>>>>>>>>>>>> record true label is {self.y_test[record]}')
            exp = exp.as_list()
            # exp = exp.as_list(self.y_test[record])

            base = f'The model predicted record {record}, mainly based on the presense of the following 6 words:\n\n'
            res = ''
            for e in exp:
                res += f'{e[0]}\n '
            t1 = time.time()
            print(f'Elapsed time for explain_why_text: {t1-t0}')
            return (exp_type, f'{base} {res[:-2]}')
        else:
            img = exp.as_pyplot_figure()
            #plot_path_simp = f'/lime_plots/LIME_WHY_RECORD_{record}.png'
            plot_path_simp = f"/lime_plots/LIME_WHY_RECORD_{record}.png"
            img.savefig(f'/Users/navid/Documents/rasa_practice/XAIBot_V1/chat-app/client/public/lime_plots/LIME_WHY_RECORD_{record}.png')
            return (exp_type, plot_path_simp) 
    
    def explain_why_table(self, record):

        def rule_to_text(rules):
        
            def sign_to_text(txt, num, bet=False):
                if not bet:
                    if '<=':
                        text = f'{txt} is less than or equal {num}'
                    elif '<':
                        text = f'{txt} is less than {num}'
                    elif '>=':
                        text = f'{txt} is greater than or equal {num}'
                    elif '>':
                        text = f'{txt} is greater than {num}'
                    return text
                else:
                    return f'{txt} is between {num[0]} and {num[1]}'
            
            
            
            def is_between(rule):
                res = False
                if all(s in rule for s in ['<', '>']) or rule.count('>') > 1 or rule.count('<') > 1:
                    return True
                return False
            

                
            res = []
            for rule in rules:
                if not is_between(rule):
                    spl = [s.strip() for s in re.split("<=|>=|<|>", rule)]
                    txt = spl[0]
                    num = spl[1]

                    res.append(f'{sign_to_text(txt, num)}, ')
                    
                else:
                    spl = [s.strip() for s in re.split("<=|>=|<|>", rule)]
                    txt = spl[1]
                    num = [spl[0]] + [spl[-1]]
                    res.append(f'{sign_to_text(txt, num, bet=True)}, ')
                    
            if res != 1:
                res = ''.join(res[:-1]) + f'and {res[-1][:-2]} .'
            
            return res

        exp = self.table_explainer.explain_instance(
            self.X_test[record], self.model.predict_proba, num_features=2, top_labels=1)
        exp = exp.as_list(self.y_test[record])
        exp = [e[0] for e in exp]
        pred_label = self.class_names[self.y_test[record]]
        base = f'The model predicted record {record} as {pred_label.title()}, becasue '
        res = rule_to_text(exp)
        return ('TEXT', f'{base} {res}')


    # def explain_why_text(self, record):

    #     data = pd.read_csv('/Users/navid/Documents/rasa_practice/XAIBot_V1/blackbox_input/iris.csv')
    #     train, test = train_test_split(
    #         data, test_size=0.4, stratify=data['species'], random_state=42)
    #     class_names = ['setosa', 'versicolor', 'virginica']
    #     X_train = train[['sepal_length', 'sepal_width',
    #                      'petal_length', 'petal_width']]
    #     y_train = train.species
    #     X_test = test[['sepal_length', 'sepal_width',
    #                    'petal_length', 'petal_width']]
    #     y_test = test.species

    #     svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    #     svm.fit(X_train, y_train)
    #     predict_proba = svm.predict_proba
    #     explainer = shap.KernelExplainer(predict_proba, X_train, link="logit")
    #     shap_values = explainer.shap_values(X_test, nsamples=100)

    #     # plot the SHAP values for the Setosa output of the first instance
    #     fig = shap.force_plot(explainer.expected_value[0],
    #                   shap_values[0][0,:], X_test.iloc[0,:], link="logit",
    #                  matplotlib=True, show= False)
        
    #     plot_path = f'/Users/navid/Documents/rasa_practice/XAIBot_V1/chat-app/client/public/lime_plots/SHAP_WHY_RECORD_{record}.png'
    #     plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
    #     return ('GRAPHIC', plot_path) 



    def most_important(self, record):
        t0 = time.time()
        c = make_pipeline(self.vectorizer, self.model)
        predict_proba = c.predict_proba

        exp = self.text_explainer.explain_instance(
            self.X_test[record], predict_proba, num_features=6)

        exp = exp.as_list()[0][0]
        exp = f'The most important feature for this prediction is the word "{exp}"'

        return exp


def load_generic_responses(file):
    with open(file, 'r') as file:
        data = json.load(file)
        return data


class Explanation:

    general_intents = ['greet',
                       'affirm',
                       'goodbye',
                       'deny',
                       'nlu_fallback',
                       'chitchat_general',
                       'help',
                       'human_handoff',
                       'thank_you']
    general_responses = load_generic_responses('generic_responses.json')

    
    def __init__(self, message):
        intent = message['intent']
        entities = message['entities']
        
        info = load_brain()
        self.intent = intent
        self.entities = entities
        self.dataset = info['dataset']
        self.model = info['model']
        self.profile = info['profile']
        self.end = False

    def __seek_explanation(self):
        res = False
        if self.intent not in Explanation.general_intents:
            res = True
        return res

    def create(self):
        if self.__seek_explanation():
            pass
        else:
            response_text = Explanation.general_responses[self.intent]
            return (response_text, "wants_answer", self.end)

def load_brain():
    with open('brain.json', 'r') as file:
        brain = json.load(file)
        return brain

def recalculate_pred(intent, dataset, token, model, record):
    # loading training data
    with open(f'{dataset}_artifacts.p', 'rb') as file:
        bb_data = pickle.load(file)

    X_test = bb_data['X_test']
    y_test = bb_data['y_test']

    # loading saved model (blackbox)
    print(f'model --> {model}')
    with open(f'{model}.p', 'rb') as file:
        blackbox = pickle.load(file)

    if intent == 'what_if_add':
        mode = 'add'
    elif intent == 'what_if_del':
        mode = 'del'
    data = X_test[record]

    vectorizer = bb_data['vectorizer']
    class_names = bb_data['class_names']


    def del_token(model, record, token):
        record = X_test[record].lower()
        record = record.replace(token.lower(), ' ')
        X = vectorizer.transform([record])
        pred = model.predict(X.reshape(1,-1))
        return pred

    def add_token(model, record, token):
        record = X_test[record].lower()
        record += f' {token.lower()}'
        X = vectorizer.transform([record])
        pred = model.predict(X.reshape(1,-1))
        return pred
    
    current_label = y_test[record]

    if mode == 'add':
        new_label = add_token(blackbox, record, token)
    elif mode == 'del':
        new_label = del_token(blackbox, record, token)

    if mode == 'add':
        p = 'Adding'
    else:
        p = 'Deleting'

    current_label = class_names[current_label]
    new_label = class_names[new_label[0]]

    if new_label == current_label:
        return f'{p} {token} will results in the same prediction: {current_label}'
    else:
        return f'{p} {token} will change the prediction to {new_label} while the previous label was {current_label}'
    



def make_response(message):
    end = False
    intent = message['intent']
    entities = message['entities']
    print(f'Response should be created with intent: {intent} and entities: {entities}')
    # and check_components(message):
    if intent in ('why_this', 'list_features'):

        # print('HERE -->> XXXXXXX00')
        # HARD - CODING:
        explainer = 'LIME'
        info = load_brain()
        dataset = info['dataset']
        model = info['model']

        if explainer == 'LIME':
            # print('HERE -->> XXXXXXX11')
            lo = LIME_explanations(dataset, model)
            # entities = get_entities(message)
            record = extract_num(entities['record'])
            if dataset == '20_news_group':
                resp_type, response = lo.explain_why_text(record)
                if resp_type == 'TEXT':
                    print('user is asking for a text response')
                    intent = "wants_answer"
                else:
                    print('Should show plot!')
                    return (response, "wants_plot", False)
            elif dataset == 'iris':
                resp_type, response = lo.explain_why_table(record)
                intent = "wants_answer"
        # return {
		# 	"intention":"wants_plot",
		# 	"url" : "https://res-3.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_256,w_256,f_auto,q_auto:eco/v1459804290/mkxozts4fsvkj73azuls.png",
		# 	"end_of_converstaion" : False,
		# }

    elif intent == 'most_important_feature':
        info = load_brain()
        dataset = info['dataset']
        model = info['model']
        lo = LIME_explanations(dataset, model)
        # entities = get_entities(message)
        record = extract_num(entities['record'])
        response = lo.most_important(record)
        intent = "wants_answer"

    if intent in ('what_if_add', 'what_if_del'):
        
        info = load_brain()
        dataset = info['dataset']
        model = info['model']
        token = entities['token']
        record = extract_num(entities['record'])

        response = recalculate_pred(intent, dataset, token, model, record)
        intent = "wants_answer"

    elif intent == 'greet':
        response = random.choice(['Hi there!', 'Hello!', 'Hi!', 'Hey there!'])
        intent = "wants_answer"

    elif intent == 'goodbye':
        response = random.choice(
            ['Goodbye!', 'See you later!', 'Bye! see you later!'])
        intent = "wants_answer"

    elif intent == 'why_not':
        intent = "wants_answer"
        info = load_brain()
        dataset = info['dataset']
        model = info['model']

        lo = FoilTree_explanations(dataset, model)
        record = extract_num(entities['record'])
        if dataset == 'iris':
            response = lo.contrastive_explain(record, info['entities']['des_label'])

    elif intent == 'nlu_fallback':
        resps = ['Sorry I didn\'t get what you said...could you rephrase it?',
                 'Hmmm...can you repeat what you said ?',
                 'Don\'t hate me but I didn\'t understand what you said :(',
                 'Sorry but it seems I have no answer to that...can you rephrase it ?',
                 'Sorry but I can\'t understand what you say']
        response = random.choice(resps)
        intent = "wants_answer"

    elif intent == 'chitchat_general':
        resps = ['I have no time for chitchat..human :D',
                 'We\'ll talk about it later but first ask me something about the blackbox!',
                 'I\'m the king (or queen ?) of the chitchat but first let\'s see how can I help you with the results!']
        response = random.choice(resps)
        intent = "wants_answer"

    elif intent == 'thankyou':
        resps = ['You\'re welcome!',
                 'My pleasure!',
                 'Happy to help! :)',
                 ]
        response = random.choice(resps)
        intent = "wants_answer"

    elif intent == 'human_handoff':
        resps = ['Sorry but my human is not available right now :(',
                 'Error 404: No human was found !',
                 'Who needs a human when such a smart bot like me exist ;)Hey '
                 ]
        response = random.choice(resps)
        intent = "wants_answer"

    elif intent == 'help':
        response = """I\'m a chat-bot you can ask anything about the outcome of the blackbox model. Here are some examples:\n
                        - Why the model classified instance number N as class X ?
                        - What are the most important features the model used for classifying the instance N ?
                        - What is the most important feature for classifying the instance N ? """
        intent = "wants_answer"

    return (response, intent, end)
