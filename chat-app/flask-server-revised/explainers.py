import re
import time
import random
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
import shap
from aux_functions import *
from sklearn.pipeline import make_pipeline
import contrastive_explanation as ce
import pickle
from components import data_type_categories, dataset_mapping, profile_mapping, model_mapping
import matplotlib.pyplot as plt
import shutil


class FoilTree_explanations(object):
    def __init__(self, dataset, model):
        # data = blackbox(dataset, model)
        with open(f'{dataset}_artifacts.p', 'rb') as file:
            data = pickle.load(file)
        with open(f'{model}.p', 'rb') as file:
            self.model = pickle.load(file)

        self.X_test = data['X_test']
        self.X_train = data['X_train']
        self.class_names = data['class_names']
        self.feature_names = data['feature_names']

    def contrastive_explain(self, record, foil):
        sample = self.X_test[record]

        # Create a domain mapper (map the explanation to meaningful labels for explanation)
        dm = ce.domain_mappers.DomainMapperTabular(self.X_train,
                                                   feature_names=self.feature_names,
                                                   contrast_names=self.class_names)

        # Create the contrastive explanation object (default is a Foil Tree explanator)
        exp = ce.ContrastiveExplanation(dm)

        # Explain the instance (sample) for the given model
        exp = exp.explain_instance_domain(
            self.model.predict_proba, sample, foil=foil)

        return exp

    def contrastive_explain_regression(self, record, level):
        sample = self.X_test[record]

        # Create a domain mapper (map the explanation to meaningful labels for explanation)
        dm = ce.domain_mappers.DomainMapperTabular(self.X_train,
                                                   feature_names=self.feature_names)

        # Create the contrastive explanation object (default is a Foil Tree explanator)
        exp = ce.ContrastiveExplanation(dm,
                                        regression=True,
                                        explanator=ce.explanators.TreeExplanator(
                                            verbose=True),
                                        verbose=False)

        # Explain the instance (sample) for the given model
        exp = exp.explain_instance_domain(
            self.model.predict_proba, sample, reg=level)

        return exp


class LIME_explanations(object):
    def __init__(self, dataset, model):
        # data = blackbox(dataset, model)
        with open(f'./pickle_files/{dataset}_artifacts.p', 'rb') as file:
            self.bb_data = pickle.load(file)
        with open(f'./pickle_files/{model}.p', 'rb') as file:
            self.model = pickle.load(file)
        self.dataset = dataset
        self.class_names = self.bb_data.get('class_names', None)
        self.X_test = self.bb_data['X_test']
        self.X_train = self.bb_data['X_train']
        self.y_test = self.bb_data['y_test']
        self.vectorizer = self.bb_data.get('vectorizer', None)
        self.feature_names = self.bb_data['feature_names']
        # self.pred_proba = bb_data['predict_proba']
        self.text_explainer = LimeTextExplainer(class_names=self.class_names)

        dataset_data_type = dataset_mapping[dataset]['data_type']
        if dataset_data_type[-3:] == 'num':
            mode = 'regression'
        else:
            mode = 'classification'

        if dataset in ['iris', 'wine', 'abalone', 'breast_cancer', 'diabetes']:
            self.table_explainer = LimeTabularExplainer(self.X_train.values,
                                                        feature_names=self.feature_names,
                                                        class_names=self.class_names,
                                                        discretize_continuous=True,
                                                        mode=mode)
        else:
            self.table_explainer = LimeTabularExplainer(self.X_train,
                                                        feature_names=self.feature_names,
                                                        class_names=self.class_names,
                                                        discretize_continuous=True,
                                                        mode=mode)
        self.model_name = model

        brain = load_brain()
        user_profile = brain['profile']
        presentations = profile_mapping[user_profile]['presentation']
        print(f'Available presentations: {presentations}')
        if len(presentations) > 1:
            self.presentation_type = random.choice(presentations)
        else:
            self.presentation_type = presentations[0]
        print(f'Chosen presentation: {self.presentation_type}')

    def explain_why_text_old(self, record):
        exp = self.text_explainer.explain_instance(
            self.X_test[record], self.pred_proba, num_features=6)
        exp = exp.as_list()

        base = f'The model predicted record {record}, mainly based on the presence of the following 6 words:\n\n'
        res = ''
        for e in exp:
            res += f'{e[0]}\n '

        return f'{base} {res[:-2]}'

    def explain_why_text(self, record):
        print(f'X_train type: {type(self.X_train)}')
        print(f'X_train type: {type(self.X_test)}')

        t0 = time.time()
        c = make_pipeline(self.vectorizer, self.model)
        predict_proba = c.predict_proba

        if self.model_name == 'naive_bayes':
            exp = self.text_explainer.explain_instance(
                self.X_test[record], predict_proba, num_features=6)
        else:
            exp = self.text_explainer.explain_instance(
                self.X_test[record], predict_proba, num_features=6)
        
        #presentation_type = 'plot'
        if self.presentation_type == 'text':
            print(f' >>>>>>>>>>>>> record true label is {self.y_test[record]}')
            exp = exp.as_list()
            # exp = exp.as_list(self.y_test[record])

            base = f'The model predicted record {record}, mainly based on the presence of the following 6 words:\n\n'
            res = ''
            for e in exp:
                res += f'{e[0]}\n '
            t1 = time.time()
            print(f'Elapsed time for explain_why_text: {t1-t0}')
            return (self.presentation_type, f'{base} {res[:-2]}')
        else:
            img = exp.as_pyplot_figure()

            file_name = f'LIME_WHY_RECORD_{record}.png'
            final_path = f"/lime_plots/{file_name}"
            original_path = f'./{file_name}'
            destination_path = f'../../client/public/lime_plots/{file_name}'
            img.savefig(original_path)
            shutil.move(original_path, destination_path)

            return (self.presentation_type, final_path)

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
        # print(f'\n\nX_test len --> {len(self.X_test[record])}')
        print(f'dataset --> {self.dataset}')
        print(f'class names --> {self.class_names}\n\n')
        if self.dataset in ['iris', 'wine', 'abalone', 'breast_cancer', 'diabetes']:
            exp = self.table_explainer.explain_instance(
                self.X_test.iloc[record, :], self.model.predict_proba, num_features=2, top_labels=1)

            exp = exp.as_list(self.y_test.values[record])
            exp = [e[0] for e in exp]
            print('-'*50)
            print(record)
            print(self.y_test.values)
            print(self.y_test.values[record])

            pred_label = self.class_names[self.y_test.values[record]]
        else:
            exp = self.table_explainer.explain_instance(
                self.X_test[record], self.model.predict_proba, num_features=2, top_labels=1)
            exp = exp.as_list(self.y_test[record])
            exp = [e[0] for e in exp]
            pred_label = self.class_names[self.y_test[record]]

        if self.presentation_type == 'text':

            base = f'The model predicted record {record} as {str(pred_label).title()}, because '
            res = rule_to_text(exp)
            return (self.presentation_type, f'{base} {res}')
        else:
            img = exp.as_pyplot_figure(label=0)

            file_name = f'LIME_WHY_RECORD_{record}.png'
            final_path = f"/lime_plots/{file_name}"
            original_path = f'./{file_name}'
            destination_path = f'../../client/public/lime_plots/{file_name}'
            img.savefig(original_path)
            shutil.move(original_path, destination_path)

            return (self.presentation_type, final_path)


    def most_important(self, record):
        t0 = time.time()
        c = make_pipeline(self.vectorizer, self.model)
        predict_proba = c.predict_proba

        exp = self.text_explainer.explain_instance(
            self.X_test[record], predict_proba, num_features=6)

        exp = exp.as_list()[0][0]
        exp = f'The most important feature for this prediction is the word "{exp}"'

        return exp


class SHAP_explanations(object):
    def __init__(self, dataset, model):

        # getting dataset's data type
        data_path = './json_files'
        with open(f'./{data_path}/datasets.json', 'r') as file:
            dataset_mapping = json.load(file)
        dataset_data_type = dataset_mapping[dataset]['data_type']

        with open(f'./pickle_files/{model}.p', 'rb') as file:
            self.model = pickle.load(file)

        # bb_data = blackbox(dataset, model)
        with open(f'./pickle_files/{dataset}_artifacts.p', 'rb') as file:
            self.data = pickle.load(file)

        self.dataset = dataset
        self.class_names = self.bb_data.get('class_names', None)
        self.X_test = self.data['X_test']
        self.X_train = self.data['X_train']
        self.y_test = self.data['y_test']
        self.vectorizer = self.data.get('vectorizer', None)
        self.feature_names = self.data.get('feature_names', None)
        # self.X = self.data['X']
        # self.pred_proba = bb_data['predict_proba']
        #self.kernel_explainer = shap.KernelExplainer(self.model.predict, self.X_test)

    def tree_based(self, intent, label, feature, record):
        shap.initjs()
        tree_explainer = shap.TreeExplainer(self.model)
        # obtain shap values for the first row of the test data
        shap_values = tree_explainer.shap_values(self.X_test)

        label_idx = label_to_index(self.dataset, label)
        if intent == 'overall_contribution_to_label':
            # use force plot | decision plot | summary plot
            choice = random.choice(range(2))
            if choice == 3:  # it's not possible to save this plot a the moment
                # force plot
                file_name = f'{self.dataset}_{label_idx}_force_plot.png'
                shap.force_plot(tree_explainer.expected_value[label_idx],
                                shap_values[label_idx],
                                self.X_test,
                                show=False)
            elif choice == 1:
                # decision plot
                file_name = f'{self.dataset}_{label_idx}_decision_plot.png'
                top_n = 10
                shap.decision_plot(tree_explainer.expected_value[label_idx],
                                   shap_values[label_idx],
                                   self.X_test,
                                   feature_display_range=slice(
                                       -1, -(top_n+1), -1),
                                   show=False)

            else:
                # summary plot
                file_name = f'{self.dataset}_{label_idx}_summary_plot.png'
                shap.summary_plot(shap_values[label_idx], self.X_test, plot_type='violin',
                                  max_display=10,
                                  show=False)

        elif intent == 'feature_importance_global':
            file_name = f'{self.dataset}_feature_importance_global.png'
            shap.summary_plot(shap_values,
                              max_display=10,
                              show=False)

        elif intent == 'single_contribution_to_label':
            data = self.X_test.iloc[record:record+1, :]
            shap_values = tree_explainer.shap_values(data)
            shap.force_plot(
                tree_explainer.expected_value[label_idx], shap_values[label_idx], data)
            file_name = f'{self.dataset}_{label_idx}__label={label_idx}_record={record}_force_plot.png'
            shap.force_plot(tree_explainer.expected_value[label_idx],
                            shap_values[label_idx],
                            data,
                            show=False,
                            matplotlib=True)

        elif intent == 'feature_effect':
            # use dependence_plot
            shap_values = tree_explainer.shap_values(self.X_test)
            file_name = f'{self.dataset}_{label_idx}_dependence_plot.png'
            shap.dependence_plot(feature, shap_values[label_idx], self.X_test,
                                 show=False,
                                 matplotlib=True)

        return self.plot_saver(file_name)

    def regression(self, intent, feature, record):

        explainer = shap.Explainer(self.model.predict, self.X_train)
        shap_values = explainer(self.X_test)

        if intent == 'feature_effect_regression':
            file_name = f'{self.dataset}_feature:{feature}_PDP.png'
            shap.plots.partial_dependence(
                feature,
                self.model.predict,
                self.X_train,
                ice=True,
                model_expected_value=True,
                feature_expected_value=True,
                show=False)

        elif intent == 'why_this_regression':
            file_name = f'{self.dataset}_record:{record}_why_this.png'
            shap.plots.waterfall(shap_values[record], show=False)

        elif intent == 'feature_importance_global':
            file_name = f'{self.dataset}_feature_importance_global.png'
            shap.plots.bar(shap_values, show=False)

        return self.plot_saver(file_name)

    def plot_saver(self, file_name):
        # saving and moving the plot
        final_path = f"/shap_plots/{file_name}"
        original_path = f'./{file_name}'
        destination_path = f'../client/public/shap_plots/{file_name}'
        plt.savefig(original_path, dpi=150, bbox_inches='tight')
        shutil.move(original_path, destination_path)

        return final_path



class FAKE_explanations(object):
    def __init__(self, dataset, model) :
        self.dataset = dataset
        with open(f'./pickle_files/{model}.p', 'rb') as file:
            self.model = pickle.load(file)
        with open(f'{dataset}_artifacts.p', 'rb') as file:
            data = pickle.load(file)

        self.X_test = data['X_test']
        self.y_test = data['y_test']

        # loading saved model (blackbox)
        print(f'model --> {self.model}')
        with open(f'{self.model}.p', 'rb') as file:
            self.blackbox = pickle.load(file)
    
    def recalculate_textual(self, token, record, intent):

        if intent == 'what_if_add':
            mode = 'add'
        elif intent == 'what_if_del':
            mode = 'del'
        # data = self.X_test[record]

        vectorizer = data['vectorizer']
        class_names = data['class_names']

        def del_token(model, record, token):
            record = self.X_test[record].lower()
            record = record.replace(token.lower(), ' ')
            X = vectorizer.transform([record])
            pred = model.predict(X.reshape(1, -1))
            return pred

        def add_token(model, record, token):
            record = self.X_test[record].lower()
            record += f' {token.lower()}'
            X = vectorizer.transform([record])
            pred = model.predict(X.reshape(1, -1))
            return pred

        current_label = self.y_test[record]

        if mode == 'add':
            new_label = add_token(self.blackbox, record, token)
        elif mode == 'del':
            new_label = del_token(self.blackbox, record, token)

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
    
    def recalculate_regression(self, feature, new_val, record):
        """currently nly supports tabular_num_num, NOT tabular_mixed_num"""
        
        original_record = self.X_test.iloc[record,:]
        modified_record = original_record.copy(deep=True)

        # original_pred = self.blackbox.predict(original_record).values[0]

        modified_record[feature] = new_val 
        new_pred = blackbox.predict(modified_record).values[0]

        response = f'By changing the {feature} feature to {new_val}, the prediction will be changed to {new_pred}'

        return response

def explainer_handler(explainer, dataset, model, memory):
    # with open('./pickle_files/available_explainers.p', 'rb') as file:
    #     available_explainers = pickle.load(file)
    # end = False
    intent = memory['intent']
    entities = memory['entities']
    label = entities['label']
    token = entities['token']
    
    feature = entities['feature']
    record = extract_num(entities['record'])
    info = load_brain()
    dataset = info['dataset']
    model = info['model']
    data_type = dataset_mapping[dataset]['data_type']
    is_tree_based = model_mapping[model]['tree_based']

    level = entities['level']
    if level:
        for i in ['small', 'low', 'less']:
            if i in level:
                level = 'smaller'
        for i in ['big', 'high', 'large']:
            if i in level:
                level = 'greater'

    print(
        f'\n\nResponse should be created with intent: {intent} and entities: {entities}')

    # if len(available_explainers) == 1:
    #     explainer = available_explainers[0]
    #     print(f'\n\nThe only available explainer is {explainer}')
    # else:
    #     explainer = random.choice(available_explainers)
    #     print(f'\n\n {explainer} explainer is randomly chosen from {available_explainers}')
    if explainer == 'fake':
        fe = FAKE_explanations(dataset, model)

        if intent in ('what_if_add', 'what_if_del'):
            if data_type == 'textual_cat':
                response = fe.recalculate_textual(token, record, intent)
        elif intent == 'what_if_subs':
            if data_type == 'tabular_num_num':
                response = fe.recalculate_regression(token, record, intent)

        return (response, 'wants_answer', False)

    if explainer == 'lime':
        lo = LIME_explanations(dataset, model)

        if intent in ('why_this', 'list_features', 'why_this_regression'):
            if data_type == 'textual_cat':
                resp_type, response = lo.explain_why_text(record)

            elif data_type in data_type_categories['tabular']:
                resp_type, response = lo.explain_why_table(record)

            if resp_type == 'text':
                print('user is asking for a text response')
                intent = "wants_answer"
            elif resp_type == 'plot':
                print('Should show plot!')
                intent = 'wants_plot'

        elif intent == 'most_important_feature':
            response = lo.most_important(record)
        return (response, intent, False)

    if explainer == 'shap':
        shap_obj = SHAP_explanations(dataset, model)

        if is_tree_based:
            response = shap_obj.tree_based(intent, label, feature, record)
        else:
            response = shap_obj.regression(intent, feature, record)
        return (response, 'wants_plot', False)

    if explainer == 'foiltree':
        lo = FoilTree_explanations(dataset, model)
        record = extract_num(entities['record'])
        if data_type in data_type_categories['tabular']:
            if intent in ('why_not', 'expectation_not_met'):
                response = lo.contrastive_explain(
                    record, info['entities']['des_label'])
                return (response, 'wants_answer', False)
            elif intent == 'why_not_regression':
                response = lo.contrastive_explain_regression(record, level)
                return (response, 'wants_answer', False)
