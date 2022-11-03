import random
from aux_functions import *
from components import  exp_mapping, dataset_mapping
from explainers import explainer_handler



generic_responses = load_component('generic_responses')
general_intents = generic_responses.keys()


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
        pred = model.predict(X.reshape(1, -1))
        return pred

    def add_token(model, record, token):
        record = X_test[record].lower()
        record += f' {token.lower()}'
        X = vectorizer.transform([record])
        pred = model.predict(X.reshape(1, -1))
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


def explain_old(message):
    with open('./pickle_files/available_explainers.p', 'rb') as file:
        available_explainers = pickle.load(file)
    end = False
    intent = message['intent']
    entities = message['entities']
    print(
        f'Response should be created with intent: {intent} and entities: {entities}')
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
                # 	"end_of_conversation" : False,
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
            response = lo.contrastive_explain(
                record, info['entities']['des_label'])

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

def explain(memory):

    
    with open('./pickle_files/available_explainers.p', 'rb') as file:
        available_explainers = pickle.load(file)
    end = False
    intent = memory['intent']
    entities = memory['entities']
    info = load_brain()
    dataset = info['dataset']
    model = info['model']
    data_type = dataset_mapping[dataset]['data_type']
    
    print(
        f'Response should be created with intent: {intent} and entities: {entities}')
    
    if intent in general_intents:
        print(f'intent({intent}) is in the general intents {general_intents}')
        response = random.choice(generic_responses[intent])
        intent = "wants_answer"
        return (response, intent, end)
    
    for explainer in  available_explainers:
        supported_intents = exp_mapping[explainer]['supported_intents']
        print(f'DP.py --> supported intents: {supported_intents}')
        if intent not in supported_intents:
            continue
        else:
           response, intent, end =  explainer_handler(explainer, dataset, model, memory)
           return (response, intent, end)



def DP(state):
    memory = load_brain()
    if state == 'wrong_question':

        return {
            "msg": """hmmm...you can\'t ask such question about this dataset :(
            Here are some examples (but remember that you don't need to use their exact wording!)
            - what are the features the model used for classifying INS ?
            - what is the most important feature in INS ?
            - what if i change the FEAT to NEW_VAL for INS ?
            - what if i add TOKEN to INS ?
            - why the INS is classified as ACT_LAB , instead of DES_LAB ?
            - why INS is classified as LAB ?
            - why the model predicted INS as ACT_LAB but the INS as DES_LAB ?""",
            "intention": 'wants_answer',
            "end_of_conversation": False,
        }

    if state == 'unsupported_datatype':
        
        dataset_obj = load_component('datasets')
        dataset = memory['dataset']
        dataset = dataset_obj[ugly(dataset)]
        ddt = dataset['data_type']

        return {
            "msg": f'Sorry but this data type ({ddt}) is not supported byt the selected model',
            "intention": 'wants_answer',
            "end_of_conversation": False,
        }
    if state == 'record is out of range':
        record = memory['entities']['record']
        dataset = memory['dataset']
        dataset_obj = dataset_mapping[dataset]
        num_records = dataset_obj['num_records']
        return {
            "msg": f'There are {num_records} records in the data but you\'re asking for record {record}. Please repeat your question with the correct record.',
            "intention": 'wants_answer',
            "end_of_conversation": False,
        }

    if state == 'can proceed with explanation':
        memory = load_brain()
        resp = explain(memory)
        print(f'resp before anything is: {resp}')

        if len(resp) == 3:
            response, intent, end = resp
            empty_brain()
            print('resp has 3 items!')
            if intent == 'wants_plot':
                print('main is responding with wants plot')
                print(f'the response which goes to chatbot is: {resp}')
                
                return {
                    "url": response,
                    "intention": intent,
                    "end_of_conversation": end,
                }
            
            return {
                "msg": response,
                "intention": intent,
                "end_of_conversation": end,
            }
        else: # options
            print('XXXXXXXX-->3')
            print(f'resp is:\n{explain(memory)}')
            response, option, intent, end = explain(memory)

            return {
                "msg": "here is the list:",
                "options": ['AAAAAA', 'BBBBBB', 'CCCCCC', 'DDDDDDDD'],
                "intention": "wants_list",
                "end_of_conversation": False,
            }

    if state == 'missing_value':
        missing = load_memory_files()[0]
        return {
            "msg": missing_ask(missing),
            "intention": "wants_answer",
            "end_of_conversation": False,
        }

    if state == 'filled_one_missing':
        memory = load_brain()
        response, intent, end = explain(memory)
        empty_brain()
        if intent == 'wants_plot':
            print('NLU is responding with wants plot')
            return {
                "intention": "wants_plot",
                "url": response,
                "end_of_conversation": False,
            }
        return {
            "msg": response,
            "intention": intent,
            "end_of_conversation": end,
        }

    if state == 'new_intent_detected':

        return {
            "msg": 'Do you want to change your initial question?',
            "intention": "wants_answer",
            "end_of_conversation": False,
        }

    if state == 'mind_change_NOT_yes_no':

        return {
            "msg": 'I didn\'t get what you mean. let\'s start from the beginning',
            "intention": "wants_answer",
            "end_of_conversation": False,
        }

    if state == 'mind_change_yes_no':

        response = random.choice(['Sorry I couldn\'nt answer :(',
                                  'Ok so let\'s start from the beggining!',
                                  'OK! let\'s start from the beginning!'])

        return {
            "msg": response,
            "intention": "wants_answer",
            "end_of_conversation": False,
        }

    if state == 'deny':
        missing = load_memory_files()[0]
        return {
            "msg": missing_ask_again(missing),
            "intention": "wants_answer",
            "end_of_conversation": False,
        }
