from NLU import NLU
from DP import DP
from aux_functions import *
from components import dataset_mapping
from pprint import pprint

general_intents = load_component('generic_responses').keys()

def DST(user_input, NLU_model):

    #global memory
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

    # filling FALSE
    if not filling:
        print('Detected --> no mind changing')
        message = NLU(preprocess(user_input), NLU_model )
        extracted_info = simplify_NLU(message)
        print(f'extracted info: {extracted_info}')
        # COMPATIBILITY CHECK
        if extracted_info['intent']['name'] not in general_intents:

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
                state = 'unsupported_datatype'
                return DP(state)

            else:
                intent_obj = load_component('intents')
                possible_intents = intent_obj[ddt]['intents']
                if extracted_info['intent']['name'] not in possible_intents:
                    state = 'wrong_question'
                    return DP(state)

        memory_filler(extracted_info)
        memory = load_brain()
        print('*'*40)
        print(f'memory after inserting {extracted_info}:')
        pprint(memory)
        print('*'*40)
        # gets missing entities and insert them into missing.p
        missing_filler(extracted_info) 
        memory = load_brain()
        print('*'*40)
        print('memory after missing filler')
        pprint(memory)
        print('*'*40)
        missing = load_memory_files()[0]
        print(f'missing --> {missing}')
        if not missing:
            print('Detected --> no missing')
            record = extract_num(memory['entities']['record'])
            dataset = memory['dataset']
            dataset_obj = dataset_mapping[dataset]
            num_records = dataset_obj['num_records']
            print(f'\n\nthe record is : {record}')
            if record <= num_records:
                print(f'\n\n{record} is within {num_records} records')
                print('Asking DP to generate an explanation...')
                empty_missing()
                state = 'can proceed with explanation'
                return DP(state)
            else:
                print(f'\n\n{record} is NOT within {num_records} records')
                state = 'record is out of range'
                return DP(state)

        else:
            # missing, filling, mind_change, can_start = load_memory_files()[:4]
            print('Detected --> Not filling')
            save_filling(True)
            print(f'filling --> {filling}')
            state = 'missing_value'
            return DP(state)
    # filling TRUE
    else:
        print('Detected --> Filling mode')
        # MIND CHANGE --> FALSE
        if not mind_change:
            print('Detected --> mind_change=False')
            message = NLU(preprocess(user_input), NLU_model)
            extracted_info = simplify_NLU(message)
            print(f'extracted info --> {extracted_info}')

            # if user gives a short answer, most probably RASA is going to f**k it up
            # in this case we should override the detected intention by RASA
            if len(user_input.split(' ')) <= 2:
                print('Detected --> user input too short')
                print(
                    f"** Replacing '{extracted_info['intent']['name']}' with '{memory['intent']}' **")
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
                    state = 'missing_value'
                    return DP(state)

                # There was just one missing and user filled it
                # XXXXXXXXXXXXXXXXXXXXXXXXXXXX
                else:
                    print('Detected --> no missing value')
                    save_filling(False)
                    state = 'filled_one_missing'
                    return DP(state)

            # if a new intent is detected
            else:
                print(f'the new intent detected --> {new_intent}')
                save_mind_change(True)
                state = 'new_intent_detected'
                return DP(state)

        # if the user changed her mind
        else:
            print('Detected --> mind_change=True')
            message = NLU(preprocess(user_input), NLU_model)
            extracted_info = simplify_NLU(message)
            new_intent = get_intent(extracted_info)
            if new_intent not in ('affirm', 'deny'):
                print('Detected --> An answer which is not yes/no')
                empty_brain()
                empty_missing()
                save_filling(False)
                save_mind_change(False)

                state = 'mind_change_NOT_yes_no'
                return DP(state)
            # if intent either positive or negative
            else:
                print('Detected --> yes/no answer')
                if new_intent == 'affirm':
                    empty_brain()
                    empty_missing()
                    save_filling(False)
                    save_mind_change(False)
                    state = 'mind_change_yes_no'
                    return DP(state)

                # intent is deny --> user doesn't want to start from the beginning
                else:
                    print('Detected --> Deny intent')
                    state = 'deny'
                    return DP(state)
