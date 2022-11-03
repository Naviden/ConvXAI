data =  {'entities': [{'confidence_entity': 0.9975384473800659,
               'end': 52,
               'entity': 'record',
               'extractor': 'DIETClassifier',
               'start': 50,
               'value': '22'}],
 'intent': {'confidence': 0.9115745425224304,
            'id': 5180335951269843842,
            'name': 'most_important_feature'},
 'intent_ranking': [{'confidence': 0.9115745425224304,
                     'id': 5180335951269843842,
                     'name': 'most_important_feature'},
                    {'confidence': 0.05457819625735283,
                     'id': 2004188661668502295,
                     'name': 'list_features'},
                    {'confidence': 0.017276303842663765,
                     'id': 1929837160119120724,
                     'name': 'why_not'},
                    {'confidence': 0.006664602551609278,
                     'id': -2444828015714000824,
                     'name': 'deny'},
                    {'confidence': 0.004060497507452965,
                     'id': -1943994785725708487,
                     'name': 'goodbye'},
                    {'confidence': 0.003065915545448661,
                     'id': 1803021602666985055,
                     'name': 'what_if_subs'},
                    {'confidence': 0.0015303411055356264,
                     'id': 633066712098676026,
                     'name': 'affirm'},
                    {'confidence': 0.0010170143796131015,
                     'id': -3245058244615762965,
                     'name': 'greet'},
                    {'confidence': 0.00013639350072480738,
                     'id': 1157944775743005363,
                     'name': 'why_this'},
                    {'confidence': 9.624343510949984e-05,
                     'id': -1307633128415191196,
                     'name': 'mood_unhappy'}],
 'response_selector': {'all_retrieval_intents': [],
                       'default': {'ranking': [],
                                   'response': {'confidence': 0.0,
                                                'id': None,
                                                'intent_response_key': None,
                                                'response_templates': None,
                                                'template_name': 'utter_None'}}},
 'text': 'what are the most important features for isnatnce 22 ?'}

data = {'entities': [{'confidence_entity': 0.9976880550384521,
               'end': 28,
               'entity': 'record',
               'extractor': 'DIETClassifier',
               'start': 27,
               'value': '5'},
              {'confidence_entity': 0.996131420135498,
               'end': 44,
               'entity': 'record',
               'extractor': 'DIETClassifier',
               'start': 33,
               'value': 'instance 11'}],
 'intent': {'confidence': 0.5580466985702515,
            'id': -5790555042134992475,
            'name': 'what_if_subs'},
 'intent_ranking': [{'confidence': 0.5580466985702515,
                     'id': -5790555042134992475,
                     'name': 'what_if_subs'},
                    {'confidence': 0.41184428334236145,
                     'id': 3526346185880995428,
                     'name': 'why_not'},
                    {'confidence': 0.019504118710756302,
                     'id': 2277899924101477218,
                     'name': 'deny'},
                    {'confidence': 0.004479675553739071,
                     'id': 3038686348013310615,
                     'name': 'most_important_feature'},
                    {'confidence': 0.004242477007210255,
                     'id': -4369594230516457956,
                     'name': 'list_features'},
                    {'confidence': 0.0005639205337502062,
                     'id': -7055600101148192387,
                     'name': 'affirm'},
                    {'confidence': 0.00046835042303428054,
                     'id': -1337348209068318293,
                     'name': 'greet'},
                    {'confidence': 0.0003540215257089585,
                     'id': 2289638269031396005,
                     'name': 'why_this'},
                    {'confidence': 0.0002589413779787719,
                     'id': -8669579071997265861,
                     'name': 'mood_great'},
                    {'confidence': 0.0002374960749875754,
                     'id': 6493966492409305435,
                     'name': 'mood_unhappy'}],
 'response_selector': {'all_retrieval_intents': [],
                       'default': {'ranking': [],
                                   'response': {'confidence': 0.0,
                                                'id': None,
                                                'intent_response_key': None,
                                                'response_templates': None,
                                                'template_name': 'utter_None'}}},
 'text': 'what if I change the f1 to 5 for instance 11 ?'}

def clean_output(data):
    intent = data['intent']['name']
    print(f'Identified intent: {intent}')
    
    ent = data.get('entities', 'None')
    if ent != 'None':
        print('\nExtracted entities:')
        for e in ent:
            print(f"\t{e['entity']}: {e['value']}")

clean_output(data)