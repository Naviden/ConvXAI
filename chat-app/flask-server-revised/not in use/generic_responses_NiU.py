import json

generic_responses = {'greet': ['Hi there!', 'Hello!', 'Hi!', 'Hey there!'],
                   'goodbye': ['Goodbye!', 'See you later!', 'Bye! see you later!'],
                   'nlu_fallback': ['Sorry I didn\'t get what you said...could you rephrase it?',
                                    'Hmmm...can you repeat what you said ?',
                                    'Don\'t hate me but I didn\'t understand what you said :(',
                                    'Sorry but it seems I have no answer to that...can you rephrase it ?',
                                    'Sorry but I can\'t understand what you say'],
                   'chitchat_general': ['I have no time for chitchat..human :D',
                                        'We\'ll talk about it later but first ask me something about the blackbox!',
                                        'I\'m the king (or queen ?) of the chitchat but first let\'s see how can I help you with the results!'],
                   'help': ["""I\'m a chat-bot you can ask anything about the outcome of the blackbox model. Here are some examples:\n
                        - Why the model classified instance number N as class X ?
                        - What are the most important features the model used for classifying the instance N ?
                        - What is the most important feature for classifying the instance N ? """],
                   'human_handoff': ['Sorry but my human is not available right now :(',
                                    'Error 404: No human was found !',
                                    'Who needs a human when such a smart bot like me exist ;)Hey '],
                   'thankyou': ['You\'re welcome!',
                                'My pleasure!',
                                'Happy to help! :)']
                                }

with open('generic_responses.json', 'w') as fp:
    json.dump(generic_responses, fp)