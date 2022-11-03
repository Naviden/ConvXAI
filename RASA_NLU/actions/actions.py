rom typing import Dict, Text, Any, List, Union, Optional
import logging
from rasa_sdk import Tracker, Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction, REQUESTED_SLOT
from rasa_sdk.events import (
    SlotSet,
    EventType,
    ActionExecuted,
    SessionStarted,
    Restarted,
    FollowupAction,
)
from actions.parsing import (
    parse_duckling_time_as_interval,
    parse_duckling_time,
    get_entity_details,
    parse_duckling_currency,
)
from actions.profile import create_mock_profile
from dateutil import parser
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


class ActionGiveExplanation(Action):
    def name(self):
        return "action_give_explanation"

    def run(self, dispatcher, tracker, domain):
        instance_num = int(tracker.get_slot("record").split()[1])
        max_instance_num = 716
        if instance_num <= 716:
            explainer = LimeTextExplainer(class_names=class_names)
            c = make_pipeline(vectorizer, rf)

            with open(f'{data_path}/test_data.p', 'wb') as file:
                pickle.dump(newsgroups_test, file)


            exp = explainer.explain_instance(
                newsgroups_test.data[instance_num], c.predict_proba, num_features=6)

            print(exp)

           

        # amount = tracker.get_slot("amount_transferred")
        # if amount:
        #     amount = float(tracker.get_slot("amount_transferred"))
        #     init_instance_num = instance_num + amount
        #     dispatcher.utter_message(
        #         template="utter_changed_instance_num",
        #         init_instance_num=f"{init_instance_num:.2f}",
        #         instance_num=f"{instance_num:.2f}",
        #     )
        #     return [SlotSet("payment_amount", None)]
        # else:
        #     dispatcher.utter_message(
        #         template="utter_instance_num",
        #         init_instance_num=f"{instance_num:.2f}",
        #     )
        #     return [SlotSet("payment_amount", None)]