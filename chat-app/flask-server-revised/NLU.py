from rasa.nlu.model import Interpreter


def NLU(text, NLU_model):
    rasa_model_path = f"../../RASA_NLU/models/{NLU_model}/nlu"

    interpreter = Interpreter.load(rasa_model_path, skip_validation=True)
    message = str(text).strip()
    result = interpreter.parse(message)
    return result
