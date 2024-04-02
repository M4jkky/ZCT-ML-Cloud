import os
import logging
import json
import numpy
import joblib
import pickle 

# debug casti kodu nie su potrebne, ale je dobre ich tam nechat, lebo deletnut a uploadnut novy deployment trva 30 minut :)

# init() sa vola pri vytvoreni noveho deploymentu

def init():
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model_json.pkl" # staci zmenit nazov .pkl suboru noveho modelu...cestu si Azure najde samo
    )

    # check if the model file exists - debug
    if not os.path.exists(model_path):
        raise Exception(f"Model file not found: {model_path}")

    # deserialize the model file back into a sklearn model
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']

    # check if the model is a dictionary - debug
    if isinstance(model, dict):
        raise Exception("The loaded model is a dictionary, not a model.")

    logging.info("Init complete")

# run() sa vola pri kazdom requeste

def run(raw_data):
    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()