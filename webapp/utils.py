import os
import pickle


def load_model(filename):
    with open(filename, "rb") as fb:
        return pickle.load(fb)


def load_all_models():
    models = os.listdir("models")
    model_dict = {}
    for m in models:
        model_dict[os.path.splitext(m)[0]] = load_model("models/{}".format(m))
    return model_dict