import logging
import os

from .data import load_data
from .dataset import preprocess_and_make_dataset
from .format import format_for_online_prediction
from .utils import save_obj, load_obj


def make_dataset(input_dir: str):

    logging.info("Running prediction pipeline.")

    beneficiary_data, call_data = load_data(input_dir)

    dataset = preprocess_and_make_dataset(beneficiary_data, call_data)
    save_obj(dataset, input_dir + "/saves/dataset.pkl")

    return dataset
