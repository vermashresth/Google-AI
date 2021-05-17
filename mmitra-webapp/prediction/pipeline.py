import logging
import os

from .data import load_data
from .dataset import preprocess_and_make_dataset
from .format import format_for_online_prediction


def run(database):
    logging.info("Running prediction pipeline.")
    beneficiary_data, call_data = load_data(database)
    dataset = preprocess_and_make_dataset(beneficiary_data, call_data)
    instances, instance_to_user_id = format_for_online_prediction(dataset)
    logging.info("Completed running prediction pipeline.")
    return beneficiary_data, instances, instance_to_user_id
