import glob
import logging
import os

from .beneficiary_data import load_beneficiary_data
from .call_data import load_call_data


def load_data(database):
    logging.info("Loading data from database ")

    beneficary_data = load_beneficiary_data(database)
    call_data = load_call_data(database, beneficary_data["user_id"].to_list())

    beneficary_data = beneficary_data[
        beneficary_data["user_id"].isin(call_data["user_id"])
    ]

    logging.info("Successfully loaded and cleaned beneficiary and call data.")
    logging.info(
        "Beneficiary data contains data for %d beneficiaries"
        % (beneficary_data.shape[0])
    )
    logging.info(
        "Call data contains %d call records for %d beneficiaries"
        % (call_data.shape[0], call_data["user_id"].nunique())
    )

    return beneficary_data, call_data
