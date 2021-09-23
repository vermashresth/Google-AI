import logging
import glob
import os

from .beneficiary_data import load_beneficiary_data
from .call_data import load_call_data


def load_data(CONFIG):#start_date):#data_dir: str):
    if CONFIG["read_sql"]:
      start_date = CONFIG["pilot_start_date"]
      beneficary_data = load_beneficiary_data(CONFIG)#data_dir)
      call_data = load_call_data(CONFIG, beneficary_data["user_id"]) #data_dir, beneficary_data["user_id"])

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
    else:
      data_dir = CONFIG["pilot_data"]
      if not os.path.exists(data_dir):
         raise FileNotFoundError("Input directory '%s' does not exist." % data_dir)

      logging.info("Loading data from folder '%s'" % (data_dir))

      beneficary_data = load_beneficiary_data(CONFIG)
      call_data = load_call_data(CONFIG, beneficary_data["user_id"]) #data_dir, beneficary_data["user_id"])

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
