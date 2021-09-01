import logging
import os

import pandas as pd
import swifter

from .data_utils import (
    get_csv_files,
    wrap_date_to_int,
    phowner_to_int,
    education_to_int,
    age_to_int,
    ch_type_to_int,
    income_bracket_to_int,
    enroll_delivery_status_to_int
)
from .utils import save_obj, load_obj


BENEFICIARY_COLUMNS = [
    "user_id",
    "enroll_gest_age",
    "registration_date",
    "entry_date",
    "language",
    "age",
    "education",
    "phone_owner",
    "call_slots",
    "enroll_delivery_status",
    "ChannelType",
    "income_bracket",
    "ngo_hosp_id",
    "g",
    "p",
    "s",
    "l",
    "a",
]


def _merge_beneficiary_files(
    file_list: list,
    converters={
        "entry_date": wrap_date_to_int("%Y-%m-%d %H:%M:%S"),
        "registration_date": wrap_date_to_int("%Y-%m-%d"),
        "phone_owner": phowner_to_int,
        "education": education_to_int,
        "age": age_to_int,
        "ChannelType": ch_type_to_int,
        "income_bracket": income_bracket_to_int,
        "enroll_delivery_status": enroll_delivery_status_to_int
    },
):

    data = pd.concat(
        [
            pd.read_csv(
                f,
                usecols=BENEFICIARY_COLUMNS,
                converters=converters,
                low_memory=True,
                engine="python",
                error_bad_lines=False,
                warn_bad_lines=False,
                sep='\t'
            )
            for f in file_list
        ],
        ignore_index=True,
    )

    data = data.swifter.progress_bar(True).apply(
        pd.to_numeric, axis=1, errors="coerce"
    )

    return data


def load_beneficiary_data(data_dir: str):
    #if os.path.exists(data_dir + "/saves/c_beneficiary.pkl"):
    #    beneficiary_data = load_obj(data_dir + "/saves/c_beneficiary.pkl")
    #    return beneficiary_data

    beneficiary_dir = os.path.join(data_dir, "beneficiary")
    if os.path.exists(beneficiary_dir):
        beneficiary_files = get_csv_files(beneficiary_dir)

        if len(beneficiary_files) == 0:
            FileNotFoundError("Input directory '%s' is empty." % beneficiary_dir)

        logging.info(
            "Found %d files in '%s'." % (len(beneficiary_files), beneficiary_dir)
        )

        logging.info("Loading and cleaning beneficiary data.")
        beneficiary_data = _merge_beneficiary_files(beneficiary_files)
        #save_obj(beneficiary_data, data_dir + "/saves/c_beneficiary.pkl")

        logging.info(
            "Beneficiary data contains data for %d beneficiaries."
            % beneficiary_data.shape[0]
        )

        #save_obj(beneficiary_data, data_dir + "/saves/c_beneficiary.pkl")

        return beneficiary_data

    else:
        raise FileNotFoundError(
            "Input directory '%s' does not exist." % beneficiary_dir
        )
