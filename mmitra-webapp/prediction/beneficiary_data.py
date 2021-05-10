import logging
import os

import pandas as pd
import sqlalchemy
import swifter

from .data_utils import (age_to_int, ch_type_to_int, education_to_int,
                         get_csv_files, income_bracket_to_int, phowner_to_int,
                         wrap_date_to_int)

logger = logging.getLogger()

USERS_COLUMNS = [
    "enroll_gest_age",
    "registration_date",
    "entry_date",
    "language",
    "call_slots",
    "enroll_delivery_status",
    "ngo_hosp_id"
]

USER_METAS_COLUMNS = [
    "user_id",
    "age",
    "education",
    "phone_owner",
    "income_bracket",
    "g",
    "p",
    "s",
    "l",
    "a"
]

BENEFICIARY_COLUMNS = USERS_COLUMNS + USER_METAS_COLUMNS

ROW_CONVERTERS = {
    "entry_date": wrap_date_to_int("%Y-%m-%d %H:%M:%S"),
    "registration_date": wrap_date_to_int("%Y-%m-%d"),
    "phone_owner": phowner_to_int,
    "education": education_to_int,
    "age": age_to_int,
    "ChannelType": ch_type_to_int,
    "income_bracket": income_bracket_to_int,
}


def _convert_input_row(row):
    for column_name, converter in ROW_CONVERTERS.items():
        column_index = BENEFICIARY_COLUMNS.index(column_name)
        if row[column_index] is None:
            print('Ignoring column %s for row : %s', column_name, row)
            continue
        row[column_index] = converter(row[column_index])


def _read_beneficieries_from_database(database):
    query = sqlalchemy.text(
        "SELECT D.*, C.channel_type as ChannelType FROM (SELECT A.{}, B.{} FROM mmitra.users A INNER JOIN mmitra.user_metas B ON A.id = B.id AND A.entry_date > '2021-01-01') D INNER JOIN channels C on D.ngo_hosp_id = C.id".format(
            ', A.'.join(USERS_COLUMNS), ', B.'.join(USER_METAS_COLUMNS)))
    db_result = None
    data = None

    try:
        with database.connect() as conn:
            db_result = conn.execute(query).fetchall()
            data = pd.DataFrame(db_result)
            data.columns = db_result[0].keys()
            logging.info('Found %s records.', data.shape[0])

            # Transform columns into integers.
            data.drop_duplicates(subset='user_id')
            logging.info('Applying wrap_date_to_int to entry_date')
            data["entry_date"] = data["entry_date"].parallel_apply(
                wrap_date_to_int("%Y-%m-%d %H:%M:%S"))
            logging.info('Applying wrap_date_to_int to registration_date')
            data["registration_date"] = data["registration_date"].parallel_apply(
                wrap_date_to_int("%Y-%m-%d"))
            logging.info('Applying phowner_to_int to phone_owner')
            data["phone_owner"] = data["phone_owner"].parallel_apply(phowner_to_int)
            logging.info('Applying education_to_int to education')
            data["education"] = data["education"].parallel_apply(education_to_int)
            logging.info('Applying age_to_int to age')
            data["age"] = data["age"].parallel_apply(age_to_int)
            logging.info('Applying income_bracket_to_int to income_bracket')
            data["income_bracket"] = data["income_bracket"].parallel_apply(
                income_bracket_to_int)
            data = data.apply(
                pd.to_numeric, axis=1, errors="coerce"
            )
        return data
    except Exception as e:
        print("Failed to execute query %s: %s", query, e)
        logger.exception(e)

    return None


def load_beneficiary_data(database):

    logging.info("Loading and cleaning beneficiary data.")
    beneficiary_data = _read_beneficieries_from_database(database)

    logging.info(
        "Beneficiary data contains data for %d beneficiaries."
        % beneficiary_data.shape[0]
    )

    return beneficiary_data
