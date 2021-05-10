{"instances": instances.tolist()}import logging
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import count, repeat

import numpy as np
import pandas as pd
import sqlalchemy
import swifter

from .data_utils import (date_to_int, gest_age_to_list, get_csv_files,
                         time_to_int)

logger = logging.getLogger()

CALL_COLUMNS = [
    "user_id",
    "startdatetime",
    "duration",
    "gest_age",
    "callStatus",
]


def _retain_best(data: pd.DataFrame):
    data = data.sort_values(
        ["user_id", "gest_age", "duration"], ascending=[True, True, False]
    ).drop_duplicates(["user_id", "gest_age"])

    return data


def _read_calls_for_batch(batch_benfs, database):
    batch_no, beneficiaries = batch_benfs
    call_data_query = sqlalchemy.text(
        "SELECT {} FROM mmitra.dialer_logs where user_id IN ({})".format(
            ','.join(CALL_COLUMNS), ','.join([str(int(x)) for x in beneficiaries]))
    )

    try:
        with database.connect() as conn:
            db_result = conn.execute(call_data_query).fetchall()
            logging.info("Executed call_data query for batch %s", batch_no)            
            call_data = pd.DataFrame(db_result)
            if len(db_result) == 0:
                logging.warn("Got empty result for call data query %s", call_data_query)
                return None
            call_data.columns = db_result[0].keys()
            return call_data
    except Exception as e:
        print("Failed to execute query %s: %s", call_data_query, e)
        logger.exception(e)

    return None

def _preprocess_call_data(call_data, batch_no):
    logging.info("Converting call log data for batch %s", batch_no)
    # Dropping rows that are likely to be erroneous
    call_data = call_data[(call_data["duration"] >= 0) & (call_data["duration"] <= 300)]

    logging.info('Applying date_to_int to startdatetime')
    call_data["startdate"] = (
        call_data["startdatetime"].apply(date_to_int, fmt="%Y-%m-%d %H:%M:%S",)
    )
    call_data = call_data.drop(columns=["startdatetime"])

    logging.info('Applying gest_age_to_list to gest_age')
    call_data["gest_stage"], call_data["gest_week_day"], call_data["gest_index"] = zip(
        *call_data["gest_age"].apply(gest_age_to_list)
    )
    call_data = call_data.drop(columns=["gest_age"])

    try:
        logging.info('Applying to_numeric to call_data for batch: %s', batch_no)
        call_data = call_data.apply(pd.to_numeric, axis=1)
    except Exception as e:
        logging.info("Skipped log data for batch %s: %s\nFull Batch: %s", batch_no, e, call_data.to_string().replace('\n', '\n\t'))
        logging.info("datatypes : %s", call_data.info())
        return None
    logging.info("Converted log data for batch %s", batch_no)

    return call_data


def _read_calls_from_database(database, include_benfs):
    """Read call data for beneficiaries in multiple batches, parallely."""

    multiple_batch_benfs = []
    batch_benfs = []
    batch_size = 400
    no_of_parallel_batches = 10
    batch_no = 1

    db_result = None
    call_data = pd.DataFrame()

    for index, beneficiary in enumerate(include_benfs):
        if (index+1) % batch_size != 0:
            batch_benfs.append(beneficiary)
            continue

        batch_benfs.append(beneficiary)

        multiple_batch_benfs.append((batch_no, batch_benfs))
        if len(multiple_batch_benfs) == no_of_parallel_batches:
            with ThreadPoolExecutor(max_workers = no_of_parallel_batches) as executor:
                results = executor.map(_read_calls_for_batch, multiple_batch_benfs, repeat(database))
                for result in results:
                    call_data = call_data.append(result)
            multiple_batch_benfs = []


        batch_benfs = []
        batch_no += 1
    if len(include_benfs) % batch_size != 0:
        multiple_batch_benfs.append((batch_no, batch_benfs))
        with ThreadPoolExecutor(max_workers = no_of_parallel_batches) as executor:
            results = executor.map(_read_calls_for_batch, multiple_batch_benfs, repeat(database))
            for result in results:
                call_data = call_data.append(result)

    # Retain maximum for each call
    logging.info('call_data contains %d records', call_data.shape[0])
    call_data = _retain_best(call_data)
    logging.info('call_data contains %d records after _retain_best', call_data.shape[0])

    # Split call_data into list of dataframes.
    call_data_batch_count = call_data.shape[0]/10000
    call_data_batches = np.array_split(call_data, call_data_batch_count)
    logging.info('call_data split into %s batches', len(call_data_batches))
    preprocessed_call_data = pd.DataFrame()

    with ThreadPoolExecutor(max_workers = 30) as executor:
        results = executor.map(_preprocess_call_data, call_data_batches, count(1))
        for result in results:
            if result is not None:
                preprocessed_call_data = preprocessed_call_data.append(result)

    return preprocessed_call_data

def load_call_data(database, beneficiaries):
    logging.info("Loading and cleaning call data.")
    call_data = _read_calls_from_database(database, beneficiaries)

    logging.info(
        "Call data contains %d relevant call records for %d beneficiaries."
        % (call_data.shape[0], call_data["user_id"].nunique())
    )

    return call_data
