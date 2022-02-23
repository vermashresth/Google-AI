from tqdm import tqdm
import logging
import os

import pandas as pd
import numpy as np
import swifter

from .utils import load_obj, save_obj
from .data_utils import (
    get_csv_files,
    date_to_int,
    time_to_int,
    gest_age_to_list,
)


CALL_COLUMNS = [
    "user_id",
    "startdatetime",
    "duration",
    "gest_age",
    "callStatus",
    "dropreason"
]


def _retain_best(data: pd.DataFrame):
    data = data.sort_values(
        ["user_id", "gest_age", "duration", "Technical Status"], ascending=[True, True, False, False]
    ).drop_duplicates(["user_id", "gest_age"])

    return data


def _find_failure_reasons(data: pd.DataFrame):
    failure_reasons = pd.read_csv("training_new/res/failure_reasons.csv", usecols=["failure_reason", "category_id"])
    failure_reasons["failure_reason"] = failure_reasons["failure_reason"].str.lower()

    mapper = dict()
    for i in failure_reasons.index:
        row = failure_reasons.iloc[i, :]
        if row["category_id"] == 1:
            mapper[row["failure_reason"]] = 1
        else:
            mapper[row["failure_reason"]] = 0

    mapper[np.nan] = 0

    data.loc[data["dropreason"].str.startswith("media menu path not", na=False), "dropreason"] = 0
    data.loc[data["duration"]==0, "dropreason"] = data.loc[data["duration"]==0, "dropreason"].str.lower().replace(mapper)
    data.loc[data["duration"]>0, "dropreason"] = 1
    data.loc[~data["dropreason"].isin([0, 1]), "dropreason"] = 0
    data = data.rename(columns={"dropreason": "Technical Status"})

    return data


def _convert_log_data(data: pd.DataFrame):
    data["startdatetime"] = pd.to_datetime(
        data["startdatetime"], 
        format="%Y-%m-%d %H:%M:%S", 
        errors="coerce"
    )
    data.loc[data["startdatetime"].notnull(), "startdate"] = (
        data.loc[data["startdatetime"].notnull(), "startdatetime"] - pd.to_datetime("2018-01-01", format="%Y-%m-%d")
    ).dt.days
    data = data.drop(columns=["startdatetime"])

    # unique_gest_ages = data[["gest_age"]].drop_duplicates()
    # unique_gest_ages["gest_stage"], unique_gest_ages["gest_week_day"], unique_gest_ages["gest_index"] = zip(
    #     *unique_gest_ages["gest_age"].swifter.progress_bar(True).apply(gest_age_to_list)
    # )
    # data = pd.merge(data, unique_gest_ages, on="gest_age", how="left", sort=False)
    # data = data.drop(columns=["gest_age"])

    # cols = ["startdate", "gest_stage", "gest_week_day", "gest_index"]
    # data[cols] = data[cols].swifter.progress_bar(True).apply(
    #     pd.to_numeric, 
    #     errors='coerce', 
    #     downcast="integer",
    #     axis=0
    # )

    return data


def _merge_call_files(
#     filelist: list,
#    start_date,
    CONFIG,
    include_benfs: pd.Series,
):
  if CONFIG['read_sql']:
    start_date = CONFIG['pilot_start_date']
    import mysql.connector
    from mysql.connector.constants import ClientFlag
#    import pandas as pd
    config = {
        'user': 'googleai',
        'password': '4UY(@{SqH{',
        'host': '34.93.237.61',
        'client_flags': [ClientFlag.SSL]
    }

    # now we establish our connection

    config['database'] = 'mmitrav2'  # add new database to config dict
    cnxn = mysql.connector.connect(**config)
    cursor = cnxn.cursor()
    query = "SELECT beneficiary_id user_id, startdatetime, enddatetime, duration, gest_age, dropreason, call_status_id callStatus, missed_call_id missedcall_id, media_id, esb_trans_id, tsp_id\
    FROM vw_call_logs\
    WHERE startdatetime < '"+str(start_date)+"' AND startdatetime >= '2021-02-16';"
    df = pd.read_sql(query, cnxn)
    df = df[CALL_COLUMNS]
    data = pd.concat(
        [   
         df
        ],
        ignore_index=True,
    )
  else:
    filelist = CONFIG['call_files']
    data = pd.concat(
        [   #df
             pd.read_csv(
                 f,
                 usecols=CALL_COLUMNS,
                 low_memory=True,
                 engine="python",
                 error_bad_lines=False,
                 warn_bad_lines=False,
                 sep='\t'
             )
             for f in tqdm(filelist)
        ],
        ignore_index=True,
    )
  print("Shape of call data (0): "+str(data.shape))
  print(data)
    # Setting type to reduce size
  data["callStatus"] = data["callStatus"].astype('int8')

    # Select specific beneficiaries
  data = data[data["user_id"].isin(include_benfs)]
  data["user_id"] = data["user_id"].astype('int32')

  print("Shape of call data obtained (1): "+str(data.shape))
    # dropping rows that are likely to be erroneous
  data = data[(data["duration"] >= 0) & (data["duration"] <= 250)]
  data["duration"] = data["duration"].astype('uint8')
  print("Shape of call data after dropping (2): "+str(data.shape))
    # get the technical status of each call
  data = _find_failure_reasons(data)
  print("Shape of call data (3): "+str(data.shape))
    # Retain maximum for each call
  data = _retain_best(data)
  print("Shape of call data (4): "+str(data.shape))
    # cleaning startdate and gestation age
  data = _convert_log_data(data)
  print("Shape of call data at end of merge call files function (5): "+str(data.shape))
  return data


def load_call_data(CONFIG, beneficiaries: pd.Series = None):#data_dir: str, beneficiaries: pd.Series = None):
  if CONFIG['read_sql']:
    call_data = _merge_call_files(CONFIG, beneficiaries)
    return call_data
  else:
    data_dir = CONFIG['pilot_data']
    call_dir = os.path.join(data_dir, "call")
    if os.path.exists(call_dir):
        call_files = get_csv_files(call_dir)

        if len(call_files) == 0:
             FileNotFoundError("Input directory '%s' is empty." % call_dir)

        logging.info("Found %d files in '%s'." % (len(call_files), call_dir))

        logging.info("Loading and cleaning call data.")
        CONFIG['call_files'] = call_files
        call_data = _merge_call_files(CONFIG, beneficiaries)

        logging.info(
            "Call data contains %d relevant call records for %d beneficiaries."
            % (call_data.shape[0], call_data["user_id"].nunique())
        )

        # save_obj(call_data, data_dir + "/saves/c_call.pkl")

        return call_data

    else:
        raise FileNotFoundError("Input directory '%s' does not exist." % call_dir)
