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
#    "a",
]


def _merge_beneficiary_files(
    CONFIG,
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
  if CONFIG['read_sql']:
    import mysql.connector
    from mysql.connector.constants import ClientFlag
    #import pandas as pd
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
    query = "SELECT u.beneficiary_id user_id, isactive, phone_no, lmp_date lmp, enrollment_gestation_age enroll_gest_age, u.project_id, u.call_slot_id call_slots, enrollment_delivery_status enroll_delivery_status,\
    language_id language, registration_date, delivery_date, entry_date, phone_type, phone_code, phone_owner,\
    u.channel_id ngo_hosp_id, CASE c.channel_type WHEN 1 THEN 'Community' WHEN 2 THEN 'Hospital' ELSE 'ARMMAN' END AS ChannelType,\
    unique_id unique_sub_id, entry_madeby, entry_updatedby,\
    forced_delivery_update force_delivery_updated, completed,  dnd_optout_status, age, education_id education,\
    alternate_phone_no, alternate_phone_owner alternate_no_owner, name_of_sakhi, name_of_project_officer, income_bracket, data_entry_officer,\
    g, p, s, l, a, ppc_bloodpressure, ppc_diabetes, ppc_cesarean, ppc_thyroid,  ppc_fibroid, ppc_spontaneousAbortion, ppc_heightLess140,\
    ppc_pretermDelivery, ppc_anaemia, ppc_otherComplications, name_of_medication_any, planned_place_of_delivery, registered_where, registered_pregnancy,\
    place_of_delivery, type_of_delivery, date_registration_hospital, term_of_delivery, medication_after_delivery\
    FROM vw_beneficiaries u\
    LEFT OUTER JOIN call_slot csl ON csl.call_slot_id = u.call_slot_id\
    LEFT OUTER JOIN channels c ON c.channel_id = u.channel_id\
    WHERE u.isactive = 1 AND registration_date >= '"+str(CONFIG["from_registration_date"])+"'\
    ORDER BY u.beneficiary_id;"
    df = pd.read_sql(query, cnxn)
    #df.to_csv("beneficiary_data.csv")
    df = df[BENEFICIARY_COLUMNS]
    for i in list(converters.keys()):
        df[i] = df[i].apply(converters[i])
    data = pd.concat(
        [
            df
        ],
        ignore_index=True,
    )

  else:
    file_list = CONFIG['file_list']
    data = pd.concat(
        [#df 
             pd.read_csv(
                 f,
                 usecols=BENEFICIARY_COLUMNS,
                 converters=converters,
                 low_memory=True,
                 engine="python",
                 error_bad_lines=False,
                 warn_bad_lines=False,
 #                sep='\t'
             )
             for f in file_list
        ],
        ignore_index=True,
    )

  data = data.swifter.progress_bar(True).apply(
        pd.to_numeric, axis=1, errors="coerce"
    )

  return data


def load_beneficiary_data(CONFIG):#data_dir: str):
    if CONFIG["read_sql"]:
        logging.info("Loading and cleaning beneficiary data.")
        beneficiary_data = _merge_beneficiary_files(CONFIG)
        #save_obj(beneficiary_data, data_dir + "/saves/c_beneficiary.pkl")

        logging.info(
            "Beneficiary data contains data for %d beneficiaries."
            % beneficiary_data.shape[0]
        )

        #save_obj(beneficiary_data, data_dir + "/saves/c_beneficiary.pkl")

        return beneficiary_data
    else:
      data_dir = CONFIG["pilot_data"]
      beneficiary_dir = os.path.join(data_dir, "beneficiary")
      CONFIG["beneficiary_dir"] = beneficiary_dir
      if os.path.exists(beneficiary_dir):
        beneficiary_files = get_csv_files(beneficiary_dir)#CONFIG)
        CONFIG["file_list"] = beneficiary_files
        if len(beneficiary_files) == 0:
             FileNotFoundError("Input directory '%s' is empty." % beneficiary_dir)

        logging.info(
             "Found %d files in '%s'." % (len(beneficiary_files), beneficiary_dir)
         )

        logging.info("Loading and cleaning beneficiary data.")
        beneficiary_data = _merge_beneficiary_files(CONFIG)
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
