import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import count

import numpy as np
import pandas as pd
import sqlalchemy
from flask import Flask, Response, render_template, request, send_file
from googleapiclient import discovery
from prediction import cloudml, cloudsql, pipeline

app = Flask(__name__)
logger = logging.getLogger()

BENEFICIARIES_COLUMNS = ["user_id", "name", "phone_no", "channel_type", "channel_name",
    "income_bracket", "call_slots", "entry_date", "ngo_hosp_id", "education", "risk"]
STRING_COLUMNS = ["name", "channel_name", "income_bracket", "entry_date"]

def predict(instances, batch_no):
    """Runs predictions of the given batch of beneficiaries.

    Args:
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        batch_no (int): The batch number currently being processed.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    logging.info("Running prediction model for batch %s", batch_no)
    service = discovery.build('ml', 'v1', cache_discovery=False)
    response = service.projects().predict(
        name=cloudml.ml_service_name,
        body={"instances": instances.tolist()}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    logging.info("Ran prediction model for batch %s", batch_no)

    return (instances, response['predictions'])

def writeBatchToTempPredictionsDatabase(beneficiaries, batch_no):
    """Writes the given batch of beneficiaries to the temp predictions database.

    Args:
        beneficiaries ([pd.DataFrame]): List of beneficiaries to be written.
    Returns:
        None
    """
    query_values = []
    for beneficiary in beneficiaries:
        query_values.append("({})".format(
            ",".join([
                # Surround string column names with quotes.
                '"{}"'.format(str(beneficiary[column]))
                if column in STRING_COLUMNS
                else str(beneficiary[column])
                for column in BENEFICIARIES_COLUMNS
            ])))

    query = sqlalchemy.text(
        "insert into {} "
        "({}) "
        "VALUES {}".format(
            cloudsql.temp_predictions_table,
            ",".join(BENEFICIARIES_COLUMNS),
            ",".join(query_values)))

    logging.info("Inserting batch %s into temp predictions database : %s", batch_no, query)
    try:
        with cloudsql.predictions_database.connect() as conn:
            conn.execute(query)
    except Exception as e:
        print("Failed to execute query %s: %s", query, e)
        logger.exception(e)
        return

def writeToPredictionsDatabase(beneficiary_data):
    """Writes predictions to the cloudsql database.

    Since writing predictions can take some time, and we dont want the live
    database to stop serving requests while new data is being written, we
    use a temporary database to store the new prediction results. Once all
    prediction results have been stored in the temp database, we copy the
    data from the temp database to the live database.
    
    Args:
        beneficiary_data (pd.DataFrame): The data to be written to the
            predictions database.
    Returns:
        None
    """

    # Truncate existing temp table.
    query = sqlalchemy.text("truncate {}".format(cloudsql.temp_predictions_table))
    logging.info("Truncating temp table with query %s", query)
    try:
        with cloudsql.predictions_database.connect() as conn:
            conn.execute(query)
    except Exception as e:
        print("Failed to execute query %s: %s", query, e)
        logger.exception(e)
        return

    beneficiary_data = beneficiary_data.to_dict(orient = "records")

    # Insert beneficiary data into temp predictions database in batches.
    batch_benfs = []
    batch_size = 100
    batch_no = 1
    pos = 0
    for beneficiary in beneficiary_data:
        # Store batch_size number of beneficiaries in batch_benfs.
        if (pos+1) % batch_size != 0:
            pos += 1
            batch_benfs.append(beneficiary)
            continue
        batch_benfs.append(beneficiary)

        writeBatchToPredictionsDatabase(batch_benfs, batch_no)

        batch_benfs = []
        pos = 0
        batch_no += 1

    # Write the last batch to the predictions database.
    if len(beneficiary_data) % batch_size != 0:
        writeBatchToPredictionsDatabase(batch_benfs, batch_no)

    # Replace live database with temp database.
    stmts = [
        "truncate {}".format(cloudsql.live_predictions_table),
        "insert into {} select * from {}".format(cloudsql.live_predictions_table, cloudsql.temp_predictions_table)
    ]

    for index, stmt in enumerate(stmts):
        query = sqlalchemy.text(stmt)
        logging.info("Executing query %s", query)
        try:
            with cloudsql.predictions_database.connect() as conn:
                conn.execute(query)
        except Exception as e:
            print("Failed to execute query %s: %s", query, e)
            logger.exception(e)

def readFromPredictionsDatabase():
    stmt = sqlalchemy.text("SELECT * FROM {}".format(cloudsql.live_predictions_table))
    try:
        with cloudsql.predictions_database.connect() as conn:
            db_result = conn.execute(stmt).fetchall()
            beneficiary_data = pd.DataFrame(db_result)
            beneficiary_data.columns = db_result[0].keys()
            beneficiary_data = beneficiary_data.sort_values(by=['risk', 'name'], ascending=[False, True])
            beneficiary_data = beneficiary_data.drop_duplicates(subset='name')
            beneficiary_data = beneficiary_data.to_dict(orient = "records")
            return beneficiary_data
    except Exception as e:
        print("Failed to execute query %s: %s", stmt, e)
        logger.exception(e)

    return None


def runPredictionModel(beneficiary_data, instances, instance_to_beneficiary_id):
    high_risk_beneficiary_data = pd.DataFrame()
    # Split into batches of 100 instances each.
    instances = instances['instances']
    logging.info('Prediction model input contains %s instances', len(instances))
    instances_batch_count = len(instances) / 100
    instance_batches = np.array_split(instances, instances_batch_count)
    logging.info('Prediction model input split into %s batches', len(instance_batches))
    all_beneficiares = []

    # Execute 10 batches at a time in parallel.
    with ThreadPoolExecutor(max_workers = 10) as executor:
        results = executor.map(predict, instance_batches, count(1))
        for instances, predictions in results:
            beneficiaries_at_high_risk = []
            risks = {}
            for index, prediction in enumerate(predictions):
                # Store beneficiaries whose risk is > 50%.
                if prediction['dense_2'][0] > 0.5:
                    beneficiary_id = int(instance_to_beneficiary_id[json.dumps(instances[index])])
                    # Some beneficiarie IDs are int and some are string so convert all to string for future use.
                    beneficiaries_at_high_risk.append(str(beneficiary_id))
                    risks[beneficiary_id] = prediction['dense_2'][0]

            # Collect additional metadata about the high risk beneficiaries.
            stmt = sqlalchemy.text(
                "select D.*, channel_type, channel_name FROM mmitra.channels C INNER JOIN (select user_id, name, phone_no, income_bracket, call_slots, entry_date, ngo_hosp_id, education from mmitra.users A INNER JOIN mmitra.user_metas B ON A.id = B.id AND B.user_id IN ({})) D ON C.id = D.ngo_hosp_id".format(
                    ','.join(beneficiaries_at_high_risk)))

            try:
                with cloudsql.mmitra_database.connect() as conn:
                    db_result = conn.execute(stmt).fetchall()
                    temp_data = pd.DataFrame(db_result)
                    temp_data.columns = db_result[0].keys()
                    temp_data['risk'] = [risks[int(id)] for id in temp_data["user_id"]]
                    temp_data['name'] = temp_data['name'].str.replace('\\', '')
                    high_risk_beneficiary_data = high_risk_beneficiary_data.append(temp_data)
            except Exception as e:
                print("Failed to execute query %s: %s", stmt, e)
                logger.exception(e)

    logging.info('Found %s high risk beneficiaries', high_risk_beneficiary_data.shape[0])
    return high_risk_beneficiary_data


def getAllowlistedEmails():
    stmt = sqlalchemy.text("SELECT * FROM {}".format(cloudsql.allowlist_table))
    try:
        with cloudsql.predictions_database.connect() as conn:
            db_result = conn.execute(stmt).fetchall()
            email_data = pd.DataFrame(db_result)
            email_data.columns = db_result[0].keys()
            return [str(email) for email in email_data['email'].to_list()]
    
    except Exception as e:
        print("Failed to execute query %s: %s", stmt, e)
        logger.exception(e)


@app.route('/computePredictions', methods=['POST', 'GET'])
def computePredictions():
    beneficiary_data, instances, instance_to_user_id = pipeline.run(cloudsql.mmitra_database)
    print("%d instances. Find example below." % len(instances["instances"]))
    print(instances["instances"][0])

    high_risk_beneficiary_data = runPredictionModel(beneficiary_data, instances, instance_to_user_id)
    high_risk_beneficiary_data = high_risk_beneficiary_data.sort_values(by=['risk', 'name'], ascending=[False, True])
    high_risk_beneficiary_data.drop_duplicates(subset='name')
    writeToPredictionsDatabase(high_risk_beneficiary_data)

@app.route('/')
def root():
    return render_template('index.html')


@app.route('/downloadCsv', methods=['POST'])
def downloadCsv():
    whitelistedEmails = getAllowlistedEmails()

    if str(request.form['authId']) not in whitelistedEmails:
        logging.info("%s is not authorized to retrieve predictions", request.form['authId'])
        return render_template('show_beneficiaries.html', isUserAuthorized=False, authId=request.form['authId'])

    beneficiary_data = readFromPredictionsDatabase()

    csv_data = []
    csv_data.append(','.join(BENEFICIARIES_COLUMNS))
    for index, beneficiary in enumerate(beneficiary_data):
        csv_data.append(','.join(str(beneficiary[column]) for column in BENEFICIARIES_COLUMNS))

    return Response(
        '\n'.join(csv_data),
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=Beneficiaries.csv"})


@app.route('/retrievePredictions', methods=['GET', 'POST'])
def retrievePredictions():
    whitelistedEmails = getAllowlistedEmails()

    if str(request.form['authId']) not in whitelistedEmails:
        logging.info("%s is not authorized to retrieve predictions", request.form['authId'])
        return render_template('show_beneficiaries.html', isUserAuthorized=False, authId=request.form['authId'])

    beneficiary_data = readFromPredictionsDatabase()
    return render_template('show_beneficiaries.html', beneficiaries=beneficiary_data, isUserAuthorized=True)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
