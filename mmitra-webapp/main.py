import logging

import pandas as pd
import sqlalchemy
from flask import Flask, render_template, request
from prediction import cloudsql

app = Flask(__name__)
import compute_predictions
logger = logging.getLogger()
 

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


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/retrievePredictions', methods=['GET', 'POST'])
def retrievePredictions():
    allowlistedEmails = getAllowlistedEmails()

    if str(request.form['authId']) not in allowlistedEmails:
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
