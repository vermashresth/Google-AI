import sqlalchemy

# Database connections.
db_user = "mmitra-google"
db_pass = "google"
mmitra_db_name = "mmitra"
mmitra_cloud_sql_connection_name = "mmitra:asia-south1:armman"
predictions_db_name = "predictions"
predictions_cloud_sql_connection_name = "mmitra:asia-south1:mmitra-predictions"
local_db_hostname = "127.0.0.1"
local_db_port = "3306"

# Table names.
temp_predictions_table = "predictions.high_risk_beneficiaries_temp"
live_predictions_table = "predictions.high_risk_beneficiaries"
allowlist_table = "predictions.whitelisted_users"

mmitra_database = sqlalchemy.create_engine(
    sqlalchemy.engine.url.URL(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_pass,
        database=mmitra_db_name,
        # Use host, port instead of query below to connect to local cloudsql database.
        # host=local_db_hostname,  # e.g. "127.0.0.1"
        # port=local_db_port,  # e.g. 3306
        query={"unix_socket": "/cloudsql/{}".format(mmitra_cloud_sql_connection_name)}
    ),
    pool_size=15,
    max_overflow=0,
    pool_timeout=30,
    pool_recycle=1800
)

predictions_database = sqlalchemy.create_engine(
    sqlalchemy.engine.url.URL(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_pass,
        database=predictions_db_name,
        # Use host, port instead of query below to connect to local cloudsql database.
        # host=local_db_hostname,  # e.g. "127.0.0.1"
        # port=local_db_port,  # e.g. 3306
        query={"unix_socket": "/cloudsql/{}".format(predictions_cloud_sql_connection_name)},
    ),
    pool_size=15,
    max_overflow=0,
    pool_timeout=30,
    pool_recycle=1800
)
