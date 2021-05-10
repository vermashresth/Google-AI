import joblib, os, logging


def save_obj(obj, file_path: str, makedir=True):
    """
    Save object in file
    """
    if makedir:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as fl:
        joblib.dump(obj, fl)
    logging.info(f"Created {file_path} successfully")


def load_obj(file_path):
    """
    Load file from path
    """
    with open(file_path, "rb") as fl:
        obj = joblib.load(fl)
    return obj
