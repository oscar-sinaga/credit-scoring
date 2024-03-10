from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as util
import data_pipeline as data_pipeline
import preprocessing as preprocessing
import numpy as np

config_data = util.load_config()
le_encoder = util.pickle_load(config_data["le_encoder_path"])
model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    monthly_income : int
    housing_type : str
    num_of_dependent : int
    lama_bekerja : int
    otr : int
    status_pernikahan : str
    pekerjaan : str
    tenor : int
    dp : int

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)
     # Convert dtype
    data = pd.concat(
        [
            data[config_data["object_columns"][:-1]].astype(np.str_),
            data[config_data["int64_columns"][:]].astype(np.int64)
        ],
        axis = 1
    )

    # Check range data
    try:
        data_pipeline.check_data(data, config_data, True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # Log Transform columns_being_logtransformed
    data = preprocessing.log_transform(data,config_data)

    # Encoding Categorical Variabel 
    data = preprocessing.ohe_transform_all(data,config_data)

    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)
    y_pred_proba = round(model_data["model_data"]["model_object"].predict_proba(data)[0][0] *100,2) 

    # Inverse tranform
    y_pred = list(le_encoder.inverse_transform(y_pred))[0] 

    return {"res" : y_pred, "prob":y_pred_proba, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
