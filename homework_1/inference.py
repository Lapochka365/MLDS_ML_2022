from datetime import datetime
import json
from json.decoder import JSONDecodeError
import os
import warnings
from typing import List, Union

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Form, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import Pipeline


warnings.simplefilter("ignore", category=FutureWarning)

SAVE_FOLDER_CSVS = "csvs"
SAVE_FOLDER_CSVS_PRED = "predicted_csvs"
if not os.path.exists(SAVE_FOLDER_CSVS):
    os.mkdir(SAVE_FOLDER_CSVS)
if not os.path.exists(SAVE_FOLDER_CSVS_PRED):
    os.mkdir(SAVE_FOLDER_CSVS_PRED)

SAVE_FOLDER_MODEL = "saved_models"

if os.path.exists(SAVE_FOLDER_MODEL):
    if times_of_saves := [
        os.path.getmtime(os.path.join(SAVE_FOLDER_MODEL, folder))
        for folder in os.listdir(SAVE_FOLDER_MODEL)
    ]:
        last_saved_model_folder = os.path.join(
            SAVE_FOLDER_MODEL, os.listdir(SAVE_FOLDER_MODEL)[np.argmax(times_of_saves)]
        )
    else:
        last_saved_model_folder = ""
else:
    last_saved_model_folder = ""

if not last_saved_model_folder:
    print("Training model...")

model = Pipeline(
    train_path=(
        "https://raw.githubusercontent.com/Murcha1990/"
        "MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    ),
    test_path=(
        "https://raw.githubusercontent.com/Murcha1990/"
        "MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"
    ),
    categorical_cols=[
        "name",
        "year",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "seats",
        "model",
    ],
    numeric_cols_w_nan=[
        "mileage",
        "engine",
        "max_power",
        "torque",
        "seats",
        "max_torque_rpm",
    ],
    alpha_start=1e-10,
    alpha_end=1e10,
    random_state_start=0,
    random_state_end=240,
    solvers=["auto", "svd", "cholesky", "lsqr"],
    number_of_trials=1000,
    save_path=SAVE_FOLDER_MODEL,
    cv=10,
)

if last_saved_model_folder:
    model.load_model(last_saved_model_folder)
else:
    model.preprocess_datasets()
    model.train_model()
    model.save_model()
print("The App is ready to use")


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: Union[int, None]
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

    @classmethod
    def as_form(cls, request: Request, raw_sample: str = Form(...)):
        json_sample = json.loads(raw_sample)
        try:
            return cls(
                name=json_sample["name"],
                year=json_sample["year"],
                selling_price=json_sample["selling_price"],
                km_driven=json_sample["km_driven"],
                fuel=json_sample["fuel"],
                seller_type=json_sample["seller_type"],
                transmission=json_sample["transmission"],
                owner=json_sample["owner"],
                mileage=json_sample["mileage"],
                engine=json_sample["engine"],
                max_power=json_sample["max_power"],
                torque=json_sample["torque"],
                seats=json_sample["seats"],
            )
        except TypeError:
            return templates.TemplateResponse(
                "json_exception.html", {"request": request}, status_code=500
            )


class WrongFileUpload(Exception):
    def __init__(self, request: Request):
        self.request = request


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


async def internal_exception_handler(request: Request, call_next):
    try:
        return await call_next(request)
    except JSONDecodeError:
        return templates.TemplateResponse(
            "json_exception.html", {"request": request}, status_code=500
        )


app.middleware("http")(internal_exception_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc):
    return templates.TemplateResponse(
        "json_exception.html", {"request": request}, status_code=422
    )


@app.exception_handler(WrongFileUpload)
async def wrong_file_exception_handler(request: Request, exc: WrongFileUpload):
    if request.headers.get("content-type") == "application/json":
        return 0
    return templates.TemplateResponse(
        "wrong_file_exc.html", {"request": request}, status_code=400
    )


@app.get("/")
def load_main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_item_raw")
def predict_item_raw(request: Request, raw_sample: Item = Depends(Item.as_form)):
    predictions = model.predict_samples(pd.DataFrame([raw_sample.dict()], index=[0]))
    predictions = np.round(predictions, 4).tolist()
    return templates.TemplateResponse(
        "response_sample.html", {"request": request, "predict_price": predictions[0]}
    )


@app.post("/predict_item")
def predict_item(request: Request, raw_sample: Item):
    predictions = model.predict_samples(pd.DataFrame([raw_sample.dict()], index=[0]))
    predictions = np.round(predictions, 4).tolist()
    return {"predict_price": predictions[0]}


@app.post("/predict_items")
def predict_items(request: Request, csv_file: UploadFile = File(...)):
    if csv_file.filename.endswith(".csv"):
        file_name = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv"
        save_file = os.path.join(SAVE_FOLDER_CSVS, file_name)
        with open(save_file, "wb") as f:
            for line in csv_file.file:
                f.write(line)
        samples_to_predict = pd.read_csv(save_file)
        try:
            predictions = model.predict_samples(samples_to_predict)
        except Exception:
            raise WrongFileUpload(request)

        predictions = np.round(predictions, 4)

        samples_to_predict = pd.read_csv(save_file)
        samples_to_predict["selling_price_pred"] = predictions

        result_file = os.path.join(SAVE_FOLDER_CSVS_PRED, file_name)
        samples_to_predict.to_csv(result_file, index=False)
        return FileResponse(result_file)

    raise WrongFileUpload(request)
