# Heart Disease Prediction API

A FastAPI service that predicts the presence of heart disease from 13 clinical
features. The model is a scikit-learn pipeline (StandardScaler + Logistic
Regression) trained on the Heart Disease dataset and saved with joblib.

Module 17 assignment: FastAPI + Docker + Deployment.

## Live deployment

https://REPLACE-WITH-YOUR-RENDER-URL.onrender.com

Swagger UI is at `/docs` on the same host.

## Project structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py           FastAPI app and endpoints
│   └── schemas.py        Pydantic request and response models
├── data/
│   └── heart.csv         training data
├── model/
│   ├── heart_model.joblib
│   └── metadata.json     model type, feature list, accuracy
├── train.py              trains the model and writes model/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Endpoints

| Method | Path       | Description                          |
| ------ | ---------- | ------------------------------------ |
| GET    | `/health`  | service status and model load state  |
| GET    | `/info`    | model type, feature list, accuracy   |
| POST   | `/predict` | returns `heart_disease: true/false`  |
| GET    | `/docs`    | Swagger UI                           |

### Input features

| Field      | Type  | Meaning                                        |
| ---------- | ----- | ---------------------------------------------- |
| `age`      | int   | age in years                                   |
| `sex`      | int   | 1 = male, 0 = female                           |
| `cp`       | int   | chest pain type, 0 to 3                        |
| `trestbps` | int   | resting blood pressure in mm Hg                |
| `chol`     | int   | serum cholesterol in mg/dl                     |
| `fbs`      | int   | fasting blood sugar > 120 mg/dl, 0 or 1         |
| `restecg`  | int   | resting ECG result, 0 to 2                     |
| `thalach`  | int   | maximum heart rate achieved                    |
| `exang`    | int   | exercise induced angina, 0 or 1                |
| `oldpeak`  | float | ST depression induced by exercise              |
| `slope`    | int   | slope of the peak exercise ST segment, 0 to 2  |
| `ca`       | int   | major vessels coloured by fluoroscopy, 0 to 4  |
| `thal`     | int   | thalassemia result, 0 to 3                     |

## Running with Docker

```bash
docker-compose build
docker-compose up
```

Then open http://localhost:8000/docs

To stop it:

```bash
docker-compose down
```

## Running without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Retraining the model

```bash
python train.py
```

This reads `data/heart.csv`, trains the pipeline and overwrites
`model/heart_model.joblib` and `model/metadata.json`.

## Example requests

Health check:

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "model_loaded": true }
```

Model info:

```bash
curl http://localhost:8000/info
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

```json
{ "heart_disease": true, "probability": 0.8745, "label": "heart disease" }
```

## Deploying on Render

1. Push this repository to GitHub.
2. On Render, create a new Web Service and connect the repository.
3. Set the language to Docker. The root directory stays empty because the
   Dockerfile is at the project root.
4. Deploy. Render injects a `PORT` environment variable and the container
   startup command binds uvicorn to it.
5. Once the build finishes, test `/health`, `/info` and `/predict` on the
   generated `.onrender.com` URL.

## Model

Logistic Regression inside a pipeline with standard scaling. The data is split
80/20 with stratification and a fixed random seed. Test accuracy is around 0.80.
The goal of the assignment is the Docker and deployment workflow, not maximum
accuracy.

## Dataset

Heart Disease dataset (UCI Cleveland), 303 rows and 14 columns. The `target`
column is 1 when heart disease is present and 0 when it is absent.
