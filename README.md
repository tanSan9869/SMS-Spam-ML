**ML Spam / Phishing Classifier**

- **Project:** Lightweight text classifier for detecting spam / phishing messages.
- **Stack:** Python (Flask, scikit-learn, pandas, NLTK) for the backend; React + Vite for the frontend.

**Overview**
- This repository contains a training pipeline and a simple REST API that serves a spam/phishing text classifier. The frontend provides a minimal UI to send text to the backend and view predictions.

**Repository Structure**
- `Backend` : Python service, training script, datasets, and model artifacts.
- `Frontend` : React + Vite UI to interact with the API.

Key files:
- [Backend/app.py](Backend/app.py) : Flask API (POST `/predict`) that returns `Spam` or `Not Spam`.
- [Backend/train.py](Backend/train.py) : Training script that produces model and vectorizer artifacts.
- [Backend/spam.csv](Backend/spam.csv) : Default dataset used by `train.py`.
- [Frontend/package.json](Frontend/package.json) : Frontend dependencies & scripts.

**Quickstart (Windows)**

1) Backend (Python)

- Create and activate a virtual environment, then install required packages:

```powershell
cd Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install flask flask-cors scikit-learn pandas nltk numpy
```

- (Optional) Download NLTK stopwords used by the service:

```powershell
python -c "import nltk; nltk.download('stopwords')"
```

- Train the default model (will save `model.pkl`, `vectorizer.pkl`, and `metrics.json`):

```powershell
python train.py --dataset spam.csv
```

- If you trained on a different dataset and the produced artifact names are `model_<slug>.pkl` / `vectorizer_<slug>.pkl`, either copy/rename them to `model.pkl` / `vectorizer.pkl` or update `app.py` to load different filenames.

- Run the API:

```powershell
python app.py
# The Flask app listens by default on http://127.0.0.1:5000
```

2) Frontend (Node.js)

- From the project root run:

```powershell
cd Frontend
npm install
npm run dev
```

- Open the URL shown by Vite (usually `http://localhost:5173`) to use the UI.

**API Usage**
- Endpoint: `POST /predict` on the backend server (default port `5000`).
- Request body (JSON): `{ "message": "your text here" }`

Curl example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
	-H "Content-Type: application/json" \
	-d '{"message":"Congratulations — you won a prize! Click here."}'
```

Response example:

```json
{
	"prediction": "Spam",
	"model_accuracy": 0.983
}
```

**Training options**
- `train.py` exposes several CLI flags. Useful ones:
- `--dataset` : path to CSV file (default: `spam.csv`)
- `--text_col` / `--label_col` : column names when auto-detection fails
- `--test_size` : test split fraction (default `0.2`)
- `--finetune` : run GridSearchCV to tune hyperparameters

Example with finetuning:

```powershell
python train.py --dataset spam.csv --finetune True
```

**Notes & Tips**
- `app.py` attempts to load `model.pkl` and `vectorizer.pkl` from the `Backend` folder; ensure those exist before starting the API.
- `train.py` will also write a `metrics.json` file summarizing results. A baseline `metrics_spam.json` can be used to compare datasets.
- If you see NLTK errors, run the NLTK download command shown above.

**Contributing**
- Improvements welcome: add a `requirements.txt`, CI checks, tests, or a Dockerfile for reproducible runs.

**License**
- This project is provided as-is. Add a license file if you intend to open-source the code.

