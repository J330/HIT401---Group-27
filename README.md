# Black Sigatoka Detection — Group 27

A website that analyses a banana leaf photo and reports whether it shows
Black Sigatoka disease or is healthy. A leaf with a different disease and a
non-banana photo are both things the classifier was never trained on, so
both come back as the same generic "not Black Sigatoka, could be something
else" result — the project does not diagnose other diseases by name, and
disease staging is out of scope.

## Prerequisites

Install these before starting:
- **Python 3.10 or 3.11** (`python --version` to check) — https://www.python.org/downloads/
- **pip** (comes with Python)
- **Git** (optional, only if cloning rather than using the zip) — https://git-scm.com/
- A code editor (VS Code recommended)
- ~5 GB free disk space (pretrained model weights download automatically on first run)
- A GPU is optional — everything here runs on CPU, just slower to train

## Folder structure

Capstone_Project/
├── data/
├── ml/
├── backend/
└── frontend/


## Order of operations

### 1. Add your images
- `data/raw/healthy/` — healthy banana leaf photos
- `data/raw/black_sigatoka/` — Black Sigatoka photos
- `data/raw/other_disease/` (optional) — other banana diseases, gate testing only
- `data/raw/non_banana/` (optional) — non-banana photos, gate testing only

### 2. Set up the ML environment and clean/split the data

cd ml
python -m venv venv && source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
python clean_data.py
python split_data.py


### 3. Train all five models (or run each individually)

python train_all.py


### 4. Evaluate the four closed-set candidates and see which wins

python evaluate.py


### 5. Build the DINOv2 open-set gate

python build_gate.py


### 6. Sanity-check everything, then test all four image types from the terminal

python test.py
python predict_cli.py ../data/raw/healthy/<a_file>.jpg
python predict_cli.py ../data/raw/black_sigatoka/<a_file>.jpg
python predict_cli.py ../data/raw/other_disease/<a_file>.jpg
python predict_cli.py ../data/raw/non_banana/<a_file>.jpg

The first two should print `healthy` / `black_sigatoka`; the last two
should both print `not_black_sigatoka`.

### 7. Run the backend locally

python manage.py makemigrations detector
python manage.py migrate
python manage.py runserver

Leave this terminal running. Confirm it worked by opening
`http://127.0.0.1:8000/api/docs/` in a browser — you should see a JSON
response listing the `/api/predict/` endpoint.

### 8. Run the frontend

Open a **second, separate terminal** (keep the backend one running) and:

cd frontend
python -m http.server 8080


Then open `http://127.0.0.1:8080/upload.html` in your browser.

#Afer clonning Full Structure Repo
## Prerequisites

- Python 3.10 or 3.11
- Git
- ~5 GB free disk space

## Setup — Run These Commands In Order

```
git clone YOUR-REPO-URL
cd Capstone_Project
```

cd ml
python -m venv venv


**Windows:**

venv\Scripts\activate

**Mac/Linux:**

source venv/bin/activate

pip install -r requirements.txt


(Optional) Confirm the pushed models actually work before touching the backend:

python test.py
python predict_cli.py ../data/raw/healthy/<a_file>.jpg
python predict_cli.py ../data/raw/black_sigatoka/<a_file>.jpg

First should print `healthy`, second `black_sigatoka`. If both come back
correctly, the models in `ml/models/` are good — skip straight to Backend.

## Backend

cd ../backend
pip install -r requirements.txt
django-admin startproject backend .
python manage.py startapp detector


Overwrite the just-generated files with the pre-written ones already in
this repo's `backend/` folder:
`backend/backend/settings.py`, `backend/backend/urls.py`,
`backend/detector/models.py`, `serializers.py`, `admin.py`, `api_docs.py`,
`inference.py`, `views.py`, `urls.py`

python manage.py makemigrations detector
python manage.py migrate
python manage.py runserver


Leave this running. Verify at `http://127.0.0.1:8000/api/docs/`.

## Frontend

**New terminal window:**

cd Capstone_Project/frontend
python -m http.server 8080


Check `frontend/js/api.js` has:
```javascript
const API_BASE_URL = "http://127.0.0.1:8000";
```

Open `http://127.0.0.1:8080/upload.html` in your browser.

## Two terminals must stay open

- Terminal A: Django, port 8000
- Terminal B: frontend server, port 8080