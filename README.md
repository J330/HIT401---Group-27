# Black Sigatoka Detection — Group 27

A website that analyses a banana leaf photo and reports whether it shows
Black Sigatoka disease or is healthy. A leaf with a different disease and a
non-banana photo are both things the classifier was never trained on, so
both come back as the same generic "not Black Sigatoka, could be something
else" result — the project does not diagnose other diseases by name, and
disease staging is out of scope.

## Folder structure

```
Capstone_Project/
├── data/            <- add your raw images here (see data/raw/*/README.md)
├── ml/               <- data cleaning, model training, evaluation, explainability
├── backend/          <- Django REST API (see backend/README.md first)
└── frontend/         <- plain HTML/CSS/JS upload page
```

## Order of operations

1. **Add your images**:
   - `data/raw/healthy/` — healthy banana leaf photos
   - `data/raw/black_sigatoka/` — Black Sigatoka photos
   - `data/raw/other_disease/` (optional) — other banana diseases, gate testing only
   - `data/raw/non_banana/` (optional) — non-banana photos, gate testing only

2. Set up the ML environment and clean/split the data:
   ```
   cd ml
   python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python clean_data.py
   python split_data.py
   ```

3. Train all five models (or run each individually):
   ```
   python train_all.py
   ```

4. Evaluate the four closed-set candidates and see which wins:
   ```
   python evaluate.py
   ```

5. Build the DINOv2 open-set gate:
   ```
   python build_gate.py
   ```

6. Sanity-check everything, then test all four image types from the terminal:
   ```
   python test.py
   python predict_cli.py ../data/raw/healthy/<a_file>.jpg
   python predict_cli.py ../data/raw/black_sigatoka/<a_file>.jpg
   python predict_cli.py ../data/raw/other_disease/<a_file>.jpg
   python predict_cli.py ../data/raw/non_banana/<a_file>.jpg
   ```
   The first two should print `healthy` / `black_sigatoka`; the last two
   should both print `not_black_sigatoka`.

7. Set up the backend — **read backend/README.md first**, it explains which
   files Django auto-generates vs. which ones in this zip you copy in.

8. Run the backend locally:
   ```
   cd backend
   python manage.py makemigrations detector
   python manage.py migrate
   python manage.py runserver
   ```

9. Open `frontend/index.html` (or `cd frontend && python -m http.server 8080`)
   to use the upload page. Update `API_BASE_URL` in `frontend/js/api.js`
   first if your backend isn't running on `127.0.0.1:8080`.

## Switching which model serves the website

`backend/backend/settings.py` has `ACTIVE_MODEL`, read from an environment
variable (default `"convnextv2"`). It can be `"efficientnetv2s"`,
`"convnextv2"`, `"swin"`, or `"vit"` — whichever one won in step 4's
`evaluate.py`. Change the environment variable and restart the server
(locally) or update it in the Render dashboard and redeploy (in
production) — no code edit needed either way.

## Deployment

See the Deployment section of the Project Cookbook document for the full
Render setup (build.sh, environment variables, Web Service + Static Site).

## Sources

Every script has a "Source:" comment at the top naming exactly where its
approach and pretrained checkpoint come from (Hugging Face model cards,
official papers, and library documentation).
