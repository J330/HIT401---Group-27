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
├── data/            
├── ml/               
├── backend/          
└── frontend/         
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

7. Set up the backend — **read backend/README.md first**

8. Run the backend locally:
   ```
   cd backend
   python manage.py makemigrations detector
   python manage.py migrate
   python manage.py runserver
   ```

## Switching which model serves the website

`backend/backend/settings.py` has `ACTIVE_MODEL`, read from an environment
variable (default `"convnextv2"`). It can be `"efficientnetv2s"`,
`"convnextv2"`, `"swin"`, or `"vit"` — whichever one won in step 4's
`evaluate.py`. Change the environment variable and restart the server
(locally).
