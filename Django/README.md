# Banana Leaf Detector — binary DINOv2 Stage 1

This Django version uses your already-trained **DINOv2 Base disease model** in
two different ways.

## Pipeline

```text
Uploaded image
    ↓
Stage 1 — DINOv2 visual similarity
    ├─ Banana Leaf
    └─ Not Banana Leaf
          ↓
Only Banana Leaf continues
          ↓
Stage 2 — health classification
    ├─ Black Sigatoka
    ├─ Cordana
    ├─ Freckle
    ├─ Healthy
    └─ Pestalotiopsis
```

Stage 1 no longer displays the five disease classes. It compares the uploaded
image against two separate reference banks only: **Banana Leaf** and
**Not BananaLeaf**.

## Required model/reference files

```text
Project/
├── Django/
├── initialscanai/                     # legacy; Stage 1 no longer uses it
└── trainedai/
    └── dinov2-base/
        ├── dinov2_banana_disease.pth
        ├── dinov2_labels.json
        ├── initial_references.pt
        ├── initial_similarity_config.json
        ├── disease_references.pt
        └── similarity_config.json
```

## Build Stage 1 references once

Your dataset should look like:

```text
Downloads/Training/Datasets/
├── Banana Leaf/
│   ├── Black_Sigatoka/
│   ├── Cordana/
│   ├── Freckle/
│   ├── Healthy/
│   └── Pestalotiopsis/
└── Not BananaLeaf/
```

From the repository root run:

```powershell
python Django\build_initial_references.py --dataset-root "C:\Users\jimmy\Downloads\Training\Datasets"
```

This does **not** retrain DINOv2. It creates embeddings using the existing
trained model and writes:

```text
trainedai/dinov2-base/initial_references.pt
trainedai/dinov2-base/initial_similarity_config.json
```

The builder samples each Banana Leaf disease/health folder separately so the
large Healthy dataset does not dominate Stage 1.

## Stage 2 disease references

If you already built `disease_references.pt`, keep it. Otherwise run:

```powershell
python Django\build_disease_references.py --dataset "C:\Users\jimmy\Downloads\Training\Datasets\Banana Leaf"
```

DINOv2 Stage 2 then checks both the five-class classifier and disease-reference
similarity before accepting a disease result.

## Run

```powershell
cd Django
python manage.py runserver
```

## Stage 1 tuning

`initial_similarity_config.json` contains:

- `banana_threshold` — minimum Banana Leaf visual similarity
- `binary_margin` — how far Banana Leaf must beat Not BananaLeaf
- `top_k` — number of nearest references used

Test with coconut/palm leaves and other hard negatives after building the bank.
