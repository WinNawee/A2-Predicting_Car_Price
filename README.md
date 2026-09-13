# A2: Predicting Car Price

AT82.03 Machine Learning  Assignment 2. Continues from Assignment 1's used-car
price dataset, replacing the modeling step with a linear regression class built
from scratch (gradient descent, Xavier initialization, optional momentum),
compares it across a cross-validated experiment grid tracked in MLflow, and
deploys a two-model web app.

**Live site**: [https://web-st127031.ml.brain.cs.ait.ac.th](https://web-st127031.ml.brain.cs.ait.ac.th)

## Repository contents

```
.
├── README.md
├── car_price_prediction_A2.ipynb   # Task 1 + Task 2, full notebook
├── result/                          # MLflow screenshots referenced by the notebook
│   ├── mlflow_params.png
│   └── mlflow_metrics.png
└── app/                             # Task 3 web application
    ├── Dockerfile
    ├── docker-compose.yaml
    └── code/
        ├── app.py                   # entry point, shared nav bar, page routing
        ├── requirements.txt
        ├── assets/
        │   └── style.css            # styling for the new-model page
        ├── pages/
        │   ├── home.py              # landing page, compares both models
        │   ├── old_model.py         # Assignment 1 model (scikit-learn pipeline)
        │   └── new_model.py         # Assignment 2 model (from-scratch class)
        └── model/
            ├── car_price_model.pkl      # original scikit-learn pipeline
            ├── car_price_model_v2.pkl   # theta + preprocessing + residual std
            └── model_metadata.pkl       # shared dropdown/owner-scale metadata
```

## Task 1 — From-scratch `LinearRegression`

Extends the base class from `03 - Regularization.ipynb` with:

- `r2()` — computes the R² score
- Selectable weight initialization: `init_method='zero'` (original) or
  `init_method='xavier'` (`U[-1/sqrt(m), 1/sqrt(m)]`)
- Optional momentum: `use_momentum=True/False`, `momentum` in `(0, 1)`
- `plot_feature_importance()` — bar chart of coefficients by magnitude

All additions default to the original behavior (`init_method='zero'`,
`use_momentum=False`), so the class is backward compatible.

## Task 2 — Experiment comparison

Cross-validated comparison across:

1. model type — normal, lasso, ridge, polynomial
2. momentum — with / without
3. batch method — stochastic, mini-batch, batch
4. weight init — zero, Xavier
5. learning rate — 0.01, 0.001, 0.0001

144 base configurations, plus 36 re-run polynomial configurations after fixing
a feature-scaling issue — 180 runs total, every one logged to MLflow (params +
per-epoch train/val loss). Each candidate is ranked purely on its internal
3-fold CV validation score; the test set is touched exactly once, on the
already-chosen winner, to avoid leaking it into model selection.

**Best configuration**: unregularized linear regression, stochastic gradient
descent, Xavier init, `lr=0.01`, no momentum.

| | CV | Test |
|---|---|---|
| R² | 0.856 | 0.875 |
| MSE | 0.084 | 0.070 |

Full writeup, the comparison table, and MLflow screenshots are in
`car_price_prediction_A2.ipynb` under **Key Findings**.

## Task 3 — Web application

Three pages, sharing one Dash Pages app (`app/code/app.py`):

- **`/`** — home page, an honest side-by-side comparison of both models
  (the original scikit-learn pipeline scores higher on R²; the new model's
  value is transparency — every coefficient is inspectable, and it was tuned
  through the fully-tracked search described above)
- **`/old`** — the original Assignment 1 model, unchanged
- **`/new`** — the Assignment 2 model. Predicts a price and a ~90% typical
  range (derived from the test-set residual standard deviation), using only
  the exported `theta` array and `prep` transformer — no custom class needs
  to be importable at load time

### Running locally

```bash
cd app/code
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8050`.

### Running with Docker

```bash
cd app
docker build -t car-price-app .
docker run -p 8050:8050 car-price-app
```

### Deployment

Deployed to the course's `ml-brain` server behind Traefik at
`web-st127031.ml.brain.cs.ait.ac.th`. The image is built locally, pushed to
Docker Hub (`naweep/car-price-app:latest`), and pulled by `docker-compose.yaml`
on the server, which joins the existing `web` Docker network and adds Traefik
routing labels (`entrypoints=websecure`, `certresolver=letsencrypt`) — no ports
are published directly; Traefik reaches the container over that network.