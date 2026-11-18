# Large datasets and common tabular formats
data/**
!data/README.md
*.csv
*.csv.gz
*.parquet
*.parquet/**

# Model artifacts
models/**
!models/README.md
*.model
*.pkl
*.pickle
*.joblib

# MLflow
mlruns/

# Credentials and local environments
.env
.env.*
.venv/
venv/
env/

# Python cache and tooling
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Jupyter
.ipynb_checkpoints/

# Spark runtime artifacts
spark-warehouse/
metastore_db/
checkpoints/
checkpoint/

# Logs
*.log
logs/

# Generated datasets
data/raw/
data/typed-parquet/
data/model-ready/

# Python
__pycache__/
*.pyc

# Model artifacts
*.pkl
*.joblib
*.pt

# Spark
spark-warehouse/
metastore_db/