# BigData_Group1
Supervised learning pipeline for the Spark course

This project builds an end-to-end supervised learning pipeline with PySpark MLlib for flight data.

The project covers two tasks:

- Regression: predict the number of minutes a flight arrives late.
- Binary classification: predict whether a flight is severely delayed.

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── custom_transformers.py
├── mllib_pipeline.py
├── inference.py
├── benchmark_results.py
├── submit_pipeline.sh
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── docs/
│   └── REPORT.md
└── models/
    └── README.md
```

## Set Up the Local Environment

### Requirements

- Eclipse Temurin JDK 17.
- Apache Spark 3.5.9.
- Conda and Python 3.11.

Set `JAVA_HOME` to the JDK 17 directory on your machine and add the JDK `bin` directory to `PATH`. Do not copy an installation path from another machine.

Example configuration for the current PowerShell session:

```powershell
$env:JAVA_HOME = '<path-to-jdk-17>'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

Create and activate the Conda environment:

```powershell
conda create -n bigdata-spark python=3.11 pip -y
conda activate bigdata-spark
```

Install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
java -version
javac -version
python --version
python -c "import pyspark; print('pyspark', pyspark.__version__)"
python -c "import pyarrow; print('pyarrow', pyarrow.__version__)"
python -c "import pandas; print('pandas', pandas.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import mlflow; print('mlflow', mlflow.__version__)"
```

## Prepare Local Data

Each team member should place raw CSV files in `data/raw` and create typed Parquet at `data/typed-parquet`. See the WSL/Bash and PowerShell commands in the [data preparation guide](data/README.md); WSL/Linux is recommended for writing Parquet.

The current Python files are project scaffolding; data processing, pipeline, modeling, and streaming functionality has not yet been implemented.
