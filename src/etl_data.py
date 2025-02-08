"""
Converts the raw CSV form to a Parquet form and drops the 'Time' column.
"""

import os
import tempfile
import click
import pandas as pd
import mlflow


@click.command(
    help="Given a CSV file (see load_raw_data), transforms it into Parquet "
    "in an mlflow artifact called 'fraud-parquet-dir'"
)
@click.option("--fraud-csv")
def etl_data(fraud_csv):
    with mlflow.start_run():
        tmpdir = tempfile.mkdtemp()
        fraud_parquet_fp = os.path.join(tmpdir, "creditcard.parquet")
        print(
            f"Converting fraud csv in {fraud_csv} to parquet {fraud_parquet_fp}")
        fraUd_csv_fp = os.path.join(fraud_csv, "creditcard.csv")
        df = pd.read_csv(fraUd_csv_fp)
        df.drop(columns=["Time"], inplace=True)
        df.to_parquet(fraud_parquet_fp, index=False)
        mlflow.log_artifact(fraud_parquet_fp, "creditcard-parquet-dir")


if __name__ == "__main__":
    etl_data()
