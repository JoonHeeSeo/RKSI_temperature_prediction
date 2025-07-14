import os
import csv

def write_metrics(
    model_name: str,
    mae: float,
    rmse: float,
    metrics_file: str = 'service/results.csv'
) -> None:
    """
    Add or update model performance metrics in a CSV file.

    If an entry for the given model_name exists, it will be replaced;
    otherwise, a new row is appended.

    Args:
        model_name: Identifier for the model (e.g., 'linear', 'tcn').
        mae: Mean absolute error.
        rmse: Root mean squared error.
        metrics_file: Path to the CSV file where metrics are stored.
    """
    # Ensure directory exists
    dirpath = os.path.dirname(metrics_file)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Define header
    header = ['model', 'MAE(℃)', 'RMSE(℃)']

    # Read existing rows, excluding header and same model_name
    rows = []
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', newline='') as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            for row in reader:
                if row and row[0] != model_name:
                    rows.append(row)

    # Prepare new row
    new_row = [model_name, f'{mae:.3f}', f'{rmse:.3f}']

    # Write back all
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)
        writer.writerow(new_row)