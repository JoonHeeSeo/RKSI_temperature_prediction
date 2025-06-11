from meteostat import Point, Daily
from datetime import datetime
import pandas as pd
import argparse

# pip install meteostat pandas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', type=str, default='RKSI')
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    # Coordinates for Incheon Airport (RKSI)
    rksi = Point(37.4602, 126.4407, 23)
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')

    data = Daily(rksi, start, end).fetch()
    data.to_csv(args.out)
    print(f"✅ Saved to {args.out}")
