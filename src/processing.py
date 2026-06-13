# processing.py

import pandas as pd
from typing import List, Dict, Optional
import io


# -------------------------------------------------
# TYPE CONVERSION
# -------------------------------------------------

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking columns to numbers and parse date column."""
    df = df.copy()

    for col in df.columns:
        if col.lower() == "date":
            df[col] = pd.to_datetime(
                df[col],
                format="%Y%m%d%H%M%S",
                errors="coerce"
            )
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -------------------------------------------------
# CORE PARSER (works on text, not file)
# -------------------------------------------------

def parse_rs_content(content: str) -> List[Dict[str, Optional[pd.DataFrame]]]:
    """
    Parse a MeteoFrance HR_complet CSV file content.
    Returns a list of soundings.
    """

    lines = content.splitlines()
    soundings = []

    i = 0
    while i < len(lines):

        line = lines[i].strip()

        if line.startswith("numer_sta"):

            # ---- HEADER ----
            header_cols = line.split(",")
            header_vals = lines[i+1].strip().split(",")

            header_df = pd.DataFrame([header_vals], columns=header_cols)
            header_df = convert_types(header_df)

            i += 2

            # ---- DATA ----
            data_cols = lines[i].strip().split(",")
            i += 1

            data_rows = []
            while i < len(lines) and not lines[i].startswith(("p_cis", "numer_sta")):
                data_rows.append(lines[i].strip().split(","))
                i += 1

            data_df = pd.DataFrame(data_rows, columns=data_cols)
            data_df = convert_types(data_df)

            # ---- CIS (optional) ----
            cis_df = None

            if i < len(lines) and lines[i].startswith("p_cis"):

                cis_cols = lines[i].strip().split(",")
                i += 1

                cis_rows = []
                while i < len(lines) and not lines[i].startswith("numer_sta"):
                    cis_rows.append(lines[i].strip().split(","))
                    i += 1

                cis_df = pd.DataFrame(cis_rows, columns=cis_cols)
                cis_df = convert_types(cis_df)

            soundings.append({
                "header": header_df,
                "data": data_df,
                "cis": cis_df
            })

        else:
            i += 1

    return soundings


# -------------------------------------------------
# FILE WRAPPER
# -------------------------------------------------

def parse_rs_file(filepath: str):
    with open(filepath, "r") as f:
        content = f.read()
    return parse_rs_content(content)


# -------------------------------------------------
# MEMORY WRAPPER (for download_data integration)
# -------------------------------------------------

def parse_rs_bytes(file_bytes: bytes):
    content = file_bytes.decode("utf-8")
    return parse_rs_content(content)

