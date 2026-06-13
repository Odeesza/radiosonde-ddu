# download_data_mem.py

from datetime import date
from typing import List, Optional, Dict
import httpx
import io
import zipfile

from processing import parse_rs_bytes


# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------

BASE_RECENT_URL = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/RS_HR"

ARCHIVE_BASE_URL = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/RS/Archive"


# -------------------------------------------------
# RECENT MODE
# -------------------------------------------------

def get_recent_day(
    day: date,
    station: str = "89642",
    hours: List[int] = [0, 12],
) -> List[Dict]:

    if (date.today() - day).days > 15:
        raise ValueError("Recent data only available for last 15 days.")

    soundings_all = []

    with httpx.Client(timeout=30.0) as client:

        for hour in hours:

            if hour not in [0, 12]:
                raise ValueError("Hour must be 0 or 12.")

            filename = f"{station}.{day.strftime('%Y%m%d')}{hour:02d}.csv"
            url = f"{BASE_RECENT_URL}/{filename}"

            response = client.get(url)

            if response.status_code != 200:
                continue

            if "text/html" in response.headers.get("Content-Type", ""):
                continue

            soundings = parse_rs_bytes(response.content)
            soundings_all.extend(soundings)

    return soundings_all


# -------------------------------------------------
# ARCHIVE MODE
# -------------------------------------------------

def get_archive_month(
    year: int,
    month: int,
    station: str = "89642",
    day: Optional[List[int]] = None,
    hours: Optional[List[int]] = None,
) -> List[Dict]:

    filename = f"rs.{year}{month:02d}.zip"
    url = f"{ARCHIVE_BASE_URL}/{filename}"


    soundings_all = []
    print(f"Downloading {filename}...")
    with httpx.Client(timeout=60.0) as client:
        try:
            response = client.get(url, follow_redirects=True)
        except httpx.RequestError as e:
            print(f"Network error for {filename}: {e}")
            return []

        if response.status_code != 200:
            print(f"{filename} HTTP {response.status_code} → skipping")
            return []

        try:
            z = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile:
            print(f"{filename} is not a valid zip file → skipping")
            return []

        for f in z.namelist():

            if not f.endswith(".HR.csv"):
                continue
            # Filter by station
            if station not in f:
                continue

            # Filter by day if requested
            if day:
                if f"{year}{month:02d}{day:02d}" not in f:
                    continue

            # Filter by hour if requested
            if hours:
                if not any(f.split(".")[1].endswith(f"{hour:02d}") for hour in hours):
                    continue

            try:
                with z.open(f) as file_obj:
                    file_bytes = file_obj.read()
                    soundings = parse_rs_bytes(file_bytes)
                    soundings_all.extend(soundings)
            except Exception as e:
                print(f"Parsing error for {f}: {e}")
                continue

    print(f"Total soundings in {filename}: {len(soundings_all)}")
    return soundings_all


# -------------------------------------------------
# UNIFIED ENTRY POINT
# -------------------------------------------------

def get_data(
    years: Optional[List[int]] = None,
    months: Optional[List[int]] = None,
    days: Optional[List[int]] = None,
    dates: Optional[List[date]] = None,
    station: str = "89642",
    hours: List[int] = [0, 12],
) -> List[Dict]:

    if years and dates:
        raise ValueError("Choose either archive mode or recent mode.")

    results = []

    # ARCHIVE
    if years:
        print("Archive mode:", years, months, days)
        if months:
            for year in years:
                for m in months:
                    if days:
                        for day in days:
                            results.extend(
                                get_archive_month(
                                    year,
                                    m,
                                    day=day,
                                    station=station,
                                    hours=hours,
                                )
                            )
                    else:
                        results.extend(
                            get_archive_month(
                                year,
                                m,
                                station=station,
                                hours=hours,
                            )
                        )
        else:
            for year in years:
                for m in range(1, 13):
                    if days:
                        for day in days:
                            results.extend(
                                get_archive_month(
                                    year,
                                    m,
                                    day=day,
                                    station=station,
                                    hours=hours,
                                )
                            )
                    else:
                        results.extend(
                            get_archive_month(
                                year,
                                m,
                                station=station,
                                hours=hours,
                            )
                        )

    # RECENT
    if dates:
        for d in dates:
            results.extend(
                get_recent_day(
                    d,
                    station=station,
                    hours=hours,
                )
            )

    return results
