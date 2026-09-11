"""Shared reproducible pipelines for Kampala PM2.5 and respiratory-health work.

The module intentionally distinguishes DHIS2 surveillance time-series data from
facility medical-record-review data.  The latter measure *recorded cases*, not
community incidence, and its spatial rate analyses must be interpreted as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_DIR = Path("data")
KAMPALA_DIVISIONS = ["Kampala Central", "Kawempe", "Makindye", "Nakawa", "Rubaga"]
OUTCOME_COLUMNS = ["ILI", "ILD_deaths", "pneumonia_u5", "SARI", "SARI_deaths", "TB"]
MEDICAL_FILE_GLOB = "Exploring_the_Association*.xlsx"
MONITOR_FILES = [
    "bq-results-20260322-100500-1774173924222.csv",
    "bq-results-20260322-100723-1774174052683.csv",
]

# Only add aliases after they have been reviewed against an administrative source.
# Validated administrative parish aliases based on the Uganda Parish Shapefile.
REVIEWED_PARISH_ALIASES: dict[str, str] = {
    "nateete": "natete", "nakulabye": "nankulabye", "bugoloobi": "bugolobi",
    "kikaya": "kikaaya", "kazoangola": "kazo", "nakawainstitutions": "nakawainstitution",
    "najjanankumbii": "najjanankubii", "najjanankumbiii": "najjanankubiii",
    "kansanga": "kansangamuyenga", "muyenga": "kansangamuyenga",
    "civiccentre": "civiccenter", "nakivubo": "nakivuboshauriyako",
    "nsambyacentral": "nsambyaestate", "nsambyarailways": "nsambyarailway",
    "mbuya": "mbuyai", "naguru": "nagurui", "kisenyi": "kisenyii",
    "kamwokya": "kamwokyai", "kawempe": "kawempei", "bwaise": "bwaisei",
    "makerere": "makererei", "katwe": "katwei", "makindye": "makindyei",
    "nsambya": "nsambyaestate", "bukoto": "bukotoi", "mulago": "mulagoi",
    "kololo": "kololoi", "kawaala": "kasubi", "kitebi": "kabowa",
    "wankulukuku": "kabowa", "namungoona": "lubya", "namuwongo": "kisugu",
    "kibuye": "kibuyei", "mengo": "mengo",
}


def normalise_text(value: object) -> str | None:
    """Return a conservative matching key; never use this for fuzzy matching."""
    if pd.isna(value):
        return None
    value = re.sub(r"[^a-z0-9]", "", str(value).lower().strip())
    return value or None


def standardise_division(value: object) -> str | None:
    key = normalise_text(value)
    if key is None:
        return None
    mapping = {
        "kampalacentral": "Kampala Central", "kampalacentraldivision": "Kampala Central", "central": "Kampala Central",
        "centraldivision": "Kampala Central", "kawempe": "Kawempe",
        "kawempedivision": "Kawempe", "makindye": "Makindye",
        "makindyedivision": "Makindye", "nakawa": "Nakawa",
        "nakawadivision": "Nakawa", "rubaga": "Rubaga", "rubagadivision": "Rubaga",
        "lubaga": "Rubaga", "lubagadivision": "Rubaga",
    }
    return mapping.get(key)


def epi_week_end(year: pd.Series, week: pd.Series) -> pd.Series:
    return pd.to_datetime(year.astype(int).astype(str) + "-W" + week.astype(int).astype(str).str.zfill(2) + "-7", format="%G-W%V-%u", errors="coerce")


def load_weekly_panel(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Create the canonical weekly DHIS2/environment panel without outcome imputation."""
    climate = pd.read_csv(data_dir / "epi_weekly_climate data_kampala.csv")
    pm = pd.read_csv(data_dir / "kampala_weekly_PM2.5 climate data averaged_across_sites.csv")
    dhis = pd.read_excel(data_dir / "Data RTI DHIS2.xls", sheet_name="MOH - Uganda, Kampala", header=2)
    dhis = dhis.rename(columns={
        "ILI Cases": "ILI", "ILD deaths": "ILD_deaths",
        "Severe pneumonia under 5 Cases": "pneumonia_u5",
        "Severe pneumonia under 5 deaths": "pneumonia_u5_deaths",
        "SARI Cases": "SARI", "SARI Deaths": "SARI_deaths", "Pulmonary TB cases": "TB",
    })
    dhis["date"] = pd.to_datetime(dhis["periodname"].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})$")[0], format="%Y-%m-%d", errors="coerce")
    dhis = dhis.dropna(subset=["date"]).drop_duplicates("date")
    env = climate.merge(pm, left_on=["epi_year", "epi_week"], right_on=["year", "week"], how="outer", validate="one_to_one")
    env["date"] = epi_week_end(env["epi_year"].fillna(env["year"]), env["epi_week"].fillna(env["week"]))
    env = env.rename(columns={
        "weekly_pm25_avg": "pm2_5", "weekly_temp_avg": "avgtemp",
        "weekly_humidity_avg": "avghumidity", "Avg_Temp(C)": "avgtemp_sat",
        "AvgWindspeed(m/s)": "windspeed", "AvgPrecipitation(mm/day)": "precip",
        "AvgRelative_Humidity(%)": "humidity_sat",
    })
    # Some source exports repeat an epidemiological week.  Retain one explicit
    # environmental observation per week before joining to DHIS2 notifications.
    env = env.dropna(subset=["date"]).groupby("date", as_index=False).agg({
        column: "mean" if pd.api.types.is_numeric_dtype(env[column]) else "first"
        for column in env.columns if column != "date"
    })
    cols = ["date", "epi_year", "epi_week", "pm2_5", "avgtemp", "avghumidity", "readings_per_week", "avgtemp_sat", "humidity_sat", "windspeed", "precip"]
    panel = env[[c for c in cols if c in env]].merge(dhis[["date", *[c for c in OUTCOME_COLUMNS + ["pneumonia_u5_deaths"] if c in dhis]]], on="date", how="left", validate="one_to_one")
    panel = panel.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    panel["month"] = panel.date.dt.month
    panel["time_index"] = np.arange(len(panel))
    panel["monitoring_complete"] = panel["readings_per_week"].ge(panel["readings_per_week"].median()).astype("Int64")
    return panel


def aggregate_time_panel(weekly: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate canonical data; outcomes use min_count=1 so missingness survives."""
    frame = weekly.copy().set_index("date")
    outcome = [c for c in OUTCOME_COLUMNS if c in frame]
    environmental = [c for c in ["pm2_5", "avgtemp", "avghumidity", "avgtemp_sat", "humidity_sat", "windspeed", "precip", "readings_per_week"] if c in frame]
    result = pd.concat([
        frame[outcome].resample(frequency).sum(min_count=1),
        frame[environmental].resample(frequency).mean(),
    ], axis=1).reset_index().dropna(subset=["date"])
    result["year"] = result.date.dt.year
    result["month"] = result.date.dt.month
    result["time_index"] = np.arange(len(result))
    return result


def add_pm_lags(frame: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    result = frame.sort_values("date").copy()
    for lag in range(max_lag + 1):
        result[f"pm25_lag{lag}"] = result["pm2_5"].shift(lag)
    return result


def fit_time_series_models(frame: pd.DataFrame, outcomes: Iterable[str] = OUTCOME_COLUMNS, max_lag: int = 4) -> pd.DataFrame:
    """Negative-binomial models with calendar/month, trend, climate and PM lags."""
    data = add_pm_lags(frame, max_lag)
    rows = []
    covariates = [c for c in ["avgtemp", "avghumidity", "precip", "time_index", "month", *[f"pm25_lag{i}" for i in range(max_lag + 1)]] if c in data]
    for outcome in outcomes:
        if outcome not in data:
            continue
        model_data = data[[outcome, *covariates]].dropna()
        if len(model_data) < max(30, len(covariates) + 8) or model_data[outcome].sum() == 0:
            rows.append({"outcome": outcome, "status": "insufficient complete observations", "n": len(model_data)})
            continue
        formula = f"{outcome} ~ " + " + ".join(["C(month)" if c == "month" else c for c in covariates])
        try:
            model = smf.glm(formula, data=model_data, family=sm.families.NegativeBinomial()).fit(cov_type="HC0")
            for term in [c for c in model.params.index if c.startswith("pm25_lag")]:
                rows.append({"outcome": outcome, "term": term, "irr_per_10ug_m3": float(np.exp(model.params[term] * 10)), "ci_low": float(np.exp((model.params[term] - 1.96 * model.bse[term]) * 10)), "ci_high": float(np.exp((model.params[term] + 1.96 * model.bse[term]) * 10)), "n": int(model.nobs), "status": "ok"})
        except Exception as exc:  # a notebook should expose, not hide, model failure
            rows.append({"outcome": outcome, "status": f"model failed: {type(exc).__name__}", "n": len(model_data)})
    return pd.DataFrame(rows)


def load_population(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw = pd.read_csv(data_dir / "Kampala Division Population.csv", header=1)
    raw = raw.rename(columns={"Location": "division_raw", "Parishes": "parish_raw", "Male": "population_male", "Female": "population_female", "Total": "population_total", "Age 0-4": "population_age_0_4", "Age 6-12": "population_age_6_12", "Age 13-18": "population_age_13_18", "Age 14-64": "population_age_14_64", "Age 65+": "population_age_65_plus"})
    numeric = [c for c in raw if c.startswith("population_")]
    for col in numeric:
        raw[col] = pd.to_numeric(raw[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    raw["division"] = raw.division_raw.map(standardise_division)
    raw["parish_key"] = raw.parish_raw.map(normalise_text)
    if raw.duplicated(["division", "parish_key"]).any():
        raise ValueError("Population file has duplicate division/parish keys")
    return raw[["division", "parish_raw", "parish_key", *numeric]].dropna(subset=["division"])


def population_denominator_summary(population: pd.DataFrame) -> pd.DataFrame:
    """Return audited division denominators for rate interpretation and offsets."""
    population_columns = [column for column in population if column.startswith("population_")]
    summary = population.groupby("division", as_index=False).agg(
        parishes=("parish_key", "nunique"),
        **{column: (column, "sum") for column in population_columns},
    )
    if (summary["population_total"] <= 0).any():
        raise ValueError("Population denominators must be positive for every Kampala division")
    if not np.allclose(
        summary["population_male"] + summary["population_female"],
        summary["population_total"],
        equal_nan=False,
    ):
        raise ValueError("Male and female population denominators do not sum to the division total")
    return summary.sort_values("division").reset_index(drop=True)


def age_band(age: object) -> str | None:
    age = pd.to_numeric(age, errors="coerce")
    if pd.isna(age) or age < 0 or age > 120:
        return None
    if age <= 4: return "0-4"
    if 6 <= age <= 12: return "6-12"
    if 13 <= age <= 18: return "13-18"
    if 19 <= age <= 64: return "19-64"
    if age >= 65: return "65+"
    return "unmapped_5"


def load_medical_records(data_dir: Path = DATA_DIR, start="2020-01-01", end="2024-12-31") -> pd.DataFrame:
    matches = list(data_dir.glob(MEDICAL_FILE_GLOB))
    if len(matches) != 1:
        raise FileNotFoundError("Expected exactly one medical-review workbook")
    raw = pd.read_excel(matches[0]).copy()
    rename = {"4. Date of hospital visit:": "visit_date", "3. Patient Diagnosis": "diagnosis", "7. Gender": "sex", "8. Age:": "age", "10. Division (Kampala)": "division_raw", "12. Parish name:": "parish_raw", "1. Name of Health facility:": "facility"}
    df = raw.rename(columns=rename)
    df["visit_date"] = pd.to_datetime(df.visit_date, errors="coerce")
    df["division"] = df.division_raw.map(standardise_division)
    df["parish_key"] = df.parish_raw.map(normalise_text).replace(REVIEWED_PARISH_ALIASES)
    df["sex"] = df.sex.astype(str).str.strip().str.title().where(lambda x: x.isin(["Male", "Female"]))
    df["age_band"] = df.age.map(age_band)
    df["diagnosis"] = df.diagnosis.fillna("Unknown").astype(str).str.strip()
    df["facility"] = df.facility.fillna("Unknown").astype(str).str.strip()
    df["in_linked_window"] = df.visit_date.between(pd.Timestamp(start), pd.Timestamp(end))
    df["valid_kampala_division"] = df.division.isin(KAMPALA_DIVISIONS)
    return df


def load_monitor_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for name in MONITOR_FILES:
        data = pd.read_csv(data_dir / name)
        data = data.rename(columns={"site_longtude": "site_longitude", "pm2_5_calibrated_value": "pm2_5"})
        frames.append(data)
    monitor = pd.concat(frames, ignore_index=True).drop_duplicates(["timestamp", "site_id"])
    monitor["timestamp"] = pd.to_datetime(monitor.timestamp, errors="coerce", utc=True)
    monitor["date"] = monitor.timestamp.dt.tz_convert(None).dt.normalize()
    monitor["division"] = monitor.site_name.str.split(",").str[-1].str.strip().map(standardise_division)
    monitor.loc[monitor.parish.astype(str).str.contains("makindye division", case=False, na=False), "division"] = "Makindye"
    monitor["pm2_5"] = pd.to_numeric(monitor.pm2_5, errors="coerce")
    if "temperature" in monitor.columns:
        monitor["temperature"] = pd.to_numeric(monitor["temperature"], errors="coerce")
    if "humidity" in monitor.columns:
        monitor["humidity"] = pd.to_numeric(monitor["humidity"], errors="coerce")
    return monitor.dropna(subset=["date", "pm2_5", "division"])


def daily_division_exposure(monitor: pd.DataFrame) -> pd.DataFrame:
    result = monitor[monitor.division.isin(KAMPALA_DIVISIONS)].groupby(["division", "date"], as_index=False).agg(pm2_5=("pm2_5", "mean"), monitor_sites=("site_id", "nunique"), monitor_records=("pm2_5", "size"))
    result["monitoring_complete"] = result.monitor_sites.ge(1)
    return result


def build_spatial_panel(data_dir: Path = DATA_DIR, frequency="W-SUN") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return rate panel, record-level data and linkage/coverage quality table."""
    population = load_population(data_dir)
    medical = load_medical_records(data_dir)
    monitor = load_monitor_data(data_dir)
    medical = medical[medical.in_linked_window & medical.valid_kampala_division].copy()
    medical = medical.merge(population[["division", "parish_key"]], on=["division", "parish_key"], how="left", indicator="parish_match")
    medical["parish_matched"] = medical.parish_match.eq("both")
    exposure = daily_division_exposure(monitor)
    medical = medical.merge(exposure, on=["division", "date"], how="left") if "date" in medical else medical
    # visit_date is the explicitly named temporal key in the record-review file
    medical = medical.drop(columns=[c for c in ["date"] if c in medical]).merge(exposure, left_on=["division", "visit_date"], right_on=["division", "date"], how="left")
    medical["period"] = medical.visit_date.dt.to_period(frequency.replace("W-SUN", "W")) .dt.to_timestamp("W") if frequency.startswith("W") else medical.visit_date.dt.to_period(frequency).dt.to_timestamp()
    division_pop = population.groupby("division", as_index=False).sum(numeric_only=True)
    counts = medical.dropna(subset=["period", "pm2_5", "sex", "age_band"]).query("age_band != 'unmapped_5'").groupby(["division", "period", "sex", "age_band", "diagnosis"], as_index=False).agg(recorded_cases=("diagnosis", "size"), pm2_5=("pm2_5", "mean"), monitor_sites=("monitor_sites", "mean"), facilities=("facility", "nunique"))
    age_col = {"0-4": "population_age_0_4", "6-12": "population_age_6_12", "13-18": "population_age_13_18", "19-64": "population_age_14_64", "65+": "population_age_65_plus"}
    pop_lookup = division_pop.set_index("division")
    def allocated_denominator(row):
        p = pop_lookup.loc[row.division]
        sex_pop = p["population_male"] if row.sex == "Male" else p["population_female"]
        return sex_pop * p[age_col[row.age_band]] / p["population_total"]
    counts["population_offset"] = counts.apply(allocated_denominator, axis=1)
    counts["recorded_rate_per_100k"] = counts.recorded_cases / counts.population_offset * 100000
    # Following the reference paper's group-mean-centering principle, separate
    # day-to-day PM2.5 variation within a division from between-division contrast.
    counts["pm2_5_division_mean"] = counts.groupby("division")["pm2_5"].transform("mean")
    counts["pm2_5_within_division"] = counts["pm2_5"] - counts["pm2_5_division_mean"]
    
    # Aggregate across diagnoses to create a 'Total Respiratory Diseases' stratum
    total_counts = counts.groupby(["division", "period", "sex", "age_band"], as_index=False).agg(
        recorded_cases=("recorded_cases", "sum"),
        pm2_5=("pm2_5", "first"),
        monitor_sites=("monitor_sites", "first"),
        facilities=("facilities", "sum"),
        population_offset=("population_offset", "first"),
        pm2_5_division_mean=("pm2_5_division_mean", "first"),
        pm2_5_within_division=("pm2_5_within_division", "first"),
    )
    total_counts["diagnosis"] = "Total Respiratory Diseases"
    total_counts["recorded_rate_per_100k"] = total_counts.recorded_cases / total_counts.population_offset * 100000
    counts = pd.concat([counts, total_counts], ignore_index=True)
    
    quality = pd.DataFrame({
        "metric": ["medical_records_linked_window", "parish_exact_or_reviewed_match", "records_with_same_day_division_pm25", "monitor_days", "monitor_sites"],
        "value": [len(medical), int(medical.parish_matched.sum()), int(medical.pm2_5.notna().sum()), monitor.date.nunique(), monitor.site_id.nunique()],
    })
    return counts, medical, quality


def fit_spatial_rate_models(panel: pd.DataFrame) -> pd.DataFrame:
    """Primary negative-binomial models for facility-recorded RTI rates."""
    rows = []
    for diagnosis, data in panel.groupby("diagnosis"):
        data = data.dropna(subset=["population_offset", "pm2_5"]).copy()
        data["time_index"] = (data.period - data.period.min()).dt.days
        data["month"] = data.period.dt.month
        if len(data) < 40 or data.recorded_cases.sum() < 20:
            rows.append({"diagnosis": diagnosis, "status": "insufficient data", "n": len(data)})
            continue
        try:
            fit = smf.glm("recorded_cases ~ pm2_5_within_division + pm2_5_division_mean + C(sex) + C(age_band) + C(month) + time_index", data=data, offset=np.log(data.population_offset), family=sm.families.NegativeBinomial()).fit(cov_type="HC0")
            beta, se = fit.params["pm2_5_within_division"], fit.bse["pm2_5_within_division"]
            rows.append({"diagnosis": diagnosis, "status": "ok", "n": int(fit.nobs), "within_division_irr_per_10ug_m3": np.exp(beta * 10), "ci_low": np.exp((beta - 1.96 * se) * 10), "ci_high": np.exp((beta + 1.96 * se) * 10), "pearson_overdispersion": fit.pearson_chi2 / fit.df_resid})
        except Exception as exc:
            rows.append({"diagnosis": diagnosis, "status": f"model failed: {type(exc).__name__}", "n": len(data)})
    return pd.DataFrame(rows)


def fit_case_mix_model(records: pd.DataFrame) -> pd.DataFrame:
    """Secondary diagnosis-vs-other-case models; never interpreted as RTI risk."""
    data = records.dropna(subset=["pm2_5", "sex", "age_band", "diagnosis", "facility"]).query("age_band != 'unmapped_5'").copy()
    if len(data) < 100 or data.diagnosis.nunique() < 2:
        return pd.DataFrame([{"status": "insufficient data", "n": len(data)}])
    data["month"] = data.visit_date.dt.month
    rows = []
    for diagnosis in sorted(data.diagnosis.unique()):
        model_data = data.copy()
        model_data["case"] = model_data.diagnosis.eq(diagnosis).astype(int)
        if model_data["case"].sum() < 25:
            rows.append({"diagnosis": diagnosis, "status": "insufficient cases", "n": len(model_data)})
            continue
        try:
            model = smf.glm("case ~ pm2_5 + C(sex) + C(age_band) + C(month) + C(facility)", data=model_data, family=sm.families.Binomial()).fit(cov_type="HC0")
            beta, se = model.params["pm2_5"], model.bse["pm2_5"]
            rows.append({"diagnosis": diagnosis, "odds_ratio_per_10ug_m3": np.exp(beta * 10), "ci_low": np.exp((beta - 1.96 * se) * 10), "ci_high": np.exp((beta + 1.96 * se) * 10), "n": int(model.nobs), "status": "ok"})
        except Exception as exc:
            rows.append({"diagnosis": diagnosis, "status": f"model failed: {type(exc).__name__}", "n": len(model_data)})
    return pd.DataFrame(rows)


def division_geometry(data_dir: Path = DATA_DIR) -> gpd.GeoDataFrame:
    shape = gpd.read_file(data_dir / "Kampala Shapefile" / "KMA Subcounties.shp")
    shape["division"] = shape.sname2019.map(standardise_division)
    return shape[shape.division.isin(KAMPALA_DIVISIONS)].drop_duplicates("division")[["division", "geometry"]]


def parish_geometry(data_dir: Path = DATA_DIR) -> gpd.GeoDataFrame:
    """Load and prepare Kampala parish boundary polygons from the cleaned shapefile.

    Filters to Kampala district (DName2016 == 'KAMPALA'), extracts official
    division mappings from the pc2016 code, standardises parish names, and
    reprojects to EPSG:4326 (WGS84).
    """
    shp_path = data_dir / "uganda_parishes_cleaned_attached" / "uganda_parishes_cleaned_attached.shp"
    if not shp_path.exists():
        shp_path = Path("data/uganda_parishes_cleaned_attached/uganda_parishes_cleaned_attached.shp")
    shp = gpd.read_file(shp_path)
    kampala = shp[shp["DName2016"].astype(str).str.upper().str.strip() == "KAMPALA"].to_crs("EPSG:4326").copy()
    div_map = {
        "102101": "Kampala Central",
        "102102": "Kawempe",
        "102103": "Rubaga",
        "102104": "Makindye",
        "102105": "Nakawa",
    }
    kampala["division"] = kampala["pc2016"].astype(str).str[:6].map(div_map)
    kampala["parish_name"] = kampala["p"].astype(str).str.strip().str.title()
    kampala["parish_key"] = kampala["p"].apply(normalise_text)
    return kampala.drop_duplicates(["division", "parish_key"])[["division", "parish_name", "parish_key", "pc2016", "geometry"]].reset_index(drop=True)


def build_parish_spatial_summary(
    medical_records: pd.DataFrame | None = None,
    monitor: pd.DataFrame | None = None,
    population: pd.DataFrame | None = None,
    data_dir: Path = DATA_DIR,
) -> gpd.GeoDataFrame:
    """Construct an integrated parish-level GeoDataFrame combining boundaries,
    IDW-interpolated PM2.5, UBOS census populations, and clinical case burdens."""
    parishes = parish_geometry(data_dir)
    
    # 1. PM2.5 IDW interpolation
    if monitor is None:
        monitor = load_monitor_data(data_dir)
    kp_utm = parishes.to_crs("EPSG:32636")
    sites = monitor.dropna(subset=["site_latitude", "site_longitude", "pm2_5"]).drop_duplicates(["site_id"]).copy()
    sites_geo = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites.site_longitude, sites.site_latitude),
        crs="EPSG:4326"
    ).to_crs("EPSG:32636")
    site_means = monitor.groupby("site_id")["pm2_5"].mean().reset_index()
    sites_geo = sites_geo.merge(site_means, on="site_id", suffixes=("", "_overall_mean"))
    
    xy_sites = np.c_[sites_geo.geometry.x, sites_geo.geometry.y]
    values = sites_geo["pm2_5_overall_mean"].to_numpy()
    parish_centers = kp_utm.geometry.representative_point()
    
    pm25_vals = []
    for p in parish_centers:
        dist = np.sqrt((xy_sites[:, 0] - p.x) ** 2 + (xy_sites[:, 1] - p.y) ** 2)
        weights = 1 / (np.maximum(dist, 100.0) ** 2)
        pm25_vals.append(float(np.average(values, weights=weights)))
    parishes["mean_pm25"] = pm25_vals
    
    # 2. Population linkage
    if population is None:
        population = load_population(data_dir)
    pop_aliases = {
        ("Kampala Central", "civiccentre"): ("Kampala Central", "civiccenter"),
        ("Kampala Central", "nakivubo"): ("Kampala Central", "nakivuboshauriyako"),
        ("Kawempe", "kazoangola"): ("Kawempe", "kazo"),
        ("Kawempe", "kikaya"): ("Kawempe", "kikaaya"),
        ("Kawempe", "mukmulukai"): ("Kawempe", "makerereuniversity"),
        ("Kawempe", "mukmulukaii"): ("Kawempe", "makerereuniversity"),
        ("Kawempe", "mukmulukaiii"): ("Kawempe", "makerereuniversity"),
        ("Kawempe", "mukmulukaiv"): ("Kawempe", "makerereuniversity"),
        ("Makindye", "kansanga"): ("Makindye", "kansangamuyenga"),
        ("Makindye", "muyenga"): ("Makindye", "kansangamuyenga"),
        ("Makindye", "nsambyacentral"): ("Makindye", "nsambyaestate"),
        ("Makindye", "nsambyarailways"): ("Makindye", "nsambyarailway"),
        ("Nakawa", "bugoloobi"): ("Nakawa", "bugolobi"),
        ("Nakawa", "nakawainstitutions"): ("Nakawa", "nakawainstitution"),
        ("Rubaga", "najjanankumbii"): ("Rubaga", "najjanankubii"),
        ("Rubaga", "najjanankumbiii"): ("Rubaga", "najjanankubiii"),
        ("Rubaga", "nakulabye"): ("Rubaga", "nankulabye"),
        ("Rubaga", "nateete"): ("Rubaga", "natete"),
    }
    pop_m = population.copy()
    for i, row in pop_m.iterrows():
        k = (row["division"], row["parish_key"])
        if k in pop_aliases:
            pop_m.at[i, "division"] = pop_aliases[k][0]
            pop_m.at[i, "parish_key"] = pop_aliases[k][1]
    pop_agg = pop_m.groupby(["division", "parish_key"], as_index=False).agg(
        population_total=("population_total", "sum"),
        population_male=("population_male", "sum"),
        population_female=("population_female", "sum"),
    )
    parishes = parishes.merge(pop_agg, on=["division", "parish_key"], how="left")
    
    # 3. Medical record linkage
    if medical_records is None:
        medical_records = load_medical_records(data_dir)
    med_valid = medical_records[medical_records["valid_kampala_division"]].copy()
    
    parish_diag = med_valid.groupby(["division", "parish_key", "diagnosis"]).size().unstack(fill_value=0)
    parish_diag.columns = [f"disease__{col}" for col in parish_diag.columns]
    disease_cols = [c for c in parish_diag.columns if c.startswith("disease__")]
    parish_diag["total_respiratory_cases"] = parish_diag[disease_cols].sum(axis=1)
    
    parishes = parishes.merge(parish_diag.reset_index(), on=["division", "parish_key"], how="left")
    for c in ["total_respiratory_cases", *disease_cols]:
        parishes[c] = parishes[c].fillna(0)
        
    parishes["annualized_rate_per_100k"] = np.where(
        parishes["population_total"] > 0,
        (parishes["total_respiratory_cases"] / 5.0) / parishes["population_total"] * 100000,
        0.0
    )
    return parishes


def plot_parish_map(
    parish_data: gpd.GeoDataFrame | pd.DataFrame,
    column: str,
    title: str,
    legend_label: str | None = None,
    cmap: str = "OrRd",
    data_dir: Path = DATA_DIR,
    figsize: tuple[int, int] = (12, 10),
    annotate_divisions: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Axes:
    """Render a high-resolution parish-level choropleth map of Kampala with division boundary overlays.

    Draws all 95 parish polygons colored by `column` with fine boundary outlines,
    overlaid with prominent bold division outlines and division labels for geographic orientation.
    """
    if not isinstance(parish_data, gpd.GeoDataFrame) or "geometry" not in parish_data.columns:
        p_geo = parish_geometry(data_dir)
        parish_data = p_geo.merge(parish_data, on=["division", "parish_key"] if "parish_key" in parish_data.columns else "division", how="left")
        
    div_gdf = division_geometry(data_dir)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    
    parish_data.plot(
        column=column,
        cmap=cmap,
        linewidth=0.5,
        edgecolor="#777777",
        legend=True,
        ax=ax,
        legend_kwds={"label": legend_label if legend_label else column, "shrink": 0.75},
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    
    # Overlay bold division boundaries
    div_gdf.boundary.plot(ax=ax, color="#111111", linewidth=2.0, linestyle="-")
    
    if annotate_divisions:
        for point, division in zip(div_gdf.geometry.representative_point(), div_gdf["division"]):
            ax.annotate(
                division.upper(),
                (point.x, point.y),
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="#111111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333333", lw=1.2, alpha=0.85)
            )
            
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_axis_off()
    plt.tight_layout()
    
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved parish map to {p}")
        
    if show:
        plt.show()
        
    return ax


def plot_division_map(summary: pd.DataFrame, column: str, title: str, data_dir: Path = DATA_DIR,
                      legend_label: str | None = None, annotate: bool = True):
    """Choropleth map of Kampala with parish boundaries included.
    
    If `summary` contains parish-level observations (e.g. parish_key or >10 rows),
    it renders at the parish level with division overlays. If `summary` contains division-level
    data, it colors by division while displaying the interior parish boundary outlines.
    """
    if "parish_key" in summary.columns or len(summary) > 10:
        return plot_parish_map(summary, column=column, title=title, legend_label=legend_label, data_dir=data_dir)
        
    div_geo = division_geometry(data_dir).merge(summary[["division", column]], on="division", how="left")
    parish_geo = parish_geometry(data_dir)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Division choropleth
    div_geo.plot(column=column, cmap="OrRd", edgecolor="#111111", linewidth=1.8, legend=True,
                 legend_kwds={"label": legend_label} if legend_label else None,
                 missing_kwds={"color": "lightgrey", "label": "No data"}, ax=ax)
                 
    # Overlay interior parish boundary outlines
    parish_geo.boundary.plot(ax=ax, color="#555555", linewidth=0.5, linestyle=":", alpha=0.7)
    
    if annotate:
        for row in div_geo.dropna(subset=[column]).itertuples():
            point = row.geometry.representative_point()
            ax.annotate(f"{row.division}\n{getattr(row, column):,.1f}", (point.x, point.y),
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#333333", "alpha": 0.85})
    ax.set_title(title, fontsize=14, fontweight="bold"); ax.set_axis_off()
    plt.tight_layout()
    return ax


def spatial_summary(panel: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    summary = panel.groupby("division", as_index=False).agg(mean_pm25=("pm2_5", "mean"), recorded_cases=("recorded_cases", "sum"), mean_rate_per_100k=("recorded_rate_per_100k", "mean"), monitor_sites=("monitor_sites", "mean"))
    linkage = records.groupby("division", as_index=False).agg(records=("diagnosis", "size"), parish_matched=("parish_matched", "sum"), pm25_matched=("pm2_5", lambda x: x.notna().sum()))
    return summary.merge(linkage, on="division", how="outer")


def demographic_spatial_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Division rates by the two reliable patient demographics."""
    return panel.groupby(["division", "sex", "age_band", "diagnosis"], as_index=False).agg(
        recorded_cases=("recorded_cases", "sum"), mean_pm25=("pm2_5", "mean"),
        mean_rate_per_100k=("recorded_rate_per_100k", "mean"), periods=("period", "nunique"),
    )


def diagnosis_population_rate_summary(records: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Calculate labelled diagnosis-specific recorded-case rates by division.

    The denominator is the total division population, and rates are annualised
    over the observed 2020--2024 record-review years. They describe facility
    records, rather than community incidence.
    """
    data = records.loc[records.in_linked_window & records.valid_kampala_division].copy()
    data = data.dropna(subset=["division", "diagnosis", "visit_date"])
    data["year"] = data.visit_date.dt.year
    division_population = population_denominator_summary(population)[["division", "population_total"]]
    observed_years = data.groupby("division", as_index=False).agg(observed_years=("year", "nunique"))
    rates = data.groupby(["division", "diagnosis"], as_index=False).agg(recorded_cases=("diagnosis", "size"))
    # Add Total Respiratory Diseases row across all diagnoses
    total_rates = data.groupby("division", as_index=False).agg(recorded_cases=("diagnosis", "size"))
    total_rates["diagnosis"] = "Total Respiratory Diseases"
    rates = pd.concat([rates, total_rates], ignore_index=True)
    rates = rates.merge(division_population, on="division", how="left", validate="many_to_one")
    rates = rates.merge(observed_years, on="division", how="left", validate="many_to_one")
    rates["annualized_recorded_rate_per_100k"] = (
        rates.recorded_cases / rates.population_total / rates.observed_years * 100000
    )
    return rates.sort_values(["diagnosis", "division"]).reset_index(drop=True)


def parish_linkage_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Auditable parish linkage report; unmatched names are never reassigned."""
    return records.groupby(["division", "parish_raw", "parish_key", "parish_matched"], dropna=False, as_index=False).agg(
        records=("diagnosis", "size"), first_visit=("visit_date", "min"), last_visit=("visit_date", "max")
    ).sort_values(["parish_matched", "records"], ascending=[True, False])


def plot_medical_record_flow(records: pd.DataFrame):
    """Sample-selection flow, adapted to the paper's analytical flowchart."""
    steps = pd.Series({
        "All reviewed records": len(records),
        "Valid visit date": int(records.visit_date.notna().sum()),
        "2020-2024 window": int(records.in_linked_window.sum()),
        "Kampala division": int(records.valid_kampala_division.sum()),
    })
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(steps.index, steps.values, color="#4575b4")
    ax.invert_yaxis(); ax.set_xlabel("Records"); ax.set_title("Medical-record linkage flow")
    for bar, value in zip(bars, steps.values): ax.text(value, bar.get_y() + bar.get_height()/2, f" {value:,}", va="center")
    fig.tight_layout()
    return ax


def plot_exposure_response(panel: pd.DataFrame):
    """Observed rate by PM2.5 quintile, stratified by respiratory diagnosis."""
    data = panel.dropna(subset=["pm2_5", "recorded_rate_per_100k"]).copy()
    data["pm25_quintile"] = pd.qcut(data.pm2_5, q=5, duplicates="drop")
    grouped = data.groupby(["diagnosis", "pm25_quintile"], observed=True).agg(pm2_5=("pm2_5", "mean"), rate=("recorded_rate_per_100k", "mean")).reset_index()
    grid = sns.FacetGrid(grouped, col="diagnosis", col_wrap=3, sharey=False, height=3.2)
    grid.map_dataframe(sns.lineplot, x="pm2_5", y="rate", marker="o", color="#d73027")
    grid.set_axis_labels("Mean PM2.5 (µg/m³)", "Recorded rate per 100,000")
    grid.fig.suptitle("Observed PM2.5-rate patterns by diagnosis", y=1.03)
    return grid


def plot_demographic_rate_heatmap(panel: pd.DataFrame):
    """Division-by-demographic heterogeneity chart inspired by spatially varying effects."""
    data = panel.groupby(["division", "sex", "age_band"], as_index=False).agg(rate=("recorded_rate_per_100k", "mean"))
    data["stratum"] = data.sex + " | " + data.age_band
    matrix = data.pivot(index="division", columns="stratum", values="rate")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    sns.heatmap(matrix, cmap="YlOrRd", linewidths=.5, annot=True, fmt=".0f", cbar_kws={"label": "Recorded rate per 100,000"}, ax=ax)
    ax.set(title="Division and demographic heterogeneity in recorded RTI rates", xlabel="Sex and age band", ylabel="Division")
    fig.tight_layout()
    return ax


def idw_parish_exposure(parishes: gpd.GeoDataFrame, monitor: pd.DataFrame, parish_name_column: str) -> pd.DataFrame:
    """Generate daily parish PM2.5 estimates once compatible parish polygons exist.

    Uses inverse-distance-squared interpolation in UTM 36N. This is deliberately
    separate from primary division estimates because geometry compatibility must
    be reviewed before inferred parish exposure is used.
    """
    required = {parish_name_column, "geometry"}
    if not required.issubset(parishes.columns):
        raise ValueError(f"Parish geometry must contain {required}")
    sites = monitor.dropna(subset=["site_latitude", "site_longitude", "pm2_5", "date"]).copy()
    sites = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites.site_longitude, sites.site_latitude), crs="EPSG:4326").to_crs("EPSG:32636")
    targets = parishes[[parish_name_column, "geometry"]].to_crs("EPSG:32636").copy()
    targets["geometry"] = targets.representative_point()
    rows = []
    for date, day in sites.groupby("date"):
        xy_sites = np.c_[day.geometry.x, day.geometry.y]
        values = day.pm2_5.to_numpy()
        for target in targets.itertuples():
            distances = np.sqrt((xy_sites[:, 0] - target.geometry.x) ** 2 + (xy_sites[:, 1] - target.geometry.y) ** 2)
            if (distances == 0).any():
                estimate = values[distances.argmin()]
            else:
                weights = 1 / distances ** 2
                estimate = np.average(values, weights=weights)
            rows.append({"date": date, "parish_key": normalise_text(getattr(target, parish_name_column)), "pm2_5_idw": estimate, "monitor_sites": len(day), "nearest_monitor_km": distances.min() / 1000})
    return pd.DataFrame(rows)


def plot_regional_atmospheric_heatmaps(
    medical_records: pd.DataFrame | None = None,
    data_dir: Path = DATA_DIR,
    save_path: str | Path | None = "figures/regional_atmospheric_heatmaps.png",
    show: bool = False,
) -> plt.Figure:
    """Generate and display region-level heatmaps of atmospheric values vs clinical & demographic features.

    Produces a 6-panel figure: 5 panels for Kampala's administrative divisions
    (Kampala Central, Kawempe, Makindye, Nakawa, Rubaga) plus 1 pooled Greater Kampala panel.
    Atmospheric metrics include ground-monitor PM2.5, temperature, humidity,
    along with meteorological wind speed and precipitation.
    Features include individual RTI diagnoses (TB, ILI, Pneumonia, Severe Pneumonia, SARI),
    pooled Total Respiratory Diseases, demographic breakdowns (Female cases, Under-5 cases),
    and active healthcare facility counts.
    """
    if medical_records is None:
        medical_records = load_medical_records(data_dir)

    # Process health records
    med = medical_records[medical_records.in_linked_window & medical_records.valid_kampala_division].dropna(subset=["division", "visit_date"]).copy()
    med["date"] = med["visit_date"].dt.normalize()

    # Diagnoses counts per division and date
    med_diag = med.groupby(["division", "date", "diagnosis"]).size().unstack(fill_value=0)
    diag_cols = [c for c in med_diag.columns]
    med_diag["Total Respiratory Diseases"] = med_diag[diag_cols].sum(axis=1)

    med["is_female"] = med["sex"].eq("Female").astype(int)
    med["is_u5"] = med["age_band"].eq("0-4").astype(int)
    demo_counts = med.groupby(["division", "date"]).agg(
        female_cases=("is_female", "sum"),
        under5_cases=("is_u5", "sum"),
        facility_count=("facility", "nunique"),
    )
    div_health = med_diag.join(demo_counts).reset_index()

    # Monitor atmospheric variables (PM2.5, temperature, humidity)
    monitor = load_monitor_data(data_dir)
    div_atm = monitor[monitor["division"].isin(KAMPALA_DIVISIONS)].groupby(["division", "date"], as_index=False).agg(
        pm2_5=("pm2_5", "mean"),
        temperature=("temperature", "mean") if "temperature" in monitor.columns else ("pm2_5", lambda x: np.nan),
        humidity=("humidity", "mean") if "humidity" in monitor.columns else ("pm2_5", lambda x: np.nan),
    )
    div_atm["year_week"] = div_atm["date"].dt.isocalendar().year.astype(str) + "-" + div_atm["date"].dt.isocalendar().week.astype(str)

    # Weekly climate variables (wind speed, precipitation)
    clim_path = data_dir / "merged_climate_respiratory_data.csv"
    if clim_path.exists():
        clim = pd.read_csv(clim_path, usecols=lambda c: c in ["date", "windspeed", "precip"]).dropna(subset=["date"])
        clim["date"] = pd.to_datetime(clim["date"]).dt.normalize()
        clim["year_week"] = clim["date"].dt.isocalendar().year.astype(str) + "-" + clim["date"].dt.isocalendar().week.astype(str)
        clim_weekly = clim[["year_week", "windspeed", "precip"]].drop_duplicates("year_week")
        div_atm = pd.merge(div_atm, clim_weekly, on="year_week", how="left")

    merged_regional = pd.merge(div_atm, div_health, on=["division", "date"], how="inner")

    atm_labels = {
        "pm2_5": "PM2.5 (µg/m³)",
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "windspeed": "Wind Speed (m/s)",
        "precip": "Precipitation (mm)",
    }
    atm_vars = [v for v in atm_labels.keys() if v in merged_regional.columns]

    feature_labels = {
        "Tuberculosis": "TB",
        "Influenza Like Illness": "ILI",
        "Pneumonia": "Pneumonia",
        "Severe Pneumonia": "Severe Pneum.",
        "Severe Acute Respiratory Infections": "SARI",
        "Total Respiratory Diseases": "Total Resp.",
        "female_cases": "Female",
        "under5_cases": "Under-5",
        "facility_count": "Facilities",
    }
    feature_vars = [f for f in feature_labels.keys() if f in merged_regional.columns]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=150)
    axes = axes.flatten()

    for idx, div in enumerate(KAMPALA_DIVISIONS):
        ax = axes[idx]
        sub = merged_regional[merged_regional["division"] == div]
        corr_df = pd.DataFrame(index=[atm_labels[a] for a in atm_vars], columns=[feature_labels[f] for f in feature_vars], dtype=float)

        for a in atm_vars:
            for f in feature_vars:
                valid = sub[[a, f]].dropna()
                if len(valid) >= 5 and valid[a].std() > 0 and valid[f].std() > 0:
                    corr_df.loc[atm_labels[a], feature_labels[f]] = valid[a].corr(valid[f])
                else:
                    corr_df.loc[atm_labels[a], feature_labels[f]] = np.nan

        sns.heatmap(
            corr_df.astype(float),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-0.5,
            vmax=0.5,
            linewidths=0.5,
            cbar_kws={"label": "Pearson r"} if idx == 1 else None,
            cbar=True if idx == 1 else False,
            ax=ax,
        )
        ax.set_title(f"{div} (N = {len(sub)} days)", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    # 6th panel: Greater Kampala pooled
    ax_all = axes[5]
    overall_corr = pd.DataFrame(index=[atm_labels[a] for a in atm_vars], columns=[feature_labels[f] for f in feature_vars], dtype=float)
    for a in atm_vars:
        for f in feature_vars:
            valid = merged_regional[[a, f]].dropna()
            if len(valid) >= 10 and valid[a].std() > 0 and valid[f].std() > 0:
                overall_corr.loc[atm_labels[a], feature_labels[f]] = valid[a].corr(valid[f])
            else:
                overall_corr.loc[atm_labels[a], feature_labels[f]] = np.nan

    sns.heatmap(
        overall_corr.astype(float),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.5,
        cbar=True,
        cbar_kws={"label": "Pearson r"},
        ax=ax_all,
    )
    ax_all.set_title("Greater Kampala (All Divisions Pooled)", fontsize=13, fontweight="bold")
    ax_all.tick_params(axis="x", rotation=30)
    ax_all.tick_params(axis="y", rotation=0)

    plt.suptitle("Region-Level Heatmaps: Atmospheric Values vs. Clinical & Demographic Features", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved regional heatmaps to {p}")

    if show:
        plt.show()

    return fig


def build_annualized_records_table(medical_records: pd.DataFrame | None = None, population_by_parish: pd.DataFrame | None = None, data_dir: Path = Path("data")) -> pd.DataFrame:
    """Construct a clean, publication-ready summary table of annualized recorded RTI cases and rates per 100,000 by division."""
    if medical_records is None:
        medical_records = load_medical_records(data_dir)
    if population_by_parish is None:
        population_by_parish = load_population(data_dir)

    d_rates = diagnosis_population_rate_summary(medical_records, population_by_parish)

    pivot_cases = d_rates[d_rates.diagnosis != "Total Respiratory Diseases"].pivot(
        index="division", columns="diagnosis", values="recorded_cases"
    ).fillna(0)
    pivot_rates = d_rates[d_rates.diagnosis != "Total Respiratory Diseases"].pivot(
        index="division", columns="diagnosis", values="annualized_recorded_rate_per_100k"
    ).fillna(0)

    div_pop = d_rates.drop_duplicates("division").set_index("division")["population_total"]
    total_cases = d_rates[d_rates.diagnosis == "Total Respiratory Diseases"].set_index("division")["recorded_cases"]
    total_rate = d_rates[d_rates.diagnosis == "Total Respiratory Diseases"].set_index("division")["annualized_recorded_rate_per_100k"]

    table = pd.DataFrame(index=div_pop.index)
    table["Population (2024)"] = div_pop.astype(int)
    for diag in sorted(pivot_rates.columns):
        table[f"{diag} (Rate/100k)"] = pivot_rates[diag].round(2)
    table["Total RTI Cases"] = total_cases.astype(int)
    table["Total RTI Rate/100k"] = total_rate.round(2)

    tot_row = pd.Series(name="Kampala total (all divisions)", dtype=object)
    tot_row["Population (2024)"] = int(div_pop.sum())
    for diag in sorted(pivot_rates.columns):
        tot_row[f"{diag} (Rate/100k)"] = round(pivot_cases[diag].sum() / div_pop.sum() / 5 * 100000, 2)
    tot_row["Total RTI Cases"] = int(total_cases.sum())
    tot_row["Total RTI Rate/100k"] = round(total_cases.sum() / div_pop.sum() / 5 * 100000, 2)

    table = pd.concat([table, pd.DataFrame([tot_row])])
    table.index.name = "Division"
    return table.reset_index()


def build_yearly_burden_table(health_spatial: pd.DataFrame | None = None, data_dir: Path = Path("data")) -> pd.DataFrame:
    """Build a summary table of recorded respiratory illness cases and division totals by year (2020-2024)."""
    if health_spatial is None:
        health_spatial = load_medical_records(data_dir)

    df = health_spatial.copy()
    if "year" not in df.columns and "visit_date" in df.columns:
        df["year"] = df["visit_date"].dt.year

    df = df[df["year"].between(2020, 2024) & df["division"].isin(KAMPALA_DIVISIONS)]
    yearly_div = df.groupby(["year", "division"]).size().unstack(fill_value=0)
    yearly_div["Kampala Total"] = yearly_div.sum(axis=1)

    tot_per_div = yearly_div.sum(axis=0)
    tot_per_div.name = "All Years (Total)"
    yearly_table = pd.concat([yearly_div, pd.DataFrame([tot_per_div])])
    yearly_table.index.name = "Year"
    return yearly_table.reset_index()


def plot_yearly_parish_burden_panel(
    parish_gdf: gpd.GeoDataFrame | None = None,
    health_spatial: pd.DataFrame | None = None,
    years: Iterable[int] = (2020, 2021, 2022, 2023, 2024),
    data_dir: Path = Path("data"),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a clean multi-panel comparative figure of Total Respiratory Diseases burden across Kampala parishes by year."""
    if parish_gdf is None:
        parish_gdf = parish_geometry(data_dir)
    if health_spatial is None:
        health_spatial = load_medical_records(data_dir)
    div_gdf = division_geometry(data_dir)

    health_spatial = health_spatial.copy()
    if "year" not in health_spatial.columns and "visit_date" in health_spatial.columns:
        health_spatial["year"] = health_spatial["visit_date"].dt.year

    years_list = list(years)
    fig, axes = plt.subplots(1, len(years_list), figsize=(4.2 * len(years_list), 5), sharex=True, sharey=True)
    if len(years_list) == 1:
        axes = [axes]
    else:
        axes = list(axes)

    year_maps = {}
    max_cases = 0
    for yr in years_list:
        yr_med = health_spatial[health_spatial.year == yr]
        yr_counts = yr_med.groupby(["division", "parish_key"]).size().rename("cases").reset_index()
        m = parish_gdf[["division", "parish_name", "parish_key", "geometry"]].merge(
            yr_counts, on=["division", "parish_key"], how="left"
        )
        m["cases"] = m["cases"].fillna(0)
        year_maps[yr] = m
        if m["cases"].max() > max_cases:
            max_cases = m["cases"].max()

    for ax, yr in zip(axes, years_list):
        m = year_maps[yr]
        tot = int(m["cases"].sum())
        m.plot(column="cases", ax=ax, cmap="YlOrRd", vmin=0, vmax=max_cases, edgecolor="#555555", linewidth=0.3)
        div_gdf.boundary.plot(ax=ax, color="#111111", linewidth=1.5)
        ax.set_title(f"{yr}\n({tot:,} cases)", fontsize=11, fontweight="bold")
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=max_cases))
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035, pad=0.08)
    cbar.set_label("Recorded Respiratory Cases per Parish", fontsize=11)
    fig.suptitle("Annual Total Respiratory Disease Burden Across Kampala Parishes (2020–2024)", fontsize=13, fontweight="bold", y=1.04)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved yearly parish panel to {p}")

    if show:
        plt.show()

    return fig


