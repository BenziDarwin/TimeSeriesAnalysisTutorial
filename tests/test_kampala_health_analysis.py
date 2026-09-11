import json
from pathlib import Path

import numpy as np

from src.kampala_health_analysis import (
    KAMPALA_DIVISIONS,
    OUTCOME_COLUMNS,
    aggregate_time_panel,
    build_spatial_panel,
    load_population,
    load_weekly_panel,
)


def test_weekly_panel_has_unique_dates_and_preserves_missing_outcomes():
    weekly = load_weekly_panel()
    assert weekly.date.is_unique
    assert set(OUTCOME_COLUMNS).issubset(weekly.columns)
    # Missing DHIS2 reports must not be silently converted to zero.
    assert weekly[OUTCOME_COLUMNS].isna().any().any()


def test_monthly_outcomes_equal_weekly_complete_case_sums():
    weekly = load_weekly_panel()
    monthly = aggregate_time_panel(weekly, "MS")
    for outcome in OUTCOME_COLUMNS:
        expected = weekly.set_index("date")[outcome].resample("MS").sum(min_count=1)
        actual = monthly.set_index("date")[outcome]
        assert np.allclose(expected.fillna(-1), actual.fillna(-1))


def test_population_and_spatial_offsets_are_valid_and_unmatched_parishes_remain_unmatched():
    population = load_population()
    assert set(population.division) == set(KAMPALA_DIVISIONS)
    assert not population.duplicated(["division", "parish_key"]).any()
    panel, records, quality = build_spatial_panel(Path("data"), frequency="W-SUN")
    assert (panel.population_offset > 0).all()
    assert set(panel.division).issubset(KAMPALA_DIVISIONS)
    assert records.parish_matched.dtype == bool
    assert records.parish_matched.sum() < len(records)
    assert quality.loc[quality.metric.eq("records_with_same_day_division_pm25"), "value"].iloc[0] > 0


def test_notebooks_are_valid_and_use_shared_pipeline():
    for filename in ["analysis.ipynb", "analysis_no_deaths.ipynb", "analysis_month.ipynb", "analysis_year.ipynb", "analysis_month_no_deaths.ipynb"]:
        notebook = json.loads(Path(filename).read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "src.kampala_health_analysis" in source
        assert "plot_regional_atmospheric_heatmaps" in source
        assert "build_parish_spatial_summary" in source
        assert "plot_parish_map" in source
        assert "build_annualized_records_table" in source
        assert "build_yearly_burden_table" in source
        assert "plot_yearly_parish_burden_panel" in source
        assert notebook["nbformat"] == 4


def test_plot_regional_atmospheric_heatmaps_creates_figure():
    from src.kampala_health_analysis import plot_regional_atmospheric_heatmaps
    fig = plot_regional_atmospheric_heatmaps(save_path=None)
    assert fig is not None
    assert len(fig.axes) >= 6


def test_parish_geometry_and_spatial_summary():
    from src.kampala_health_analysis import parish_geometry, build_parish_spatial_summary, plot_parish_map
    parishes = parish_geometry()
    assert len(parishes) == 95
    assert set(parishes.division) == set(KAMPALA_DIVISIONS)
    assert parishes.crs.to_epsg() == 4326

    summary = build_parish_spatial_summary()
    assert len(summary) == 95
    assert "mean_pm25" in summary.columns
    assert "total_respiratory_cases" in summary.columns
    assert "annualized_rate_per_100k" in summary.columns
    assert summary["mean_pm25"].notna().all()

    ax = plot_parish_map(summary, "mean_pm25", "Test PM2.5 Parish Map")
    assert ax is not None


def test_annualized_and_yearly_burden_tables():
    from src.kampala_health_analysis import build_annualized_records_table, build_yearly_burden_table, plot_yearly_parish_burden_panel
    annualized_tbl = build_annualized_records_table()
    assert len(annualized_tbl) == 6  # 5 divisions + 1 Kampala total
    assert "Kampala total (all divisions)" in annualized_tbl["Division"].values
    assert "Total RTI Cases" in annualized_tbl.columns
    assert "Total RTI Rate/100k" in annualized_tbl.columns

    yearly_tbl = build_yearly_burden_table()
    assert "All Years (Total)" in yearly_tbl["Year"].values
    assert "Kampala Total" in yearly_tbl.columns

    fig = plot_yearly_parish_burden_panel(show=False)
    assert fig is not None
    assert len(fig.axes) >= 5



