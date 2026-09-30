import json
import pytest
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.connection import Base
from backend.database.seed import seed_data
from backend.ml.dataset_generator import SyntheticDatasetGenerator, PRD_FEATURE_COLUMNS


@pytest.fixture
def in_memory_db():
    """In-memory SQLite database populated with the 3-route seed network."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    seed_data(session)
    yield session
    session.close()


def test_correct_feature_columns(in_memory_db, tmp_path):
    """Verify that all 11 PRD feature columns and metadata fields exist in the generated dataset."""
    gen = SyntheticDatasetGenerator(db=in_memory_db, seed=42)
    csv_path = tmp_path / "test_data.csv"
    meta_path = tmp_path / "test_meta.json"

    df = gen.generate_dataset(
        num_days=2,
        samples_per_segment=3,
        output_csv_path=csv_path,
        output_metadata_path=meta_path,
    )

    # Required metadata fields
    expected_meta = [
        "journey_id",
        "train_id",
        "route_id",
        "station",
        "target_station",
        "timestamp",
        "actual_remaining_minutes",
        "disruption_present",
        "is_synthetic",
    ]
    for col in expected_meta:
        assert col in df.columns, f"Missing required metadata column: {col}"

    # Required 11 PRD features
    for col in PRD_FEATURE_COLUMNS:
        assert col in df.columns, f"Missing PRD feature column: {col}"

    assert len(df) > 0
    assert csv_path.exists()
    assert meta_path.exists()


def test_target_generation(in_memory_db, tmp_path):
    """Verify target actual_remaining_minutes is strictly positive, realistic, and correlated with distance/time."""
    gen = SyntheticDatasetGenerator(db=in_memory_db, seed=42)
    df = gen.generate_dataset(num_days=3, samples_per_segment=3, output_csv_path=tmp_path / "target_test.csv")

    target = df["actual_remaining_minutes"]

    # 1. Target must be positive and non-zero
    assert (target > 0.0).all(), "Found non-positive target actual_remaining_minutes"

    # 2. Strong physical correlation with scheduled time and distance
    corr_time = df["scheduled_time_to_go_min"].corr(target)
    corr_dist = df["distance_to_go_km"].corr(target)

    assert corr_time > 0.85, f"Expected strong correlation with scheduled time, got {corr_time}"
    assert corr_dist > 0.85, f"Expected strong correlation with distance, got {corr_dist}"

    # 3. Disruption samples have higher average remaining time relative to scheduled time
    disrupted = df[df["disruption_present"] == 1]
    normal = df[df["disruption_present"] == 0]

    assert len(disrupted) > 0, "No disruption samples were generated"
    assert len(normal) > 0, "No normal samples were generated"

    disrupted_ratio = (disrupted["actual_remaining_minutes"] / disrupted["scheduled_time_to_go_min"]).mean()
    normal_ratio = (normal["actual_remaining_minutes"] / normal["scheduled_time_to_go_min"]).mean()
    assert disrupted_ratio > normal_ratio, "Disrupted journeys should have higher actual/scheduled time ratio"


def test_no_missing_required_fields(in_memory_db, tmp_path):
    """Verify zero NaN/null values across the entire generated dataset."""
    gen = SyntheticDatasetGenerator(db=in_memory_db, seed=100)
    df = gen.generate_dataset(num_days=2, samples_per_segment=4, output_csv_path=tmp_path / "null_test.csv")

    null_counts = df.isna().sum()
    for col, count in null_counts.items():
        assert count == 0, f"Column '{col}' contains {count} null/NaN values"


def test_reproducibility_using_fixed_seed(in_memory_db, tmp_path):
    """Verify that using the same random seed yields 100% identical datasets."""
    gen1 = SyntheticDatasetGenerator(db=in_memory_db, seed=42)
    df1 = gen1.generate_dataset(num_days=2, samples_per_segment=3, output_csv_path=tmp_path / "d1.csv")

    gen2 = SyntheticDatasetGenerator(db=in_memory_db, seed=42)
    df2 = gen2.generate_dataset(num_days=2, samples_per_segment=3, output_csv_path=tmp_path / "d2.csv")

    # Both runs with seed=42 must be completely identical
    pd.testing.assert_frame_equal(df1, df2)

    # Different seed must yield distinct operational variations
    gen3 = SyntheticDatasetGenerator(db=in_memory_db, seed=999)
    df3 = gen3.generate_dataset(num_days=2, samples_per_segment=3, output_csv_path=tmp_path / "d3.csv")

    assert not df1["actual_remaining_minutes"].equals(df3["actual_remaining_minutes"])


def test_metadata_file_content(in_memory_db, tmp_path):
    """Verify dataset_metadata.json contains proper disclaimer and summary statistics."""
    meta_path = tmp_path / "dataset_metadata.json"
    csv_path = tmp_path / "dataset_test.csv"
    gen = SyntheticDatasetGenerator(db=in_memory_db, seed=42)
    gen.generate_dataset(
        num_days=2,
        samples_per_segment=3,
        output_csv_path=csv_path,
        output_metadata_path=meta_path,
    )

    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["is_synthetic"] is True
    assert "SYNTHETIC SIMULATED DATASET ONLY" in meta["disclaimer"]
    assert meta["random_seed"] == 42
    assert meta["total_samples"] > 0
    assert "mean" in meta["target_distribution"]
    assert "std" in meta["target_distribution"]
    assert meta["feature_columns"] == PRD_FEATURE_COLUMNS
