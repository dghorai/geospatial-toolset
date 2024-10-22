from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    s1_data_file: Path


@dataclass(frozen=True)
class StaticFilepathConfig:
    gee_private_key: Path
    ind_10x10km_grid: Path
    sixs_exe_file: Path


@dataclass(frozen=True)
class ConstantParametersConfig:
    crop_year: int
    crop_season: str
    country: str
    days_offset_inside: int
    days_offset_outside: int
    days_offset_type: str


@dataclass(frozen=True)
class MLFeatureSelectorConfig:
    supervised_type: str
    label_id_columns: str
    label_name_columns: str
    drop_columns: list
    feature_selection_option: list
    n_feature_to_select: int
