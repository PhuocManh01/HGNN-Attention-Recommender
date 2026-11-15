"""Deterministic raw-ID to contiguous integer mappings."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class IDMapping:
    """Immutable mapping with explicit forward and reverse lookups."""

    raw_to_id: dict[str, int]

    @property
    def id_to_raw(self) -> dict[int, str]:
        return {value: key for key, value in self.raw_to_id.items()}

    def encode(self, values: Iterable[object], unknown: str = "error") -> list[int | None]:
        result: list[int | None] = []
        for value in values:
            key = str(value)
            if key not in self.raw_to_id:
                if unknown == "error":
                    raise KeyError(f"Unknown ID: {key}")
                if unknown != "null":
                    raise ValueError("unknown must be 'error' or 'null'")
                result.append(None)
            else:
                result.append(self.raw_to_id[key])
        return result

    def decode(self, values: Iterable[object], unknown: str = "error") -> list[str | None]:
        reverse = self.id_to_raw
        result: list[str | None] = []
        for value in values:
            key = int(value)
            if key not in reverse:
                if unknown == "error":
                    raise KeyError(f"Unknown mapped ID: {key}")
                if unknown != "null":
                    raise ValueError("unknown must be 'error' or 'null'")
                result.append(None)
            else:
                result.append(reverse[key])
        return result


def build_mapping(values: Iterable[object], raw_name: str = "raw_id") -> pd.DataFrame:
    """Build a deterministic mapping sorted by the string form of raw IDs."""
    unique = {str(value) for value in values if pd.notna(value)}
    ordered = sorted(unique)
    return pd.DataFrame({raw_name: ordered, "id": range(len(ordered))})


def mapping_object(mapping: pd.DataFrame, raw_name: str = "raw_id") -> IDMapping:
    """Validate a mapping dataframe and expose explicit forward/reverse lookup."""
    required = {raw_name, "id"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping is missing columns: {sorted(missing)}")
    if mapping[raw_name].isna().any() or mapping["id"].isna().any():
        raise ValueError("Mapping cannot contain null raw IDs or mapped IDs")
    if mapping[raw_name].astype(str).duplicated().any():
        raise ValueError("Mapping raw IDs must be unique")
    if mapping["id"].duplicated().any():
        raise ValueError("Mapping mapped IDs must be unique")
    ids = sorted(mapping["id"].astype(int).tolist())
    if ids != list(range(len(ids))):
        raise ValueError("Mapping IDs must be contiguous starting at zero")
    return IDMapping(dict(zip(mapping[raw_name].astype(str), mapping["id"].astype(int))))


def reverse_mapping(mapping: pd.DataFrame, raw_name: str = "raw_id") -> pd.DataFrame:
    """Return the explicit mapped-ID to original-ID view."""
    validated = mapping_object(mapping, raw_name)
    return pd.DataFrame({"id": list(validated.id_to_raw), raw_name: list(validated.id_to_raw.values())})


def apply_mapping(
    values: pd.Series,
    mapping: pd.DataFrame,
    raw_name: str = "raw_id",
) -> pd.Series:
    """Map raw IDs to integers and fail on an unknown ID."""
    lookup = dict(zip(mapping[raw_name].astype(str), mapping["id"]))
    mapped = values.astype("string").map(lookup)
    if mapped.isna().any():
        unknown = values[mapped.isna()].drop_duplicates().head(5).tolist()
        raise ValueError(f"Unknown IDs while applying mapping: {unknown}")
    return mapped.astype("int64")


def save_mapping(mapping: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(path, index=False)
