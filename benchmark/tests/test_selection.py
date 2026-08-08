# ABOUTME: Tests for choosing the FIVES benchmark subset and staging it for the pipeline.
# ABOUTME: Covers the per-disease quota, the quality filter, determinism, and file staging.

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.selection import DISEASES, SPLIT, UNSCORABLE_KEYS, select, stage


def make_manifest(per_disease=20):
    """A manifest with a mix of quality scores and both splits in every disease group."""
    rows = []
    for disease in DISEASES:
        for split in ("test", "train"):
            for number in range(per_disease):
                rows.append(
                    {
                        "key": f"{split}_{number}_{disease[0].upper()}",
                        "split": split,
                        "disease": disease,
                        "quality_score": 3 if number % 2 == 0 else 1,
                    }
                )
    return pd.DataFrame(rows)


def test_selects_eight_of_each_disease():
    chosen = select(make_manifest())
    assert len(chosen) == 32
    assert chosen["disease"].value_counts().to_dict() == {disease: 8 for disease in DISEASES}


def test_selects_only_the_requested_quality():
    chosen = select(make_manifest())
    assert set(chosen["quality_score"]) == {3}


def test_is_deterministic():
    manifest = make_manifest()
    first = select(manifest)
    second = select(manifest.sample(frac=1, random_state=7))
    assert list(first["key"]) == list(second["key"])


def test_selects_only_from_the_held_out_split():
    chosen = select(make_manifest())
    assert set(chosen["split"]) == {SPLIT}


def test_never_selects_an_unscorable_image():
    """FIVES ships two empty annotations and one that belongs to a different image."""
    manifest = make_manifest()
    unscorable = sorted(UNSCORABLE_KEYS)[0]
    manifest.loc[manifest.index[0], "key"] = unscorable
    manifest.loc[manifest.index[0], "quality_score"] = 3

    assert unscorable not in set(select(manifest, split=None)["key"])


def test_unscorable_keys_names_the_known_fives_problems():
    assert UNSCORABLE_KEYS == frozenset({"train_447_G", "train_448_G", "train_174_D"})


def test_rejects_a_disease_with_too_few_images():
    manifest = make_manifest()
    manifest = manifest[~((manifest["disease"] == "amd") & (manifest["quality_score"] == 3))]
    with pytest.raises(ValueError, match="amd"):
        select(manifest)


def test_rejects_a_disease_short_of_images_in_the_chosen_split():
    manifest = make_manifest()
    keep = ~((manifest["disease"] == "dr") & (manifest["split"] == SPLIT))
    with pytest.raises(ValueError, match="dr"):
        select(manifest[keep])


def test_stage_copies_native_originals(tmp_path):
    store = tmp_path / "fives"
    (store / "original" / "images").mkdir(parents=True)
    chosen = select(make_manifest())
    for key in chosen["key"]:
        (store / "original" / "images" / f"{key}.png").write_bytes(b"fake-png")

    destination = tmp_path / "images"
    staged = stage(chosen, store, destination)

    assert len(staged) == 32
    assert sorted(p.name for p in destination.iterdir()) == sorted(f"{k}.png" for k in chosen["key"])
    assert (destination / staged[0]).read_bytes() == b"fake-png"


def test_stage_clears_stale_images(tmp_path):
    store = tmp_path / "fives"
    (store / "original" / "images").mkdir(parents=True)
    chosen = select(make_manifest())
    for key in chosen["key"]:
        (store / "original" / "images" / f"{key}.png").write_bytes(b"fake-png")

    destination = tmp_path / "images"
    destination.mkdir()
    (destination / "leftover.png").write_bytes(b"stale")

    stage(chosen, store, destination)

    assert not (destination / "leftover.png").exists()


def test_stage_reports_a_missing_original(tmp_path):
    store = tmp_path / "fives"
    (store / "original" / "images").mkdir(parents=True)
    chosen = select(make_manifest())
    with pytest.raises(FileNotFoundError, match="original"):
        stage(chosen, store, tmp_path / "images")
