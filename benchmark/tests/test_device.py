# ABOUTME: Tests for the shared device selection used by every AutoMorph inference module.
# ABOUTME: Covers the AUTOMORPH_DEVICE override, auto-detection, and rejection of bad values.

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from automorph_device import select_device


def test_override_forces_cpu(monkeypatch):
    monkeypatch.setenv("AUTOMORPH_DEVICE", "cpu")
    assert select_device() == torch.device("cpu")


def test_override_wins_over_available_accelerators(monkeypatch):
    """The override must hold even on a machine where CUDA or MPS is present."""
    monkeypatch.setenv("AUTOMORPH_DEVICE", "cpu")
    device = select_device()
    assert device.type == "cpu"
    assert torch.zeros(1, device=device).device.type == "cpu"


def test_auto_detection_returns_a_usable_device(monkeypatch):
    monkeypatch.delenv("AUTOMORPH_DEVICE", raising=False)
    device = select_device()
    assert torch.zeros(1, device=device).device.type == device.type


def test_local_rank_is_applied_to_cuda_only(monkeypatch):
    monkeypatch.setenv("AUTOMORPH_DEVICE", "cpu")
    assert select_device(local_rank=1) == torch.device("cpu")


def test_unknown_device_is_rejected(monkeypatch):
    monkeypatch.setenv("AUTOMORPH_DEVICE", "tpu")
    with pytest.raises(RuntimeError, match="AUTOMORPH_DEVICE"):
        select_device()
