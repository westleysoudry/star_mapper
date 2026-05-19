"""Tests for config loading."""

import researcher_mapper.settings as settings_module


def test_config_loads_from_package_resources_without_checkout_config(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_module, "_ROOT", tmp_path)

    caps = settings_module.load_bucket_caps()

    assert caps["in_area_collaborators"] == 20
