"""Tests for runtime preflight helpers."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plane_llm_utils import build_runtime_preflight, list_oci_profiles, resolve_runtime_config_data


class TestPlaneLLMUtils(unittest.TestCase):
    def test_list_oci_profiles_includes_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config"
            config_path.write_text("[DEFAULT]\nregion=us-ashburn-1\n\n[plane]\nregion=us-chicago-1\n", encoding="utf-8")

            profiles = list_oci_profiles(str(config_path))

        self.assertEqual(profiles, ["DEFAULT", "plane"])

    def test_resolve_runtime_config_data_uses_environment_and_default_profile(self):
        with patch.dict(
            os.environ,
            {
                "PLANELLM_COMPARTMENT_ID": "ocid1.compartment.oc1..test",
                "PLANELLM_MODEL_ID": "ocid1.generativeaimodel.oc1.test",
            },
            clear=False,
        ):
            with patch("plane_llm_utils.list_oci_profiles", return_value=["DEFAULT", "plane"]):
                config_data = resolve_runtime_config_data("missing-config.yaml")

        self.assertEqual(config_data["compartment_id"], "ocid1.compartment.oc1..test")
        self.assertEqual(config_data["model_id"], "ocid1.generativeaimodel.oc1.test")
        self.assertEqual(config_data["config_profile"], "DEFAULT")

    def test_build_runtime_preflight_prefers_demo_when_live_not_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resources = Path(temp_dir) / "resources"
            resources.mkdir()
            with patch("plane_llm_utils.list_oci_profiles", return_value=["DEFAULT"]):
                with patch("plane_llm_utils.importlib.util.find_spec", side_effect=lambda name: object()):
                    with patch("plane_llm_utils.shutil.which", return_value="/usr/bin/tool"):
                        status = build_runtime_preflight("missing-config.yaml", resources_dir=resources)

        self.assertFalse(status["live_ready"])
        self.assertTrue(status["demo_ready"])
        self.assertEqual(status["recommended_mode"], "demo")
