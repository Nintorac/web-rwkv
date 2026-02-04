#!/usr/bin/env python3
"""
Pytest tests for benchmark config validation.

Run with:
    pytest benchmarks/test_validate_config.py -v
"""

import pytest
from pathlib import Path

from validate_config import ConfigValidator, load_config, validate_config_file


# Path to the actual config file
CONFIG_PATH = Path(__file__).parent / "config.yaml"


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_valid_minimal_config(self):
        """Test that a valid minimal config passes validation."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {
                    "models": ["test_model"],
                    "backends": ["wgpu"],
                    "scenarios": ["decode_only"],
                },
                "dev": {
                    "models": ["test_model"],
                    "backends": ["wgpu"],
                    "scenarios": ["decode_only"],
                },
                "full": {
                    "models": ["test_model"],
                    "backends": ["wgpu"],
                    "scenarios": ["decode_only"],
                },
            },
            "models": [
                {
                    "model_id": "abc123",
                    "model_name": "test_model",
                    "model_size": "1m",
                    "path": "test.st",
                    "tags": {"rwkv_version": "v7"},
                }
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {
                "directory": "results",
                "filename_pattern": "bench_{profile}.jsonl",
            },
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        assert validator.validate() is True
        assert len(validator.get_errors()) == 0

    def test_missing_schema_version(self):
        """Test that missing schema_version is detected."""
        config = {
            "profiles": {},
            "models": [],
            "backends": [],
            "scenarios": {},
            "output": {},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("schema_version" in e.path for e in errors)

    def test_invalid_schema_version_type(self):
        """Test that non-integer schema_version is rejected."""
        config = {
            "schema_version": "1.0",
            "profiles": {"smoke": {}, "dev": {}, "full": {}},
            "models": [],
            "backends": [],
            "scenarios": {},
            "output": {},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("schema_version" in e.path and "integer" in e.message for e in errors)

    def test_missing_required_profiles(self):
        """Test that missing smoke/dev/full profiles are detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "custom_only": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [],
            "backends": [],
            "scenarios": {},
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("profiles.smoke" in e.path for e in errors)
        assert any("profiles.dev" in e.path for e in errors)
        assert any("profiles.full" in e.path for e in errors)

    def test_missing_required_scenarios(self):
        """Test that missing required scenarios are detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [],
            "backends": [],
            "scenarios": {
                "decode_only": {},
                # missing prefill_uniform and prefill_mixed
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("prefill_uniform" in e.path for e in errors)
        assert any("prefill_mixed" in e.path for e in errors)

    def test_missing_model_fields(self):
        """Test that missing required model fields are detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_name": "test",
                    # missing model_id, model_size, path, tags
                }
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("model_id" in e.path for e in errors)
        assert any("model_size" in e.path for e in errors)
        assert any("path" in e.path for e in errors)
        assert any("tags" in e.path for e in errors)

    def test_missing_rwkv_version_in_tags(self):
        """Test that missing rwkv_version in tags is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_id": "abc",
                    "model_name": "test",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"domain": "test"},  # missing rwkv_version
                }
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("rwkv_version" in e.path for e in errors)

    def test_duplicate_model_name(self):
        """Test that duplicate model_name is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_id": "abc",
                    "model_name": "duplicate",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"rwkv_version": "v7"},
                },
                {
                    "model_id": "def",
                    "model_name": "duplicate",
                    "model_size": "2m",
                    "path": "y.st",
                    "tags": {"rwkv_version": "v7"},
                },
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("Duplicate model_name" in e.message for e in errors)

    def test_duplicate_model_id(self):
        """Test that duplicate model_id is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_id": "same_id",
                    "model_name": "model1",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"rwkv_version": "v7"},
                },
                {
                    "model_id": "same_id",
                    "model_name": "model2",
                    "model_size": "2m",
                    "path": "y.st",
                    "tags": {"rwkv_version": "v7"},
                },
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("Duplicate model_id" in e.message for e in errors)

    def test_duplicate_backend_id(self):
        """Test that duplicate backend_id is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [],
            "backends": [
                {"backend_id": "wgpu"},
                {"backend_id": "wgpu"},
            ],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("Duplicate backend_id" in e.message for e in errors)

    def test_invalid_model_reference_in_profile(self):
        """Test that invalid model reference in profile is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {
                    "models": ["nonexistent"],
                    "backends": ["wgpu"],
                    "scenarios": ["decode_only"],
                },
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_id": "abc",
                    "model_name": "real_model",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"rwkv_version": "v7"},
                }
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("nonexistent" in e.message for e in errors)

    def test_invalid_backend_reference_in_profile(self):
        """Test that invalid backend reference in profile is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {
                    "models": ["test"],
                    "backends": ["cuda"],  # invalid
                    "scenarios": ["decode_only"],
                },
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_id": "abc",
                    "model_name": "test",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"rwkv_version": "v7"},
                }
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("cuda" in e.message for e in errors)

    def test_invalid_scenario_reference_in_profile(self):
        """Test that invalid scenario reference in profile is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {
                    "models": ["test"],
                    "backends": ["wgpu"],
                    "scenarios": ["nonexistent_scenario"],
                },
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_id": "abc",
                    "model_name": "test",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"rwkv_version": "v7"},
                }
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("nonexistent_scenario" in e.message for e in errors)

    def test_invalid_model_reference_in_sweep(self):
        """Test that invalid model reference in sweep is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_id": "abc",
                    "model_name": "test",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"rwkv_version": "v7"},
                }
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "sweeps": {
                "test_sweep": {
                    "models": ["nonexistent"],
                    "backends": ["wgpu"],
                }
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("nonexistent" in e.message for e in errors)

    def test_invalid_backend_reference_in_sweep(self):
        """Test that invalid backend reference in sweep is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_id": "abc",
                    "model_name": "test",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"rwkv_version": "v7"},
                }
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "sweeps": {
                "test_sweep": {
                    "models": ["test"],
                    "backends": ["hip"],  # invalid
                }
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("hip" in e.message for e in errors)

    def test_missing_output_fields(self):
        """Test that missing required output fields are detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [],
            "backends": [],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {},  # missing directory and filename_pattern
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("directory" in e.path for e in errors)
        assert any("filename_pattern" in e.path for e in errors)

    def test_empty_models_list(self):
        """Test that empty models list is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("At least one model" in e.message for e in errors)

    def test_empty_backends_list(self):
        """Test that empty backends list is detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {"models": [], "backends": [], "scenarios": []},
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [
                {
                    "model_id": "abc",
                    "model_name": "test",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"rwkv_version": "v7"},
                }
            ],
            "backends": [],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("At least one backend" in e.message for e in errors)

    def test_wrong_type_for_sections(self):
        """Test that wrong types for sections are detected."""
        config = {
            "schema_version": 1,
            "profiles": "should be dict",
            "models": "should be list",
            "backends": "should be list",
            "scenarios": "should be dict",
            "output": "should be dict",
            "skip_conditions": "should be dict",
            "limits": "should be dict",
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert len(errors) > 0

    def test_wrong_type_for_profile_fields(self):
        """Test that wrong types for profile fields are detected."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {
                    "models": "should be list",
                    "backends": "should be list",
                    "scenarios": "should be list",
                    "batch_sizes": "should be list",
                },
                "dev": {"models": [], "backends": [], "scenarios": []},
                "full": {"models": [], "backends": [], "scenarios": []},
            },
            "models": [],
            "backends": [],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        validator.validate()
        errors = validator.get_errors()
        assert any("Expected array" in e.message for e in errors)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_existing_config(self):
        """Test loading the actual config file."""
        if CONFIG_PATH.exists():
            config, error = load_config(CONFIG_PATH)
            assert error is None
            assert config is not None
            assert isinstance(config, dict)

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        config, error = load_config(Path("/nonexistent/path/config.yaml"))
        assert config is None
        assert "not found" in error.lower()


class TestValidateConfigFile:
    """Tests for validate_config_file function."""

    def test_validate_actual_config(self):
        """Test validating the actual config file."""
        if CONFIG_PATH.exists():
            is_valid, errors = validate_config_file(CONFIG_PATH)
            # The actual config should be valid (no errors, possibly warnings)
            actual_errors = [e for e in errors if e.severity == "error"]
            assert len(actual_errors) == 0, f"Config has errors: {actual_errors}"


class TestExitCodes:
    """Tests for exit code behavior."""

    def test_exit_zero_on_valid(self):
        """Test that valid config would result in exit 0."""
        config = {
            "schema_version": 1,
            "profiles": {
                "smoke": {
                    "models": ["test"],
                    "backends": ["wgpu"],
                    "scenarios": ["decode_only"],
                },
                "dev": {
                    "models": ["test"],
                    "backends": ["wgpu"],
                    "scenarios": ["decode_only"],
                },
                "full": {
                    "models": ["test"],
                    "backends": ["wgpu"],
                    "scenarios": ["decode_only"],
                },
            },
            "models": [
                {
                    "model_id": "abc",
                    "model_name": "test",
                    "model_size": "1m",
                    "path": "x.st",
                    "tags": {"rwkv_version": "v7"},
                }
            ],
            "backends": [{"backend_id": "wgpu"}],
            "scenarios": {
                "decode_only": {},
                "prefill_uniform": {},
                "prefill_mixed": {},
            },
            "output": {"directory": ".", "filename_pattern": "x"},
            "skip_conditions": {},
            "limits": {},
        }
        validator = ConfigValidator(config)
        is_valid = validator.validate()
        exit_code = 0 if is_valid else 1
        assert exit_code == 0

    def test_exit_nonzero_on_invalid(self):
        """Test that invalid config would result in exit 1."""
        config = {}  # Empty config
        validator = ConfigValidator(config)
        is_valid = validator.validate()
        exit_code = 0 if is_valid else 1
        assert exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
