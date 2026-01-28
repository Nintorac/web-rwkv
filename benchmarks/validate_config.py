#!/usr/bin/env python3
"""
Config validation for web-rwkv benchmark configuration.

Validates the YAML config file against the expected schema, checking:
- Required sections and fields
- Cross-references between sections
- File paths for models (with warnings)

Usage:
    python benchmarks/validate_config.py benchmarks/config.yaml
    python benchmarks/validate_config.py --test  # Run built-in tests

Exit codes:
    0 - Valid config
    1 - Validation errors found
    2 - File not found or parse error
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


class ValidationError:
    """Represents a single validation error."""

    def __init__(self, path: str, message: str, severity: str = "error"):
        self.path = path  # JSON-like path to the field, e.g., "models[0].model_id"
        self.message = message
        self.severity = severity  # "error" or "warning"

    def __str__(self) -> str:
        prefix = "ERROR" if self.severity == "error" else "WARNING"
        return f"{prefix}: {self.path}: {self.message}"


class ConfigValidator:
    """Validates benchmark configuration."""

    REQUIRED_TOP_LEVEL = [
        "schema_version",
        "profiles",
        "models",
        "backends",
        "scenarios",
        "output",
        "skip_conditions",
        "limits",
    ]

    REQUIRED_PROFILES = ["smoke", "dev", "full"]

    REQUIRED_MODEL_FIELDS = ["model_id", "model_name", "model_size", "path", "tags"]

    REQUIRED_BACKEND_FIELDS = ["backend_id"]

    REQUIRED_SCENARIOS = ["decode_only", "prefill_uniform", "prefill_mixed"]

    REQUIRED_OUTPUT_FIELDS = ["directory", "filename_pattern"]

    def __init__(self, config: dict, config_path: Optional[Path] = None):
        self.config = config
        self.config_path = config_path
        self.errors: list[ValidationError] = []

    def add_error(self, path: str, message: str, severity: str = "error") -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(path, message, severity))

    def validate(self) -> bool:
        """Run all validations. Returns True if valid (no errors)."""
        self._validate_schema_version()
        self._validate_top_level_sections()
        self._validate_profiles()
        self._validate_models()
        self._validate_backends()
        self._validate_scenarios()
        self._validate_sweeps()
        self._validate_output()
        self._validate_skip_conditions()
        self._validate_limits()
        self._validate_cross_references()
        self._validate_model_files()

        return not any(e.severity == "error" for e in self.errors)

    def _validate_schema_version(self) -> None:
        """Validate schema_version field."""
        if "schema_version" not in self.config:
            self.add_error("schema_version", "Missing required field 'schema_version'")
            return

        version = self.config["schema_version"]
        if not isinstance(version, int):
            self.add_error(
                "schema_version",
                f"Expected integer, got {type(version).__name__}: {version}",
            )

    def _validate_top_level_sections(self) -> None:
        """Validate all required top-level sections exist."""
        for section in self.REQUIRED_TOP_LEVEL:
            if section not in self.config:
                self.add_error(section, f"Missing required section '{section}'")

    def _validate_profiles(self) -> None:
        """Validate profiles section."""
        if "profiles" not in self.config:
            return  # Already reported as missing

        profiles = self.config["profiles"]
        if not isinstance(profiles, dict):
            self.add_error("profiles", f"Expected object, got {type(profiles).__name__}")
            return

        # Check required profiles exist
        for profile_name in self.REQUIRED_PROFILES:
            if profile_name not in profiles:
                self.add_error(
                    f"profiles.{profile_name}",
                    f"Missing required profile '{profile_name}'",
                )

        # Validate each profile
        for name, profile in profiles.items():
            self._validate_profile(name, profile)

    def _validate_profile(self, name: str, profile: dict) -> None:
        """Validate a single profile."""
        path = f"profiles.{name}"

        if not isinstance(profile, dict):
            self.add_error(path, f"Expected object, got {type(profile).__name__}")
            return

        # Check for models list
        if "models" not in profile:
            self.add_error(f"{path}.models", "Missing required field 'models'")
        elif not isinstance(profile["models"], list):
            self.add_error(
                f"{path}.models",
                f"Expected array, got {type(profile['models']).__name__}",
            )

        # Check for backends list
        if "backends" not in profile:
            self.add_error(f"{path}.backends", "Missing required field 'backends'")
        elif not isinstance(profile["backends"], list):
            self.add_error(
                f"{path}.backends",
                f"Expected array, got {type(profile['backends']).__name__}",
            )

        # Check for scenarios list
        if "scenarios" not in profile:
            self.add_error(f"{path}.scenarios", "Missing required field 'scenarios'")
        elif not isinstance(profile["scenarios"], list):
            self.add_error(
                f"{path}.scenarios",
                f"Expected array, got {type(profile['scenarios']).__name__}",
            )

        # Check numeric fields if present
        numeric_fields = [
            "batch_sizes",
            "token_chunk_sizes",
            "decode_steps",
            "seq_lens",
        ]
        for field in numeric_fields:
            if field in profile:
                if not isinstance(profile[field], list):
                    self.add_error(
                        f"{path}.{field}",
                        f"Expected array, got {type(profile[field]).__name__}",
                    )
                elif not all(isinstance(x, (int, float)) for x in profile[field]):
                    self.add_error(
                        f"{path}.{field}",
                        "All elements must be numbers",
                    )

    def _validate_models(self) -> None:
        """Validate models section."""
        if "models" not in self.config:
            return  # Already reported

        models = self.config["models"]
        if not isinstance(models, list):
            self.add_error("models", f"Expected array, got {type(models).__name__}")
            return

        if len(models) == 0:
            self.add_error("models", "At least one model is required")
            return

        seen_model_names = set()
        seen_model_ids = set()

        for i, model in enumerate(models):
            path = f"models[{i}]"

            if not isinstance(model, dict):
                self.add_error(path, f"Expected object, got {type(model).__name__}")
                continue

            # Check required fields
            for field in self.REQUIRED_MODEL_FIELDS:
                if field not in model:
                    self.add_error(f"{path}.{field}", f"Missing required field '{field}'")

            # Validate model_id
            if "model_id" in model:
                model_id = model["model_id"]
                if not isinstance(model_id, str):
                    self.add_error(
                        f"{path}.model_id",
                        f"Expected string, got {type(model_id).__name__}",
                    )
                elif model_id in seen_model_ids:
                    self.add_error(
                        f"{path}.model_id",
                        f"Duplicate model_id: {model_id}",
                    )
                else:
                    seen_model_ids.add(model_id)

            # Validate model_name
            if "model_name" in model:
                model_name = model["model_name"]
                if not isinstance(model_name, str):
                    self.add_error(
                        f"{path}.model_name",
                        f"Expected string, got {type(model_name).__name__}",
                    )
                elif model_name in seen_model_names:
                    self.add_error(
                        f"{path}.model_name",
                        f"Duplicate model_name: {model_name}",
                    )
                else:
                    seen_model_names.add(model_name)

            # Validate tags
            if "tags" in model:
                tags = model["tags"]
                if not isinstance(tags, dict):
                    self.add_error(
                        f"{path}.tags",
                        f"Expected object, got {type(tags).__name__}",
                    )
                elif "rwkv_version" not in tags:
                    self.add_error(
                        f"{path}.tags.rwkv_version",
                        "Missing required field 'rwkv_version' in tags",
                    )

    def _validate_backends(self) -> None:
        """Validate backends section."""
        if "backends" not in self.config:
            return

        backends = self.config["backends"]
        if not isinstance(backends, list):
            self.add_error("backends", f"Expected array, got {type(backends).__name__}")
            return

        if len(backends) == 0:
            self.add_error("backends", "At least one backend is required")
            return

        seen_backend_ids = set()

        for i, backend in enumerate(backends):
            path = f"backends[{i}]"

            if not isinstance(backend, dict):
                self.add_error(path, f"Expected object, got {type(backend).__name__}")
                continue

            # Check required fields
            for field in self.REQUIRED_BACKEND_FIELDS:
                if field not in backend:
                    self.add_error(f"{path}.{field}", f"Missing required field '{field}'")

            # Check for duplicate backend_id
            if "backend_id" in backend:
                backend_id = backend["backend_id"]
                if not isinstance(backend_id, str):
                    self.add_error(
                        f"{path}.backend_id",
                        f"Expected string, got {type(backend_id).__name__}",
                    )
                elif backend_id in seen_backend_ids:
                    self.add_error(
                        f"{path}.backend_id",
                        f"Duplicate backend_id: {backend_id}",
                    )
                else:
                    seen_backend_ids.add(backend_id)

    def _validate_scenarios(self) -> None:
        """Validate scenarios section."""
        if "scenarios" not in self.config:
            return

        scenarios = self.config["scenarios"]
        if not isinstance(scenarios, dict):
            self.add_error("scenarios", f"Expected object, got {type(scenarios).__name__}")
            return

        # Check required scenarios exist
        for scenario_name in self.REQUIRED_SCENARIOS:
            if scenario_name not in scenarios:
                self.add_error(
                    f"scenarios.{scenario_name}",
                    f"Missing required scenario '{scenario_name}'",
                )

    def _validate_sweeps(self) -> None:
        """Validate sweeps section."""
        if "sweeps" not in self.config:
            # Sweeps are optional, but if present should be valid
            return

        sweeps = self.config["sweeps"]
        if not isinstance(sweeps, dict):
            self.add_error("sweeps", f"Expected object, got {type(sweeps).__name__}")
            return

        for name, sweep in sweeps.items():
            path = f"sweeps.{name}"

            if not isinstance(sweep, dict):
                self.add_error(path, f"Expected object, got {type(sweep).__name__}")
                continue

            # Validate sweep references models and backends
            if "models" in sweep:
                if not isinstance(sweep["models"], list):
                    self.add_error(
                        f"{path}.models",
                        f"Expected array, got {type(sweep['models']).__name__}",
                    )

            if "backends" in sweep:
                if not isinstance(sweep["backends"], list):
                    self.add_error(
                        f"{path}.backends",
                        f"Expected array, got {type(sweep['backends']).__name__}",
                    )

    def _validate_output(self) -> None:
        """Validate output section."""
        if "output" not in self.config:
            return

        output = self.config["output"]
        if not isinstance(output, dict):
            self.add_error("output", f"Expected object, got {type(output).__name__}")
            return

        for field in self.REQUIRED_OUTPUT_FIELDS:
            if field not in output:
                self.add_error(f"output.{field}", f"Missing required field '{field}'")

    def _validate_skip_conditions(self) -> None:
        """Validate skip_conditions section."""
        if "skip_conditions" not in self.config:
            return

        skip_conditions = self.config["skip_conditions"]
        if not isinstance(skip_conditions, dict):
            self.add_error(
                "skip_conditions",
                f"Expected object, got {type(skip_conditions).__name__}",
            )

    def _validate_limits(self) -> None:
        """Validate limits section."""
        if "limits" not in self.config:
            return

        limits = self.config["limits"]
        if not isinstance(limits, dict):
            self.add_error("limits", f"Expected object, got {type(limits).__name__}")

    def _validate_cross_references(self) -> None:
        """Validate that references between sections are valid."""
        # Build lookup tables
        model_names = self._get_model_names()
        backend_ids = self._get_backend_ids()
        scenario_names = self._get_scenario_names()

        # Validate profile references
        if "profiles" in self.config and isinstance(self.config["profiles"], dict):
            for profile_name, profile in self.config["profiles"].items():
                if not isinstance(profile, dict):
                    continue

                path = f"profiles.{profile_name}"

                # Check model references
                if "models" in profile and isinstance(profile["models"], list):
                    for i, model_ref in enumerate(profile["models"]):
                        if model_ref not in model_names:
                            self.add_error(
                                f"{path}.models[{i}]",
                                f"Referenced model '{model_ref}' not found in models section",
                            )

                # Check backend references
                if "backends" in profile and isinstance(profile["backends"], list):
                    for i, backend_ref in enumerate(profile["backends"]):
                        if backend_ref not in backend_ids:
                            self.add_error(
                                f"{path}.backends[{i}]",
                                f"Referenced backend '{backend_ref}' not found in backends section",
                            )

                # Check scenario references
                if "scenarios" in profile and isinstance(profile["scenarios"], list):
                    for i, scenario_ref in enumerate(profile["scenarios"]):
                        if scenario_ref not in scenario_names:
                            self.add_error(
                                f"{path}.scenarios[{i}]",
                                f"Referenced scenario '{scenario_ref}' not found in scenarios section",
                            )

        # Validate sweep references
        if "sweeps" in self.config and isinstance(self.config["sweeps"], dict):
            for sweep_name, sweep in self.config["sweeps"].items():
                if not isinstance(sweep, dict):
                    continue

                path = f"sweeps.{sweep_name}"

                # Check model references
                if "models" in sweep and isinstance(sweep["models"], list):
                    for i, model_ref in enumerate(sweep["models"]):
                        if model_ref not in model_names:
                            self.add_error(
                                f"{path}.models[{i}]",
                                f"Referenced model '{model_ref}' not found in models section",
                            )

                # Check backend references
                if "backends" in sweep and isinstance(sweep["backends"], list):
                    for i, backend_ref in enumerate(sweep["backends"]):
                        if backend_ref not in backend_ids:
                            self.add_error(
                                f"{path}.backends[{i}]",
                                f"Referenced backend '{backend_ref}' not found in backends section",
                            )

    def _validate_model_files(self) -> None:
        """Check if model files exist (warnings only)."""
        if "models" not in self.config:
            return

        models = self.config["models"]
        if not isinstance(models, list):
            return

        base_path = self.config_path.parent.parent if self.config_path else Path.cwd()

        for i, model in enumerate(models):
            if not isinstance(model, dict):
                continue

            if "path" not in model:
                continue

            model_path = model["path"]
            if not isinstance(model_path, str):
                continue

            # Try to resolve path relative to config file or cwd
            full_path = base_path / model_path

            if not full_path.exists():
                self.add_error(
                    f"models[{i}].path",
                    f"Model file not found: {model_path} (resolved to {full_path})",
                    severity="warning",
                )

    def _get_model_names(self) -> set:
        """Get all model names from config."""
        if "models" not in self.config or not isinstance(self.config["models"], list):
            return set()

        names = set()
        for model in self.config["models"]:
            if isinstance(model, dict) and "model_name" in model:
                names.add(model["model_name"])
        return names

    def _get_backend_ids(self) -> set:
        """Get all backend IDs from config."""
        if "backends" not in self.config or not isinstance(self.config["backends"], list):
            return set()

        ids = set()
        for backend in self.config["backends"]:
            if isinstance(backend, dict) and "backend_id" in backend:
                ids.add(backend["backend_id"])
        return ids

    def _get_scenario_names(self) -> set:
        """Get all scenario names from config."""
        if "scenarios" not in self.config or not isinstance(self.config["scenarios"], dict):
            return set()
        return set(self.config["scenarios"].keys())

    def get_errors(self) -> list[ValidationError]:
        """Get all validation errors."""
        return [e for e in self.errors if e.severity == "error"]

    def get_warnings(self) -> list[ValidationError]:
        """Get all validation warnings."""
        return [e for e in self.errors if e.severity == "warning"]

    def print_results(self) -> None:
        """Print validation results to stdout/stderr."""
        errors = self.get_errors()
        warnings = self.get_warnings()

        for warning in warnings:
            print(str(warning), file=sys.stderr)

        for error in errors:
            print(str(error), file=sys.stderr)

        if errors:
            print(f"\nValidation FAILED: {len(errors)} error(s)", file=sys.stderr)
            if warnings:
                print(f"                   {len(warnings)} warning(s)", file=sys.stderr)
        elif warnings:
            print(f"\nValidation PASSED with {len(warnings)} warning(s)")
        else:
            print("\nValidation PASSED: config is valid")


def load_config(path: Path) -> tuple[Optional[dict], Optional[str]]:
    """Load YAML config from file. Returns (config, error_message)."""
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                return None, "Config file is empty"
            if not isinstance(config, dict):
                return None, f"Config root must be an object, got {type(config).__name__}"
            return config, None
    except FileNotFoundError:
        return None, f"Config file not found: {path}"
    except yaml.YAMLError as e:
        return None, f"YAML parse error: {e}"
    except Exception as e:
        return None, f"Error reading config: {e}"


def validate_config_file(path: Path) -> tuple[bool, list[ValidationError]]:
    """Validate a config file. Returns (is_valid, errors)."""
    config, error = load_config(path)
    if error:
        return False, [ValidationError("", error)]

    validator = ConfigValidator(config, path)
    is_valid = validator.validate()
    return is_valid, validator.errors


def run_builtin_tests() -> bool:
    """Run built-in validation tests with sample configs."""
    print("Running built-in validation tests...\n")
    all_passed = True

    # Test 1: Valid minimal config
    print("Test 1: Valid minimal config")
    valid_config = {
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
    validator = ConfigValidator(valid_config)
    if validator.validate():
        print("  PASSED: Valid config accepted\n")
    else:
        print("  FAILED: Valid config rejected")
        for e in validator.errors:
            print(f"    {e}")
        print()
        all_passed = False

    # Test 2: Missing schema_version
    print("Test 2: Missing schema_version")
    invalid_config = dict(valid_config)
    del invalid_config["schema_version"]
    validator = ConfigValidator(invalid_config)
    if not validator.validate() and any(
        "schema_version" in e.path for e in validator.get_errors()
    ):
        print("  PASSED: Missing schema_version detected\n")
    else:
        print("  FAILED: Missing schema_version not detected\n")
        all_passed = False

    # Test 3: Missing required profile
    print("Test 3: Missing required profile (smoke)")
    invalid_config = {
        "schema_version": 1,
        "profiles": {
            "dev": {"models": [], "backends": [], "scenarios": []},
            "full": {"models": [], "backends": [], "scenarios": []},
        },
        "models": [],
        "backends": [],
        "scenarios": {},
        "output": {"directory": ".", "filename_pattern": "x"},
        "skip_conditions": {},
        "limits": {},
    }
    validator = ConfigValidator(invalid_config)
    validator.validate()
    if any("profiles.smoke" in e.path for e in validator.get_errors()):
        print("  PASSED: Missing 'smoke' profile detected\n")
    else:
        print("  FAILED: Missing 'smoke' profile not detected\n")
        all_passed = False

    # Test 4: Invalid model reference in profile
    print("Test 4: Invalid model reference in profile")
    invalid_config = {
        "schema_version": 1,
        "profiles": {
            "smoke": {
                "models": ["nonexistent_model"],
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
    validator = ConfigValidator(invalid_config)
    validator.validate()
    if any("nonexistent_model" in e.message for e in validator.get_errors()):
        print("  PASSED: Invalid model reference detected\n")
    else:
        print("  FAILED: Invalid model reference not detected\n")
        all_passed = False

    # Test 5: Missing rwkv_version in tags
    print("Test 5: Missing rwkv_version in model tags")
    invalid_config = {
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
    validator = ConfigValidator(invalid_config)
    validator.validate()
    if any("rwkv_version" in e.path for e in validator.get_errors()):
        print("  PASSED: Missing rwkv_version detected\n")
    else:
        print("  FAILED: Missing rwkv_version not detected\n")
        all_passed = False

    # Test 6: Duplicate model_name
    print("Test 6: Duplicate model_name")
    invalid_config = {
        "schema_version": 1,
        "profiles": {
            "smoke": {"models": [], "backends": [], "scenarios": []},
            "dev": {"models": [], "backends": [], "scenarios": []},
            "full": {"models": [], "backends": [], "scenarios": []},
        },
        "models": [
            {
                "model_id": "abc",
                "model_name": "duplicate_name",
                "model_size": "1m",
                "path": "x.st",
                "tags": {"rwkv_version": "v7"},
            },
            {
                "model_id": "def",
                "model_name": "duplicate_name",
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
    validator = ConfigValidator(invalid_config)
    validator.validate()
    if any("Duplicate model_name" in e.message for e in validator.get_errors()):
        print("  PASSED: Duplicate model_name detected\n")
    else:
        print("  FAILED: Duplicate model_name not detected\n")
        all_passed = False

    # Test 7: Invalid sweep reference
    print("Test 7: Invalid backend reference in sweep")
    invalid_config = {
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
            "test": {
                "models": ["test"],
                "backends": ["cuda"],  # invalid reference
            }
        },
        "output": {"directory": ".", "filename_pattern": "x"},
        "skip_conditions": {},
        "limits": {},
    }
    validator = ConfigValidator(invalid_config)
    validator.validate()
    if any("cuda" in e.message for e in validator.get_errors()):
        print("  PASSED: Invalid backend reference in sweep detected\n")
    else:
        print("  FAILED: Invalid backend reference in sweep not detected\n")
        all_passed = False

    # Summary
    print("=" * 50)
    if all_passed:
        print("All tests PASSED")
        return True
    else:
        print("Some tests FAILED")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate web-rwkv benchmark configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python validate_config.py benchmarks/config.yaml
    python validate_config.py --test
        """,
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        type=Path,
        help="Path to the config YAML file to validate",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run built-in validation tests",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only print errors, not success message",
    )

    args = parser.parse_args()

    if args.test:
        return 0 if run_builtin_tests() else 1

    if not args.config_path:
        parser.print_help()
        return 2

    config_path = args.config_path
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    config, error = load_config(config_path)
    if error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    validator = ConfigValidator(config, config_path)
    is_valid = validator.validate()

    if not args.quiet or not is_valid:
        validator.print_results()

    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
