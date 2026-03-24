"""Parser tests: both grammars, hard walls on all constraint types, lint_config."""

import pytest

from numerail.engine import SchemaError, ValidationError, ConstraintError, ResolutionError
from numerail.parser import PolicyParser, lint_config


@pytest.fixture
def parser():
    return PolicyParser()


@pytest.fixture
def valid_config():
    return {
        "schema": {"fields": ["x", "y"]},
        "polytope": {
            "A": [[1, 0], [0, 1], [-1, 0], [0, -1]],
            "b": [1, 1, 0, 0],
            "names": ["ux", "uy", "lx", "ly"],
        },
        "enforcement": {"mode": "project"},
    }


class TestParserBasic:
    def test_valid_config(self, parser, valid_config):
        result = parser.parse(valid_config)
        assert result is not None

    def test_action_schema_key(self, parser):
        cfg = {
            "action_schema": {"fields": ["x", "y"]},
            "polytope": {
                "A": [[1, 0], [0, 1], [-1, 0], [0, -1]],
                "b": [1, 1, 0, 0],
                "names": ["ux", "uy", "lx", "ly"],
            },
            "enforcement": {"mode": "project"},
        }
        result = parser.parse(cfg)
        assert result is not None

    def test_empty_fields_rejected(self, parser):
        with pytest.raises(SchemaError):
            parser.parse({"schema": {"fields": []}, "enforcement": {"mode": "project"}})

    def test_duplicate_fields_rejected(self, parser):
        with pytest.raises(SchemaError):
            parser.parse({"schema": {"fields": ["x", "x"]}, "enforcement": {"mode": "project"}})

    def test_invalid_mode_rejected(self, parser, valid_config):
        valid_config["enforcement"]["mode"] = "invalid"
        with pytest.raises(ValidationError):
            parser.parse(valid_config)


class TestHardWalls:
    def test_hard_wall_linear(self, parser, valid_config):
        valid_config["enforcement"]["hard_wall_constraints"] = ["ux"]
        result = parser.parse(valid_config)
        assert result is not None

    def test_hard_wall_quadratic(self, parser, valid_config):
        valid_config["quadratic_constraints"] = [
            {"Q": [[1, 0], [0, 1]], "a": [0, 0], "b": 1.0, "name": "ball"}
        ]
        valid_config["enforcement"]["hard_wall_constraints"] = ["ball"]
        result = parser.parse(valid_config)
        assert result is not None

    def test_hard_wall_socp(self, parser, valid_config):
        valid_config["socp_constraints"] = [
            {"M": [[1, 0], [0, 1]], "q": [0, 0], "c": [0, 0], "d": 1.0, "name": "cone"}
        ]
        valid_config["enforcement"]["hard_wall_constraints"] = ["cone"]
        result = parser.parse(valid_config)
        assert result is not None

    def test_hard_wall_psd(self, parser, valid_config):
        valid_config["psd_constraints"] = [
            {"matrices": [[[1, 0], [0, 1]], [[-1, 0], [0, 0]]], "name": "psd_cstr"}
        ]
        valid_config["enforcement"]["hard_wall_constraints"] = ["psd_cstr"]
        result = parser.parse(valid_config)
        assert result is not None

    def test_hard_wall_unknown_rejected(self, parser, valid_config):
        valid_config["enforcement"]["hard_wall_constraints"] = ["nonexistent"]
        with pytest.raises(ResolutionError):
            parser.parse(valid_config)


class TestBudgetGrammar:
    def test_scalar_weight(self, parser, valid_config):
        valid_config["budgets"] = [{
            "name": "cap", "constraint_name": "ux",
            "dimension_name": "x", "weight": 1.0,
            "initial": 100.0, "consumption_mode": "nonnegative",
        }]
        result = parser.parse(valid_config)
        assert result is not None

    def test_weight_map(self, parser, valid_config):
        valid_config["budgets"] = [{
            "name": "cap", "constraint_name": "ux",
            "weight": {"x": 0.6, "y": 0.4},
            "initial": 100.0, "mode": "nonnegative",
        }]
        result = parser.parse(valid_config)
        assert result is not None

    def test_weight_map_unknown_field_rejected(self, parser, valid_config):
        valid_config["budgets"] = [{
            "name": "cap", "constraint_name": "ux",
            "weight": {"x": 0.6, "z": 0.4},
            "initial": 100.0,
        }]
        with pytest.raises(SchemaError):
            parser.parse(valid_config)

    def test_budget_unknown_constraint_rejected(self, parser, valid_config):
        valid_config["budgets"] = [{
            "name": "cap", "constraint_name": "nonexistent",
            "dimension_name": "x", "initial": 100.0,
        }]
        with pytest.raises(ResolutionError):
            parser.parse(valid_config)


class TestLintConfig:
    def test_valid_returns_empty(self, valid_config):
        assert lint_config(valid_config) == []

    def test_multiple_issues_collected(self):
        bad = {
            "schema": {"fields": ["x", "y"], "normalizers": {"x": [5, 5], "z": [0, 1]}},
            "polytope": {
                "A": [[1, 0], [0, 1], [-1, 0], [0, -1]],
                "b": [1, 1, 0, 0],
                "names": ["ux", "uy", "lx", "ly"],
            },
            "enforcement": {
                "mode": "project",
                "hard_wall_constraints": ["nonexistent"],
                "dimension_policies": {"bad_field": "forbidden"},
            },
            "budgets": [{"name": "b", "constraint_name": "missing",
                         "dimension_name": "z", "initial": -1}],
            "trusted_fields": ["unknown"],
        }
        issues = lint_config(bad)
        assert len(issues) >= 5
        # Verify it found specific problems
        issue_text = "\n".join(issues)
        assert "normalizer" in issue_text.lower()
        assert "hard_wall" in issue_text.lower()
        assert "trusted_fields" in issue_text.lower()

    def test_lint_catches_action_schema_key(self):
        cfg = {
            "action_schema": {"fields": ["x"]},
            "polytope": {"A": [[1], [-1]], "b": [1, 0], "names": ["ux", "lx"]},
            "enforcement": {"mode": "project"},
        }
        assert lint_config(cfg) == []


class TestHybridModeValidation:
    """Regression for P1#3: parse() must reject hybrid mode without max_distance."""

    def test_hybrid_without_max_distance_rejected(self, parser, valid_config):
        valid_config["enforcement"] = {"mode": "hybrid"}
        with pytest.raises(ValidationError):
            parser.parse(valid_config)

    def test_hybrid_with_max_distance_accepted(self, parser, valid_config):
        valid_config["enforcement"] = {"mode": "hybrid", "max_distance": 0.5}
        result = parser.parse(valid_config)
        assert result is not None
