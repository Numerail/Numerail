"""Numerail strict policy parser and linting.

PolicyParser.parse() validates a policy configuration dict and raises on the
first error. lint_config() collects ALL issues independently without raising.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from numerail.engine import (
    DimensionPolicy,
    SchemaError,
    ValidationError,
    ConstraintError,
    ResolutionError,
)

_PARSER_ALLOWED_MODES = {"project", "reject", "hybrid"}
_PARSER_DIM_POLICIES = {p.value for p in DimensionPolicy}
_PARSER_CONSUMPTION_MODES = {"nonnegative", "abs", "raw"}
_PARSER_ROUTING_KEYS = ("silent", "flagged", "confirmation", "hard_reject")


class PolicyParser:
    """Strict parser for the Numerail policy grammar (Spec S14).

    Validates every field reference, constraint dimension, budget target,
    trusted field, and routing threshold at parse time. Returns a config
    dict compatible with NumerailSystem.from_config().
    """

    def parse(self, payload: Dict[str, Any]) -> dict:
        schema_raw = payload.get("action_schema", payload.get("schema", {}))
        fields = list(schema_raw.get("fields", []))
        if not fields:
            raise SchemaError("schema.fields must be non-empty")
        if len(fields) != len(set(fields)):
            raise SchemaError("schema.fields contains duplicates")

        for fname, bounds in schema_raw.get("normalizers", {}).items():
            if fname not in fields:
                raise SchemaError(f"Normalizer references unknown field '{fname}'")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise SchemaError(f"Normalizer for '{fname}' must be [lo, hi]")
            lo, hi = float(bounds[0]), float(bounds[1])
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise SchemaError(f"Normalizer for '{fname}' must have finite bounds")
            if hi <= lo:
                raise SchemaError(f"Normalizer for '{fname}' must have hi > lo")

        polytope = payload.get("polytope", {})
        A = np.asarray(polytope.get("A", []), dtype=float)
        names = list(polytope.get("names", []))
        if A.ndim == 2 and A.shape[1] != len(fields):
            raise ConstraintError(f"polytope.A width {A.shape[1]} != schema dimension {len(fields)}")
        if len(names) != len(set(names)):
            raise ConstraintError("polytope.names contains duplicates")
        linear_names = set(names)

        # Build full constraint name index (all types) for hard-wall validation.
        # Budgets remain linear-only; hard walls can reference any named constraint.
        all_constraint_names = set(linear_names)
        for qc in payload.get("quadratic_constraints", []):
            qname = qc.get("name", "")
            if qname:
                all_constraint_names.add(qname)
        for sc in payload.get("socp_constraints", []):
            sname = sc.get("name", "")
            if sname:
                all_constraint_names.add(sname)
        for pc in payload.get("psd_constraints", []):
            pname = pc.get("name", "")
            if pname:
                all_constraint_names.add(pname)

        ec = payload.get("enforcement", {})
        mode = ec.get("mode", "project")
        if mode not in _PARSER_ALLOWED_MODES:
            raise ValidationError(f"enforcement.mode must be one of {_PARSER_ALLOWED_MODES}")

        if mode == "hybrid" and ec.get("max_distance") is None:
            raise ValidationError("max_distance is required for hybrid mode")

        for hw in ec.get("hard_wall_constraints", []):
            if hw not in all_constraint_names:
                raise ResolutionError(f"hard_wall_constraints references unknown constraint '{hw}'")

        for fname, pval in ec.get("dimension_policies", {}).items():
            if fname not in fields:
                raise SchemaError(f"dimension_policies references unknown field '{fname}'")
            pstr = pval if isinstance(pval, str) else str(pval)
            if pstr not in _PARSER_DIM_POLICIES:
                raise ValidationError(f"dimension_policies['{fname}'] invalid: '{pstr}'")

        if "routing_thresholds" in ec and ec["routing_thresholds"]:
            rt = ec["routing_thresholds"]
            vals = [float(rt.get(k, 0)) for k in _PARSER_ROUTING_KEYS]
            if not all(a <= b for a, b in zip(vals, vals[1:])):
                raise ValidationError("routing_thresholds must satisfy silent <= flagged <= confirmation <= hard_reject")

        sm = float(ec.get("safety_margin", 1.0))
        if not (0.0 < sm <= 1.0):
            raise ValidationError("safety_margin must be in (0, 1]")

        for i, b in enumerate(payload.get("budgets", [])):
            cname = b.get("constraint_name", "")
            if cname not in linear_names:
                raise ResolutionError(f"budgets[{i}].constraint_name '{cname}' not in linear rows")
            # Accept both weight-map and scalar dimension_name + weight
            w = b.get("weight")
            if isinstance(w, dict):
                for wf in w:
                    if wf not in fields:
                        raise SchemaError(f"budgets[{i}].weight references unknown field '{wf}'")
            else:
                dname = b.get("dimension_name", "")
                if dname and dname not in fields:
                    raise SchemaError(f"budgets[{i}].dimension_name '{dname}' not in schema fields")
            if float(b.get("initial", 0)) < 0:
                raise ValidationError(f"budgets[{i}].initial must be non-negative")
            cm = b.get("consumption_mode", b.get("mode", "nonnegative"))
            if cm not in _PARSER_CONSUMPTION_MODES:
                raise ValidationError(f"budgets[{i}].consumption_mode invalid: '{cm}'")

        for tf in payload.get("trusted_fields", []):
            if tf not in fields:
                raise SchemaError(f"trusted_fields references unknown field '{tf}'")

        return dict(payload)


def lint_config(payload: Dict[str, Any]) -> List[str]:
    """Validate a policy configuration and return ALL issues found.

    Unlike PolicyParser.parse() which raises on the first error,
    lint_config collects every issue independently. This supports
    specification engineering workflows where the author needs to
    see all problems at once, not fix them one at a time.

    Returns an empty list if the configuration is valid.
    """
    issues: List[str] = []
    parser = PolicyParser()

    # Schema
    schema_raw = payload.get("action_schema", payload.get("schema", {}))
    fields: List[str] = []
    try:
        fields = list(schema_raw.get("fields", []))
        if not fields:
            issues.append("schema.fields must be non-empty")
        elif len(fields) != len(set(fields)):
            issues.append("schema.fields contains duplicates")
    except Exception as e:
        issues.append(f"schema.fields: {e}")

    for fname, bounds in schema_raw.get("normalizers", {}).items():
        if fname not in fields:
            issues.append(f"normalizer references unknown field '{fname}'")
        try:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                issues.append(f"normalizer for '{fname}' must be [lo, hi]")
            else:
                lo, hi = float(bounds[0]), float(bounds[1])
                if not np.isfinite(lo) or not np.isfinite(hi):
                    issues.append(f"normalizer for '{fname}' must have finite bounds")
                elif hi <= lo:
                    issues.append(f"normalizer for '{fname}' must have hi > lo")
        except Exception as e:
            issues.append(f"normalizer for '{fname}': {e}")

    # Polytope
    polytope = payload.get("polytope", {})
    names = list(polytope.get("names", []))
    linear_names = set(names)
    if names and len(names) != len(set(names)):
        issues.append("polytope.names contains duplicates")
    try:
        A = np.asarray(polytope.get("A", []), dtype=float)
        if A.ndim == 2 and fields and A.shape[1] != len(fields):
            issues.append(f"polytope.A width {A.shape[1]} != schema dimension {len(fields)}")
    except Exception as e:
        issues.append(f"polytope.A: {e}")

    # Collect all constraint names
    all_constraint_names = set(linear_names)
    for qc in payload.get("quadratic_constraints", []):
        n = qc.get("name", "")
        if n:
            all_constraint_names.add(n)
    for sc in payload.get("socp_constraints", []):
        n = sc.get("name", "")
        if n:
            all_constraint_names.add(n)
    for pc in payload.get("psd_constraints", []):
        n = pc.get("name", "")
        if n:
            all_constraint_names.add(n)

    # Enforcement
    ec = payload.get("enforcement", {})
    mode = ec.get("mode", "project")
    if mode not in _PARSER_ALLOWED_MODES:
        issues.append(f"enforcement.mode must be one of {_PARSER_ALLOWED_MODES}")

    for hw in ec.get("hard_wall_constraints", []):
        if hw not in all_constraint_names:
            issues.append(f"hard_wall_constraints references unknown constraint '{hw}'")

    for fname, pval in ec.get("dimension_policies", {}).items():
        if fname not in fields:
            issues.append(f"dimension_policies references unknown field '{fname}'")
        pstr = pval if isinstance(pval, str) else str(pval)
        if pstr not in _PARSER_DIM_POLICIES:
            issues.append(f"dimension_policies['{fname}'] invalid: '{pstr}'")

    if "routing_thresholds" in ec and ec["routing_thresholds"]:
        rt = ec["routing_thresholds"]
        vals = [float(rt.get(k, 0)) for k in _PARSER_ROUTING_KEYS]
        if not all(a <= b for a, b in zip(vals, vals[1:])):
            issues.append("routing_thresholds must satisfy silent <= flagged <= confirmation <= hard_reject")

    sm = float(ec.get("safety_margin", 1.0))
    if not (0.0 < sm <= 1.0):
        issues.append("safety_margin must be in (0, 1]")

    if mode == "hybrid" and ec.get("max_distance") is None:
        issues.append("max_distance is required for hybrid mode")

    # Budgets
    for i, b in enumerate(payload.get("budgets", [])):
        cname = b.get("constraint_name", "")
        if cname not in linear_names:
            issues.append(f"budgets[{i}].constraint_name '{cname}' not in linear rows")
        w = b.get("weight")
        if isinstance(w, dict):
            for wf in w:
                if wf not in fields:
                    issues.append(f"budgets[{i}].weight references unknown field '{wf}'")
        else:
            dname = b.get("dimension_name", "")
            if dname and dname not in fields:
                issues.append(f"budgets[{i}].dimension_name '{dname}' not in schema fields")
        if float(b.get("initial", 0)) < 0:
            issues.append(f"budgets[{i}].initial must be non-negative")
        cm = b.get("consumption_mode", b.get("mode", "nonnegative"))
        if cm not in _PARSER_CONSUMPTION_MODES:
            issues.append(f"budgets[{i}].consumption_mode invalid: '{cm}'")

    # Trusted fields
    for tf in payload.get("trusted_fields", []):
        if tf not in fields:
            issues.append(f"trusted_fields references unknown field '{tf}'")

    return issues
