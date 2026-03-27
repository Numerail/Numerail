"""Tests for EnforcementExperience and EnforcementExperienceBuffer."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any, Dict, Optional

import numpy as np
import pytest

from numerail_learn.experience import EnforcementExperienceBuffer, EnforcementExperience


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _fake_output(decision: str, enforced: Optional[Dict[str, float]] = None,
                 action_id: str = "act-001") -> Dict[str, Any]:
    """Minimal enforcement output dict."""
    return {
        "decision":       decision,
        "enforced_values": enforced,
        "action_id":      action_id,
        "feedback":       [],
        "solver_method":  "cvxpy",
        "routing_decision": None,
        "distance":       -1.0 if decision.lower() == "reject" else 0.0,
    }


def _buf() -> EnforcementExperienceBuffer:
    return EnforcementExperienceBuffer(max_size=100)


def _record(buf: EnforcementExperienceBuffer, result: str,
            distance: float = 0.0, breaker: str = "closed",
            violations=None) -> EnforcementExperience:
    proposed = np.array([10.0, 5.0, 2.0], dtype=np.float64)
    enforced_arr = None
    enforced_dict = None

    if result == "project":
        enforced_dict = {"a": 8.0, "b": 4.0, "c": 2.0}
        enforced_arr  = np.array([8.0, 4.0, 2.0], dtype=np.float64)
    elif result == "approve":
        enforced_dict = {"a": 10.0, "b": 5.0, "c": 2.0}

    out = _fake_output(result, enforced_dict)
    if result == "project":
        out["distance"] = distance if distance > 0 else 2.449
    if violations:
        out["feedback"] = [(name, mag) for name, mag in violations]

    return buf.record(
        conversation_context=[{"role": "user", "content": "do something"}],
        tool_call={"name": "act", "arguments": {"a": 10.0, "b": 5.0, "c": 2.0}},
        proposed_vector=proposed,
        enforcement_output=out,
        enforced_vector=enforced_arr,
        breaker_mode=breaker,
        budget_remaining={"gpu_shift": 1000.0, "external_api_shift": 200.0},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_record_approve():
    buf = _buf()
    exp = _record(buf, "approve")
    assert exp.result == "approve"
    assert exp.distance == 0.0
    assert exp.violations == []
    assert exp.enforced_vector is not None  # extracted from enforced_dict


def test_record_project():
    buf = _buf()
    exp = _record(buf, "project", distance=2.449)
    assert exp.result == "project"
    assert exp.distance > 0
    assert exp.enforced_vector is not None


def test_record_reject():
    buf = _buf()
    exp = _record(buf, "reject")
    assert exp.result == "reject"
    assert exp.enforced_vector is None
    assert exp.distance == -1.0


def test_buffer_max_size():
    buf = EnforcementExperienceBuffer(max_size=5)
    for i in range(8):
        _record(buf, "approve")
    assert len(buf) == 5
    # Oldest should be evicted; IDs start from exp-000001
    ids = [e.experience_id for e in buf.get_all()]
    assert all(int(eid.split("-")[1]) >= 4 for eid in ids)


def test_sample_batch():
    buf = _buf()
    for _ in range(20):
        _record(buf, "approve")
    batch = buf.sample_batch(5)
    assert len(batch) == 5
    assert all(isinstance(e, EnforcementExperience) for e in batch)


def test_sample_with_filter():
    buf = _buf()
    for _ in range(10):
        _record(buf, "approve")
    for _ in range(5):
        _record(buf, "project")
    batch = buf.sample_batch(3, filter_result="project")
    assert len(batch) == 3
    assert all(e.result == "project" for e in batch)


def test_get_project_experiences():
    buf = _buf()
    for _ in range(6):
        _record(buf, "approve")
    for _ in range(4):
        _record(buf, "project")
    projects = buf.get_project_experiences()
    assert len(projects) == 4
    assert all(e.result == "project" for e in projects)


def test_get_approve_reject_pairs():
    buf = _buf()
    for _ in range(5):
        _record(buf, "approve", breaker="closed")
    for _ in range(5):
        _record(buf, "reject", breaker="closed")
    pairs = buf.get_approve_reject_pairs()
    assert len(pairs) == 5
    for preferred, dispreferred in pairs:
        assert preferred.result == "approve"
        assert dispreferred.result == "reject"


def test_approval_rate():
    buf = _buf()
    for _ in range(7):
        _record(buf, "approve")
    for _ in range(3):
        _record(buf, "reject")
    assert abs(buf.approval_rate - 0.7) < 1e-9


def test_dimension_violation_frequency():
    buf = _buf()
    proposed = np.array([10.0, 5.0], dtype=np.float64)
    for _ in range(3):
        buf.record(
            conversation_context=[],
            tool_call={"name": "act", "arguments": {}},
            proposed_vector=proposed,
            enforcement_output={"decision": "project", "enforced_values": {"a": 8.0},
                                 "action_id": "x", "feedback": [("gpu_limit", 0.5)],
                                 "solver_method": "", "distance": 1.0},
        )
    for _ in range(2):
        buf.record(
            conversation_context=[],
            tool_call={"name": "act", "arguments": {}},
            proposed_vector=proposed,
            enforcement_output={"decision": "reject", "enforced_values": None,
                                 "action_id": "y", "feedback": [("api_limit", 1.0)],
                                 "solver_method": "", "distance": -1.0},
        )
    freq = buf.dimension_violation_frequency()
    assert freq.get("gpu_limit", 0) == 3
    assert freq.get("api_limit", 0) == 2


def test_mean_distance_by_result():
    buf = _buf()
    for _ in range(5):
        _record(buf, "approve")
    for _ in range(5):
        _record(buf, "project", distance=3.0)
    means = buf.mean_distance_by_result()
    assert means.get("approve", 0.0) == 0.0
    assert means.get("project", 0.0) > 0.0


def test_export_import_json():
    buf = _buf()
    for i in range(10):
        result = "approve" if i % 2 == 0 else "project"
        _record(buf, result)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name

    try:
        buf.export_json(path)
        buf2 = EnforcementExperienceBuffer(max_size=100)
        count = buf2.import_json(path)
        assert count == 10
        assert len(buf2) == 10
        orig = buf.get_all()
        restored = buf2.get_all()
        for o, r in zip(orig, restored):
            assert o.experience_id == r.experience_id
            assert o.result == r.result
            np.testing.assert_array_almost_equal(o.proposed_vector, r.proposed_vector)
    finally:
        os.unlink(path)


def test_thread_safety():
    buf = EnforcementExperienceBuffer(max_size=200)
    errors = []

    def worker():
        try:
            for _ in range(10):
                _record(buf, "approve")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(buf) == 40
