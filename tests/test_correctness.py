"""pytest view of the same checks the standalone harness runs."""
import pytest
from sparseattn.harness import all_checks

CHECKS = all_checks("cpu")


@pytest.mark.parametrize("check", CHECKS, ids=[c.name for c in CHECKS])
def test_check(check):
    r = check.run()
    assert r.ok, f"{check.name}: {r.detail}"
