from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.matching.engine import (
    _age_score,
    _build_reasons,
    _hard_compatible,
    _jaccard_score,
    rank_matches,
    score_pair,
)
from app.schemas.user import UserResponse
from tests.conftest import make_test_user


def make_profile(**kwargs) -> UserResponse:
    data = make_test_user(name=kwargs.pop("name", "test"), **kwargs)
    return UserResponse(
        id="abc123",
        agent_summary="Test agent",
        agent_generated=False,
        created_at=datetime.now(timezone.utc),
        **data.model_dump(),
    )


class TestJaccardScore:
    def test_empty_lists(self):
        assert _jaccard_score([], []) == 0.0

    def test_identical(self):
        assert _jaccard_score(["a", "b"], ["a", "b"]) == 1.0

    def test_no_overlap(self):
        assert _jaccard_score(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        score = _jaccard_score(["a", "b", "c"], ["b", "c", "d"])
        assert 0.49 < score < 0.51  # 2/4 = 0.5

    def test_case_insensitive(self):
        assert _jaccard_score(["Hiking"], ["hiking"]) == 1.0


class TestAgeScore:
    def test_same_age(self):
        a = make_profile(name="a", age=30)
        b = make_profile(name="b", age=30)
        assert _age_score(a, b) == 1.0

    def test_max_gap(self):
        a = make_profile(name="a", age=20)
        b = make_profile(name="b", age=60)
        assert _age_score(a, b) == 0.0

    def test_moderate_gap(self):
        a = make_profile(name="a", age=25)
        b = make_profile(name="b", age=35)
        assert _age_score(a, b) == 0.5


class TestHardCompatible:
    def test_different_intent_incompatible(self):
        a = make_profile(name="a", intent="companion")
        b = make_profile(name="b", intent="partner")
        assert not _hard_compatible(a, b)

    def test_age_out_of_range_incompatible(self):
        a = make_profile(name="a", age=50, preferred_age_min=20, preferred_age_max=30)
        b = make_profile(name="b", age=50, preferred_age_min=20, preferred_age_max=30)
        assert not _hard_compatible(a, b)

    def test_age_within_range_compatible(self):
        a = make_profile(name="a", age=28, preferred_age_min=24, preferred_age_max=35)
        b = make_profile(name="b", age=30, preferred_age_min=20, preferred_age_max=35)
        assert _hard_compatible(a, b)

    def test_different_city_no_remote_incompatible(self):
        a = make_profile(name="a", city="Shanghai", accept_remote=False)
        b = make_profile(name="b", city="Beijing", accept_remote=False)
        assert not _hard_compatible(a, b)

    def test_different_city_with_remote_compatible(self):
        a = make_profile(name="a", city="Shanghai", accept_remote=True)
        b = make_profile(name="b", city="Beijing", accept_remote=False)
        assert _hard_compatible(a, b)

    def test_mutual_compatible(self):
        a = make_profile(name="a")
        b = make_profile(name="b")
        assert _hard_compatible(a, b)


class TestBuildReasons:
    def test_shared_hobbies_included(self):
        a = make_profile(name="a", hobbies=["hiking", "coding"])
        b = make_profile(name="b", hobbies=["hiking", "music"])
        reasons = _build_reasons(a, b)
        assert any("hiking" in r for r in reasons)

    def test_same_city_included(self):
        a = make_profile(name="a", city="Shanghai")
        b = make_profile(name="b", city="Shanghai")
        reasons = _build_reasons(a, b)
        assert any("Shanghai" in r for r in reasons)

    def test_fallback_reason(self):
        a = make_profile(name="a", city="Shanghai", accept_remote=False,
                         communication_style="direct",
                         hobbies=["hiking"], values=["honesty"], availability=["weeknight"])
        b = make_profile(name="b", city="Beijing", accept_remote=False,
                         communication_style="warm",
                         hobbies=["coding"], values=["kindness"], availability=["weekend morning"])
        reasons = _build_reasons(a, b)
        assert "General" in reasons[0]


class TestScorePair:
    def test_incompatible_returns_none(self):
        a = make_profile(name="a", intent="companion")
        b = make_profile(name="b", intent="partner")
        assert score_pair(a, b) is None

    def test_compatible_returns_score_and_reasons(self):
        a = make_profile(name="a")
        b = make_profile(name="b")
        result = score_pair(a, b)
        assert result is not None
        score, reasons = result
        assert 0.0 <= score <= 1.0
        assert len(reasons) >= 1


class TestRankMatches:
    def test_returns_top_n(self):
        user = make_profile(name="target")
        others = [make_profile(name=f"c{i}") for i in range(20)]
        results = rank_matches(user, others, top_n=5)
        assert len(results) == 5

    def test_sorted_by_score_descending(self):
        user = make_profile(name="target")
        others = [make_profile(name=f"c{i}") for i in range(10)]
        results = rank_matches(user, others, top_n=10)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_filters_incompatible(self):
        user = make_profile(name="target", intent="companion")
        others = [make_profile(name="partner", intent="partner")]
        assert len(rank_matches(user, others)) == 0
