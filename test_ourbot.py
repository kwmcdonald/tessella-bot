"""Offline tests for ourbot.py. No network, no API keys: every model call is faked."""
import asyncio
import os
import re
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("METACULUS_TOKEN", "test-token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.pop("ASKNEWS_CLIENT_ID", None)
os.environ.pop("ASKNEWS_SECRET", None)

import forecasting_tools  # noqa: E402
from forecasting_tools import BinaryQuestion, GeneralLlm  # noqa: E402
from forecasting_tools.data_models.binary_report import BinaryPrediction  # noqa: E402

import main  # noqa: E402
import ourbot  # noqa: E402

# Each panel model answers with its own distinctive probability so we can see who contributed.
PANEL_ANSWERS = {
    "openrouter/openai/gpt-5.6-sol": 0.20,
    "openrouter/anthropic/claude-opus-5.5": 0.30,
    "openrouter/google/gemini-4-argon": 0.90,
    "openrouter/openai/gpt-5.6-terra": 0.25,
    "openrouter/anthropic/claude-sonnet-5.5": 0.35,
    "openrouter/google/gemini-3.8-flash": 0.45,
}
CALLS: list[str] = []
FAILING: set[str] = set()


async def fake_invoke(self, prompt, *args, **kwargs):
    CALLS.append(self.model)
    if self.model in FAILING:
        raise RuntimeError(f"simulated outage for {self.model}")
    if "You do not produce forecasts yourself" in str(prompt):
        return f"Research notes from {self.model}: nothing has changed recently."
    if self.model in PANEL_ANSWERS:
        return f"(a)... (b)... reasoning by {self.model}\nProbability: {round(PANEL_ANSWERS[self.model]*100)}%"
    return f"summary from {self.model}"


async def fake_structure_output(text, output_type, **kwargs):
    assert output_type is BinaryPrediction
    m = re.search(r"Probability:\s*(\d+)%", text)
    return BinaryPrediction(prediction_in_decimal=int(m.group(1)) / 100)


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    CALLS.clear()
    FAILING.clear()
    monkeypatch.setattr(GeneralLlm, "invoke", fake_invoke)
    monkeypatch.setattr(main, "structure_output", fake_structure_output)
    ourbot.OurBot._slot_counters = {}
    yield


def make_question(qid=101, close_in_minutes=120, forecasted=False):
    now = datetime.now(timezone.utc)
    return BinaryQuestion(
        question_text=f"Will test event {qid} happen by 2026-12-31?",
        id_of_post=qid, id_of_question=qid,
        page_url=f"https://www.metaculus.com/questions/{qid}/",
        resolution_criteria="Resolves Yes if the event happens.", fine_print="", background_info="Background.",
        close_time=now + timedelta(minutes=close_in_minutes), open_time=now - timedelta(hours=1),
        already_forecasted=forecasted,
    )


def run(bot, questions):
    return asyncio.run(bot.forecast_questions(questions, return_exceptions=True))


def test_panel_uses_four_different_models_and_submits_median():
    bot = ourbot.build_bot(publish=False)
    [report] = run(bot, [make_question()])
    assert not isinstance(report, BaseException), report
    forecasters = [c for c in CALLS if c in list(PANEL_ANSWERS)[:3]]
    assert sorted(forecasters) == sorted(ourbot.FORECASTER_MODELS)        # each panel model exactly once
    assert report.prediction == pytest.approx(0.30)                       # median of .2 .3 .9


def test_failed_model_is_replaced_by_backup():
    FAILING.add("openrouter/anthropic/claude-opus-5.5")
    bot = ourbot.build_bot(publish=False)
    [report] = run(bot, [make_question()])
    assert not isinstance(report, BaseException), report
    assert "openrouter/openai/gpt-5.6-terra" in CALLS          # first backup stepped in
    assert report.prediction == pytest.approx(0.25)             # median of .2 .25 .9


def test_question_still_submitted_when_only_one_model_survives():
    FAILING.update(m for m in PANEL_ANSWERS if m != "openrouter/google/gemini-4-argon")
    bot = ourbot.build_bot(publish=False)
    [report] = run(bot, [make_question()])
    assert not isinstance(report, BaseException), report
    assert report.prediction == pytest.approx(0.90)


def test_total_outage_reports_error_instead_of_crashing_run():
    FAILING.update(PANEL_ANSWERS)
    bot = ourbot.build_bot(publish=False)
    reports = run(bot, [make_question(1), make_question(2)])
    assert all(isinstance(r, BaseException) for r in reports)   # surfaced, run continues


def _spy(monkeypatch):
    """Record (model, prompt) for every call while still answering like fake_invoke."""
    seen = []
    async def spy_invoke(self, prompt, *a, **k):
        seen.append((self.model, str(prompt)))
        return await fake_invoke(self, prompt)
    monkeypatch.setattr(GeneralLlm, "invoke", spy_invoke)
    return seen


def test_each_panel_member_gets_its_own_search():
    bot = ourbot.build_bot(publish=False)
    [report] = run(bot, [make_question()])
    assert not isinstance(report, BaseException), report
    used = sorted((slot, model) for _, slot, model in bot.research_log)
    assert used == [(i, ourbot.SLOT_SEARCH_MODELS[i]) for i in range(3)]


def test_members_see_different_evidence(monkeypatch):
    seen = _spy(monkeypatch)
    bot = ourbot.build_bot(publish=False)
    run(bot, [make_question()])
    for forecaster, searcher in zip(ourbot.FORECASTER_MODELS, ourbot.SLOT_SEARCH_MODELS):
        prompts = [p for m, p in seen if m == forecaster]
        assert len(prompts) == 1
        assert f"Research notes from {searcher}" in prompts[0]                     # its own search
        others = [s for s in ourbot.SLOT_SEARCH_MODELS if s != searcher]
        assert not any(f"Research notes from {o}" in prompts[0] for o in others)  # not the others'


def test_search_prompts_ask_for_markets_and_latest_figures(monkeypatch):
    seen = _spy(monkeypatch)
    run(ourbot.build_bot(publish=False), [make_question()])
    searches = [p for m, p in seen if m in ourbot.SLOT_SEARCH_MODELS]
    assert len(searches) == 3 and all("Polymarket" in p and "most recent value" in p for p in searches)


def test_own_search_falls_back_to_another_members_search():
    FAILING.add(ourbot.SLOT_SEARCH_MODELS[1])
    bot = ourbot.build_bot(publish=False)
    [report] = run(bot, [make_question()])
    assert not isinstance(report, BaseException), report
    slot1 = [model for _, slot, model in bot.research_log if slot == 1]
    assert slot1 and slot1[0] != ourbot.SLOT_SEARCH_MODELS[1]


def test_asknews_digest_is_shared_with_every_member(monkeypatch):
    monkeypatch.setenv("ASKNEWS_API_KEY", "test")
    async def fake_news(self, preset, query):
        return "AskNews: three relevant articles."
    monkeypatch.setattr(forecasting_tools.AskNewsSearcher, "call_preconfigured_version", fake_news)
    seen = _spy(monkeypatch)
    run(ourbot.build_bot(publish=False), [make_question()])
    for forecaster in ourbot.FORECASTER_MODELS:
        [prompt] = [p for m, p in seen if m == forecaster]
        assert "AskNews: three relevant articles." in prompt


def test_research_never_blocks_forecast():
    FAILING.update(ourbot.SLOT_SEARCH_MODELS)
    bot = ourbot.build_bot(publish=False)
    [report] = run(bot, [make_question()])
    assert not isinstance(report, BaseException), report


def test_many_questions_each_get_full_panel():
    bot = ourbot.build_bot(publish=False)
    reports = run(bot, [make_question(i) for i in range(1, 6)])
    assert all(not isinstance(r, BaseException) for r in reports)
    assert all(r.prediction == pytest.approx(0.30) for r in reports)


def test_already_forecast_questions_are_skipped():
    bot = ourbot.build_bot(publish=False)
    reports = run(bot, [make_question(1, forecasted=True), make_question(2)])
    assert len(reports) == 1


def test_urgent_alarm_logic():
    now = datetime.now(timezone.utc)
    qs = [make_question(1, close_in_minutes=30),                     # urgent
          make_question(2, close_in_minutes=30, forecasted=True),    # done
          make_question(3, close_in_minutes=600),                    # not urgent yet
          make_question(4, close_in_minutes=-5)]                     # already closed
    urgent = ourbot.find_urgent_misses(qs, now=now, minutes=90)
    assert [q.id_of_question for q in urgent] == [1]


def test_empty_github_variables_fall_back_to_defaults(monkeypatch):
    import importlib
    for name in ("FORECASTER_MODELS", "BACKUP_MODELS", "REASONING_EFFORT", "INCLUDE_MINIBENCH"):
        monkeypatch.setenv(name, "")
    mod = importlib.reload(ourbot)
    assert len(mod.FORECASTER_MODELS) == 3 and mod.BACKUP_MODELS and mod.REASONING_EFFORT == "medium"
    assert mod.INCLUDE_MINIBENCH is True
    monkeypatch.setenv("INCLUDE_MINIBENCH", "false")
    assert importlib.reload(ourbot).INCLUDE_MINIBENCH is False
    monkeypatch.delenv("INCLUDE_MINIBENCH")
    importlib.reload(ourbot)


def test_only_donated_credit_providers_by_default():
    for m in (ourbot.FORECASTER_MODELS + ourbot.BACKUP_MODELS + ourbot.SEARCH_MODELS
              + ourbot.SLOT_SEARCH_MODELS + [ourbot.PARSER_MODEL]):
        provider = m.split("/")[1]
        assert provider in {"openai", "anthropic", "google"}, m


def test_shared_briefing_mode_still_works(monkeypatch):
    import importlib
    monkeypatch.setenv("INDEPENDENT_RESEARCH", "false")
    mod = importlib.reload(ourbot)
    try:
        monkeypatch.setattr(main, "structure_output", fake_structure_output)
        bot = mod.build_bot(publish=False)
        research = asyncio.run(bot.run_research(make_question()))
        assert f"Research notes from {mod.SEARCH_MODELS[0]}" in research
        [report] = asyncio.run(bot.forecast_questions([make_question()], return_exceptions=True))
        assert not isinstance(report, BaseException) and not bot.research_log
    finally:
        monkeypatch.delenv("INDEPENDENT_RESEARCH")
        importlib.reload(ourbot)
