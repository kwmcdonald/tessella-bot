"""
Our FutureEval bot: Metaculus's official Fall 2026 template, plus four changes.
Design follows the FutureEval resources page and the Spring 2026 bot-maker advice.

1. A panel of three current frontier models, one each from OpenAI, Anthropic and Google
   (the only providers the donated credits cover), instead of the template's default GPT-4o.
   Each question gets one forecast per model and the median is submitted, as the top
   open-source bot does. Forecasting prompts are the template's own, unchanged.
2. Independent research per panel member (our exploratory twist). The top open-source bot
   found its worst misses came when all models agreed on one shared briefing. So only the
   AskNews digest is shared; each model also gets its OWN web search, run through its own
   lab's native search, so the members' evidence, and hopefully their errors, differ.
   Searches ask for prediction-market prices (allowed by the rules) and the latest value of
   any number the question depends on. INDEPENDENT_RESEARCH=false restores one shared briefing.
3. Fallbacks everywhere, because a skipped question scores zero:
     - if a model call fails, the same forecast is retried on backup models;
     - if research fails, we fall back to other search models, then to no research;
     - only 1 of the 3 panel forecasts has to succeed for the question to be submitted;
     - a stronger model turns replies into numbers (parse failures were winners' top loss).
4. Honest confidence: no extremizing. Our simulation showed that boldness beyond
   honest confidence lowers the chance of a payout, and several bot-makers reported that
   extremizing hurt. A per-call output cap keeps any one reasoning call from running away.
5. An alarm: if an open question is close to its deadline and still has no forecast from
   us, the run exits with an error, which makes GitHub email the account owner.

Run modes (same as the template):
    python ourbot.py --mode test_questions   # smoke test on Metaculus's bot-testing area
    python ourbot.py --mode tournament       # live: Fall 2026 FutureEval + MiniBench
    python ourbot.py --mode tournament --dry-run   # forecast but do not submit
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    AskNewsSearcher,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
)

from main import FallTemplateBot2026  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger("ourbot")


def _env(name: str, default: str) -> str:
    """Environment setting; unset OR empty (GitHub passes "" for unset variables) means default."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_list(name: str, default: str) -> list[str]:
    return [m.strip() for m in _env(name, default).split(",") if m.strip()]


# One frontier model per lab covered by the donated credits. Override via FORECASTER_MODELS.
FORECASTER_MODELS = _env_list(
    "FORECASTER_MODELS",
    "openrouter/openai/gpt-5.6-sol,"
    "openrouter/anthropic/claude-opus-5.5,"
    "openrouter/google/gemini-4-argon",
)
# Cheaper models tried, in order, if a panel model fails.
BACKUP_MODELS = _env_list(
    "BACKUP_MODELS",
    "openrouter/openai/gpt-5.6-terra,"
    "openrouter/anthropic/claude-sonnet-5.5,"
    "openrouter/google/gemini-3.8-flash",
)
# Turning a model's written answer into a number. Parse failures were bot-makers' biggest
# source of lost forecasts, so this is a mid-tier model rather than the cheapest one.
PARSER_MODEL = _env("PARSER_MODEL", "openrouter/openai/gpt-5.6-terra")
# Native web search through OpenRouter (":online"), tried in order. Covered by donated credits.
SEARCH_MODELS = _env_list(
    "SEARCH_MODELS",
    "openrouter/openai/gpt-5.6-terra:online,"
    "openrouter/google/gemini-3.8-flash:online",
)
SEARCH_MODEL = SEARCH_MODELS[0]
MAX_OUTPUT_TOKENS = int(_env("MAX_OUTPUT_TOKENS", "12000"))  # per-call cost ceiling
# Independent research: one search model per panel member, in panel order (same lab as the forecaster).
INDEPENDENT_RESEARCH = _env("INDEPENDENT_RESEARCH", "true").lower() not in ("false", "0", "no")
SLOT_SEARCH_MODELS = _env_list(
    "SLOT_SEARCH_MODELS",
    "openrouter/openai/gpt-5.6-terra:online,"
    "openrouter/anthropic/claude-sonnet-5.5:online,"
    "openrouter/google/gemini-3.8-flash:online",
)
REASONING_EFFORT = _env("REASONING_EFFORT", "medium")
URGENT_MINUTES = int(_env("URGENT_MINUTES", "90"))
# MiniBench rounds are cheap lottery tickets; switch off with INCLUDE_MINIBENCH=false to save credits.
INCLUDE_MINIBENCH = _env("INCLUDE_MINIBENCH", "true").lower() not in ("false", "0", "no")

# Which panel member the current forecast should use (set per forecast task).
_panel_slot: contextvars.ContextVar[int | None] = contextvars.ContextVar("panel_slot", default=None)


def make_llm(model: str, **extra) -> GeneralLlm:
    kwargs = dict(model=model, timeout=300, allowed_tries=2, max_tokens=MAX_OUTPUT_TOKENS)
    if REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    kwargs.update(extra)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return GeneralLlm(**kwargs)


RESEARCH_ADDENDUM = """
Also include, where they exist:
- current prices or forecasts on prediction markets and forecasting sites (Polymarket, Kalshi,
  Manifold, Metaculus) for this question or closely related ones, with the date seen;
- the most recent value of any number, count or official figure the question depends on,
  and its source and date;
- anything scheduled before the resolution date that could decide the question.
Remember the question is forward-looking: if something must happen "before" a date, it has
not happened yet unless you find clear evidence that it already has."""


class FallbackLlm:
    """Looks like a GeneralLlm to the template (it only needs `.model` and `.invoke`), but tries
    a chain of models until one answers. Records which model actually produced each answer."""

    def __init__(self, models: list[str]):
        if not models:
            raise ValueError("FallbackLlm needs at least one model")
        self.models = models
        self.model = models[0]
        self.used: list[str] = []
        self._llms = [make_llm(m) for m in models]

    async def invoke(self, prompt, *args, **kwargs) -> str:
        last_error: Exception | None = None
        for model, llm in zip(self.models, self._llms):
            try:
                text = await llm.invoke(prompt, *args, **kwargs)
                if not text or not str(text).strip():
                    raise RuntimeError(f"{model} returned an empty response")
                self.used.append(model)
                return text
            except Exception as error:  # noqa: BLE001 - we deliberately try the next model
                last_error = error
                logger.warning(f"Model {model} failed ({type(error).__name__}: {error}); trying next model")
        raise RuntimeError(f"All models failed. Last error: {last_error}")


class OurBot(FallTemplateBot2026):
    """Template bot with a four-model panel, fallbacks, and research redundancy."""

    _slot_counters: dict[str, int] = {}

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("llms", {
            "default": make_llm(FORECASTER_MODELS[0]),
            "summarizer": make_llm(PARSER_MODEL),
            "parser": make_llm(PARSER_MODEL),
            "researcher": "asknews/news-summaries" if _has_asknews() else make_llm(SEARCH_MODEL),
        })
        super().__init__(*args, **kwargs)
        self.panels = [FallbackLlm([m] + [b for b in BACKUP_MODELS if b != m]) for m in FORECASTER_MODELS]
        self.research_log: list[tuple[str, int, str]] = []  # (question, panel slot, search model used)

    # -- model panel ---------------------------------------------------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        key = str(question.id_of_question or question.page_url)
        slot = self._slot_counters.get(key, 0)
        self._slot_counters[key] = slot + 1
        slot %= len(self.panels)
        token = _panel_slot.set(slot)
        try:
            if INDEPENDENT_RESEARCH:
                research = research + "\n\n" + await self._own_search(question, slot)
            return await super()._make_prediction(question, research)
        finally:
            _panel_slot.reset(token)

    async def _own_search(self, question: MetaculusQuestion, slot: int) -> str:
        """This panel member's private web search. Falls back to the other members' search
        models only if its own fails (then evidence is less independent, but never missing)."""
        own = SLOT_SEARCH_MODELS[slot % len(SLOT_SEARCH_MODELS)]
        chain = [own] + [m for m in SLOT_SEARCH_MODELS if m != own]
        searcher = FallbackLlm(chain)
        prompt = self._get_research_prompt(question, own) + RESEARCH_ADDENDUM
        try:
            web = await searcher.invoke(prompt)
            self.research_log.append((question.page_url, slot, searcher.used[-1]))
            return "## Your own web search and prediction markets\n" + web
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Own search failed for slot {slot} on {question.page_url}: {error}")
            return "## Your own web search\nNo web search results could be retrieved."

    def get_llm(self, purpose: str = "default", guarantee_type=None):
        slot = _panel_slot.get()
        if purpose == "default" and slot is not None and guarantee_type in (None, "llm"):
            return self.panels[slot]
        return super().get_llm(purpose, guarantee_type)

    # -- research: AskNews + web search, each with fallbacks -----------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            sections = []
            if _has_asknews():
                try:
                    news = await AskNewsSearcher().call_preconfigured_version(
                        "asknews/news-summaries", question.question_text)
                    if news and news.strip():
                        sections.append("## Recent news (AskNews)\n" + news)
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"AskNews failed for {question.page_url}: {error}")
            if not INDEPENDENT_RESEARCH:
                prompt = self._get_research_prompt(question, SEARCH_MODEL) + RESEARCH_ADDENDUM
                try:
                    web = await FallbackLlm(SEARCH_MODELS).invoke(prompt)
                    sections.append("## Web search and prediction markets\n" + web)
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"Web search failed for {question.page_url}: {error}")
            elif not sections:
                return "## Shared news\nNo shared news digest for this question; see your own search below."
            if sections:
                logger.info(f"Research for {question.page_url}: {len(sections)} source(s)")
                return "\n\n".join(sections)
            logger.warning(f"No research available for {question.page_url}; forecasting without it")
            return "No research could be retrieved for this question. Rely on base rates and the question text."


def _has_asknews() -> bool:
    return bool((os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")) or os.getenv("ASKNEWS_API_KEY"))


def find_urgent_misses(questions: list[MetaculusQuestion], now: datetime | None = None,
                       minutes: int = URGENT_MINUTES) -> list[MetaculusQuestion]:
    """Open questions we still haven't forecast that close within `minutes`."""
    now = now or datetime.now(timezone.utc)
    urgent = []
    for q in questions:
        if q.already_forecasted:
            continue
        close = q.close_time
        if close is None:
            continue
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        if now <= close <= now + timedelta(minutes=minutes):
            urgent.append(q)
    return urgent


def build_bot(publish: bool) -> OurBot:
    return OurBot(
        research_reports_per_question=1,
        predictions_per_research_report=len(FORECASTER_MODELS),
        required_successful_predictions=1 / len(FORECASTER_MODELS),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run our forecasting bot")
    parser.add_argument("--mode", choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument("--dry-run", action="store_true", help="Forecast but do not submit to Metaculus")
    args = parser.parse_args()

    check_environment(strict=True)
    publish = not args.dry_run
    print_startup_banner(args.mode, will_publish=publish)
    bot = build_bot(publish)
    client = MetaculusClient()

    if args.mode == "tournament":
        targets = [client.CURRENT_AI_COMPETITION_ID] + ([client.CURRENT_MINIBENCH_ID] if INCLUDE_MINIBENCH else [])
        url = "https://www.metaculus.com/tournament/fall-futureeval-2026/"
    else:
        bot.skip_previously_forecasted_questions = False
        targets = ["bot-testing-area"]
        url = "https://www.metaculus.com/tournament/bot-testing-area/"

    reports = []
    for target in targets:
        reports += asyncio.run(bot.forecast_on_tournament(target, return_exceptions=True))
    bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=url)

    used = [m for panel in bot.panels for m in panel.used]
    if used:
        from collections import Counter
        logger.info(f"Models that produced forecasts this run: {dict(Counter(used))}")

    # Alarm: anything still unforecast and about to close?
    if args.mode == "tournament" and publish:
        still_open = []
        for target in targets:
            try:
                still_open += client.get_all_open_questions_from_tournament(target)
            except Exception as error:  # noqa: BLE001
                logger.error(f"Could not re-check open questions for {target}: {error}")
        urgent = find_urgent_misses(still_open)
        _write_status(still_open, urgent)
        if urgent:
            for q in urgent:
                logger.error(f"URGENT: no forecast yet and closing soon: {q.page_url} (closes {q.close_time})")
            return 1
    return 0


def _write_status(open_questions: list[MetaculusQuestion], urgent: list[MetaculusQuestion]) -> None:
    """One-glance status for the GitHub Actions run page."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    done = sum(1 for q in open_questions if q.already_forecasted)
    lines = [
        "## Forecasting bot status",
        f"- Open questions right now: **{len(open_questions)}**",
        f"- Already forecast by us: **{done}**",
        f"- Still waiting: **{len(open_questions) - done}**",
        f"- Urgent (closing within {URGENT_MINUTES} min, no forecast): **{len(urgent)}**",
    ]
    lines += [f"  - {q.page_url}" for q in urgent]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
