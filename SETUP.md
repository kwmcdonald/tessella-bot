# Getting the bot running (about 45 minutes, once)

The bot is Metaculus's official Fall 2026 template with these additions (details in `ourbot.py`):
- a panel of three frontier models (OpenAI GPT-5.6 Sol, Anthropic Claude Opus 5.5, Google
  Gemini 4 Argon), submitting the median, the approach of the top open-source bot;
- research from AskNews news plus native web search, including prediction-market prices;
- fallbacks so one failure never costs a question, and a stronger model for reading answers;
- honest (not extremized) forecasts, with a per-call cost ceiling;
- an email alarm if a question is about to close unanswered.

## 1. GitHub (done)
1. Sign in at github.com (or create a free account).
2. Create a new **public** repository called `tessella-bot` (done).
   *Public matters: unlimited free GitHub Actions minutes, and open-source bots can earn up to
   2.5x more free credits. Your keys are stored as encrypted secrets and are never visible.*
3. The code is already pushed to https://github.com/kwmcdonald/tessella-bot (done).

## 2. Metaculus bot account (10 min)
1. Create a normal account at metaculus.com.
2. Go to https://www.metaculus.com/futureeval/participate/ and create your bot account.
3. Copy the bot's **API token**.
4. Fill in the participation form (https://forms.gle/aQdYMq9Pisrf1v7d8). Answer **Yes** to the free
   LLM credits question, and mention that the bot is open source (give the repository link).
   Credits arrive later as an OpenRouter key; the Fall 2026 starting amount is about $100.
5. Check that Ramp bank transfers support your country (prizes are paid that way).

## 3. AskNews free news search (5 min, recommended)
Each registered bot maker gets 1,000 free news calls a month.
1. Create an account at https://my.asknews.app using your **bot's email**.
2. Email contact@asknews.app (or message @freqai on the AskNews Discord) with your bot name,
   bot email, your name, your LinkedIn, and affiliation ("independent" is fine), asking for
   FutureEval bot access.
3. When approved, generate an API key. The bot works without this, just with less research.

## 4. A starter OpenRouter key, so we don't wait for the free credits (5 min, $15)
Every day of waiting costs questions, and missed questions score zero.
1. Create an account at openrouter.ai and add $15 of credit.
2. Create an API key with a **credit limit of $15**, so it can never spend more.
3. When the free key from Metaculus arrives, swap it in (step 5).

## 5. Put the keys into GitHub (5 min)
Repository **Settings → Secrets and variables → Actions → New repository secret**:
- `METACULUS_TOKEN`: the bot token from step 2
- `OPENROUTER_API_KEY`: the key from step 4 (later, the free one)
- `ASKNEWS_API_KEY`: from step 3, if you have it

While running on the $15 starter key, also add under the **Variables** tab:
- `REASONING_EFFORT` = `low` (cheaper; delete it once free credits arrive)

## 6. A reliable trigger (10 min)
GitHub's own schedule fires unreliably (another bot measured only ~22% of scheduled runs), and
questions are only open for 3 hours. So an outside service starts the bot every 15 minutes.
1. In GitHub: **Settings (your profile) → Developer settings → Fine-grained tokens → Generate**.
   Repository access: only `tessella-bot`. Permissions: **Actions: Read and write**.
   Expiration: 6 months. Copy the token.
2. Create a free account at https://cron-job.org and add a cron job:
   - URL: `https://api.github.com/repos/kwmcdonald/tessella-bot/actions/workflows/run_bot_on_tournament.yaml/dispatches`
   - Schedule: every 15 minutes
   - Advanced → Request method: **POST**
   - Headers: `Authorization: Bearer YOUR-TOKEN` and `Accept: application/vnd.github+json`
   - Request body: `{"ref":"main"}`
   - Save, then use "Test run": a **204** response means it worked.

## 7. Turn it on (5 min)
1. Open the **Actions** tab and enable workflows.
2. Run **Test Bot** once (Actions → Test Bot → Run workflow). It runs the offline tests, then
   forecasts on Metaculus's unscored practice questions. Check it finishes green.
3. Done. The bot now runs every 15 minutes on its own.

## Afterwards
- GitHub emails you **only** if a question is about to close without a forecast.
- Each run's page shows a status box: open questions, how many we've answered, anything urgent.
- Don't rerun or adjust the bot because of a forecast you dislike, and don't forecast by hand
  for it: the rules require no human in the loop.
- Prizes are paid after **every** question in the season has resolved and checks are done,
  so expect payment months after the season ends.
- If credits run low: set the variable `INCLUDE_MINIBENCH` = `false` to roughly halve spending.

## Settings you can change without touching code (Actions → Variables)
| Variable | Default | What it does |
|---|---|---|
| `FORECASTER_MODELS` | GPT-5.6 Sol, Claude Opus 5.5, Gemini 4 Argon | The panel (OpenRouter names, comma-separated) |
| `BACKUP_MODELS` | GPT-5.6 Terra, Claude Sonnet 5.5, Gemini 3.8 Flash | Tried in order if a panel model fails |
| `SEARCH_MODELS` | GPT-5.6 Terra and Gemini 3.8 Flash with `:online` | Web search, tried in order |
| `PARSER_MODEL` | GPT-5.6 Terra | Reads each written forecast into a number |
| `REASONING_EFFORT` | `medium` | `low` is cheaper, `high` costs more |
| `INCLUDE_MINIBENCH` | `true` | MiniBench's $1,000 two-week rounds |
