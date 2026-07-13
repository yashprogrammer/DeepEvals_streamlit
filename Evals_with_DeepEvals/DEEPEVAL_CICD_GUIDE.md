# Running DeepEval in CI/CD

A companion guide to the DeepEval notebooks in this folder. Once you have an eval suite that
scores your RAG pipeline, agent, or chatbot, the next step is running it automatically on every
pull request — the same way you'd run unit tests — so a regression in retrieval quality or agent
behavior gets caught before it merges, not after a user reports it.

This guide covers: structuring an eval suite for CI, wiring secrets, a full **GitHub Actions**
walkthrough, and configs for **GitLab CI, CircleCI, Jenkins, and Azure DevOps**.

---

## 1. Why run evals in CI, not just locally

- **Regression gating.** A prompt tweak, a model swap, or a retrieval change can silently drop
  your Faithfulness or Tool Correctness scores. CI catches it on the PR, not in production.
- **A durable, reviewable suite.** An eval file in the repo is code — it gets reviewed, versioned,
  and diffed like anything else, unlike a notebook someone ran once.
- **History over time.** Run identifiers plus (optionally) Confident AI give you a trend line
  instead of a single point-in-time score.

DeepEval ships a CLI built for exactly this: **`deepeval test run`**. It's a thin wrapper around
`pytest` that also manages result collection and (optionally) Confident AI reporting — always
prefer it over calling raw `pytest` on your eval files.

---

## 2. ⚠️ Known issues to pin around (read this before writing YAML)

Two real bugs you will hit if you don't pin correctly, discovered while building this course's
DeepEval demo app:

### `deepeval==4.0.6`'s CLI is broken

`deepeval test run` fails immediately with:

```
ModuleNotFoundError: No module named 'deepeval.deepeval'
```

This is a packaging bug in 4.0.6 (`deepeval/cli/test/command.py` has a bad import). It's fixed in
**4.0.7**. Always pin:

```
deepeval>=4.0.7
```

Never pin bare `deepeval==4.0.6` in a requirements file that runs `deepeval test run` in CI.

### Confident AI sync + a Gemini judge can crash the run

If you use `deepeval.models.GeminiModel` as your judge (as this course does, to avoid requiring an
OpenAI key) **and** set `CONFIDENT_API_KEY`, some 4.0.x releases raise:

```
TypeError: Client.__init__() got an unexpected keyword argument 'model'
```

during the Confident-AI sync step at the end of the run — DeepEval's cloud sync builds a
`google.genai.Client(model=...)` internally, but that client only accepts `model` at the
inference call, not at construction. This fires whenever a Confident key is present alongside a
native Gemini judge.

**Workaround until upstream fixes it:** don't export `CONFIDENT_API_KEY` for the step that runs
`deepeval test run` / `evaluate()`. Keep your pass/fail CI gate running key-free (or with only
`GROQ_API_KEY` / `GEMINI_API_KEY` set), and do any Confident AI syncing — pushing a dataset, or
viewing a report — as a **separate** step/script that doesn't go through that code path, e.g.:

```python
# a separate, standalone step — not the eval run itself
from deepeval.dataset import EvaluationDataset
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file("tests/evals/.dataset.json")
dataset.push(alias="my-eval-dataset")   # only needs CONFIDENT_API_KEY for this call
```

If you've confirmed your installed DeepEval version no longer has this bug, you can set
`CONFIDENT_API_KEY` directly on the test-run step instead.

---

## 3. Structure your repo for CI

```
tests/
  evals/
    metrics.py        # shared metric lists — import these, don't build metrics ad hoc per file
    .dataset.json      # committed goldens (from the synthetic-data notebooks, or hand-written)
    test_rag_app.py
    test_agent.py
ai_app/                # your actual application code
requirements.txt        # pins deepeval>=4.0.7 (see above)
```

Commit the dataset file. CI needs to be reproducible — pulling from Confident AI or regenerating
goldens on every run makes failures nondeterministic and hard to bisect.

### `tests/evals/metrics.py`

```python
import os
from deepeval.models import GeminiModel
from deepeval.metrics import (
    AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric,
    ContextualRelevancyMetric, FaithfulnessMetric, ToolCorrectnessMetric,
    ArgumentCorrectnessMetric, GEval,
)
from deepeval.test_case import SingleTurnParams

judge = GeminiModel(model="gemini-2.5-flash", api_key=os.environ["GEMINI_API_KEY"], temperature=0)

RAG_METRICS = [
    FaithfulnessMetric(model=judge),
    AnswerRelevancyMetric(model=judge),
    ContextualRelevancyMetric(model=judge),
    ContextualPrecisionMetric(model=judge),
    ContextualRecallMetric(model=judge),
]

AGENT_METRICS = [
    ToolCorrectnessMetric(model=judge),
    ArgumentCorrectnessMetric(model=judge),
    GEval(
        name="Task Completion",
        criteria="Did the final answer fully accomplish what the user asked?",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
    ),
]
```

### `tests/evals/test_rag_app.py`

```python
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase

from ai_app.rag import answer_question   # your real RAG pipeline
from tests.evals.metrics import RAG_METRICS

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")


@pytest.mark.parametrize("golden", dataset.goldens)
def test_rag(golden):
    result = answer_question(golden.input)   # {"answer": ..., "retrieved_context": [...]}
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result["answer"],
        expected_output=golden.expected_output,
        retrieval_context=result["retrieved_context"],
    )
    assert_test(test_case=test_case, metrics=RAG_METRICS)
```

Run it locally first:

```bash
export GROQ_API_KEY=... GEMINI_API_KEY=...
deepeval test run tests/evals/test_rag_app.py --identifier "pr-smoke-test" -i -s
```

`deepeval test run` exits non-zero if any test fails — that non-zero exit code is what fails the
CI job and blocks the PR, exactly like a failing unit test would.

---

## 4. Secrets you'll need in CI

| Secret | Required | Used for |
| --- | --- | --- |
| `GROQ_API_KEY` | if your app or judge uses Groq | the model under test |
| `GEMINI_API_KEY` | if using `GeminiModel` as judge | the DeepEval judge |
| `CONFIDENT_API_KEY` | optional | Confident AI sync — **see the caveat above** before wiring this into the main test-run step |

Set these as encrypted secrets in your CI provider's settings — never commit keys, never echo them
in logs. Every example below reads them from the provider's native secrets mechanism.

---

## 5. GitHub Actions — step by step

### Step 1: add the secrets

Repo → **Settings → Secrets and variables → Actions → New repository secret** → add
`GROQ_API_KEY` and `GEMINI_API_KEY` (and `CONFIDENT_API_KEY` if you're using it outside the
test-run step).

### Step 2: add the workflow file

Create `.github/workflows/deepeval.yml`:

```yaml
name: DeepEval

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  evals:
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
      - name: Check out code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"                       # caches ~/.cache/pip between runs

      - name: Install dependencies
        run: pip install -r requirements.txt  # must contain deepeval>=4.0.7

      - name: Run DeepEval suite
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          DEEPEVAL_TELEMETRY_OPT_OUT: "YES"
          # Do NOT set CONFIDENT_API_KEY here — see the known-issue note above.
        run: |
          deepeval test run tests/evals/ \
            --identifier "gh-actions-${{ github.sha }}" \
            --num-processes 2 \
            --ignore-errors \
            --skip-on-missing-params
```

Notes on the choices above:

- **`--num-processes 2`**, not 10 — GitHub-hosted runners plus free-tier Groq/Gemini rate limits
  mean high concurrency causes 429s, not speed. Raise it only if your provider tier supports it.
- **`timeout-minutes: 20`** — LLM calls are slow and occasionally hang; a job-level timeout stops a
  stuck run from blocking the queue.
- **`pull_request` + `push: [main]`** — gate every PR, and also catch anything that slipped through
  on direct pushes to main.
- Branch-protect `main` on this check (**Settings → Branches → Branch protection rules → Require
  status checks to pass**) so a failing eval actually blocks the merge, not just shows a red X.

### Step 3 (optional): a separate job for Confident AI sync

If you want Confident AI's cloud report, run it as its own job/step so the known-issue workaround
holds — the main gating job stays crash-free either way:

```yaml
  sync-to-confident:
    needs: evals
    if: success()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - name: Push dataset to Confident AI
        env:
          CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }}
        run: python scripts/push_dataset_to_confident.py
```

### Step 4: nightly full-suite run (optional but recommended)

PR checks should be fast — run a small, representative slice on every PR. Run your *entire*
dataset on a schedule instead, so slow/expensive coverage doesn't block every commit:

```yaml
on:
  schedule:
    - cron: "0 3 * * *"   # 03:00 UTC daily
  workflow_dispatch: {}    # lets you trigger it manually from the Actions tab
```

---

## 6. GitLab CI

`.gitlab-ci.yml`:

```yaml
deepeval:
  image: python:3.12-slim
  stage: test
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
    - if: '$CI_COMMIT_BRANCH == "main"'
  cache:
    paths:
      - .cache/pip
  variables:
    PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
    DEEPEVAL_TELEMETRY_OPT_OUT: "YES"
  before_script:
    - pip install -r requirements.txt
  script:
    - >
      deepeval test run tests/evals/
      --identifier "gitlab-$CI_PIPELINE_ID"
      --num-processes 2
      --ignore-errors
      --skip-on-missing-params
  # GROQ_API_KEY / GEMINI_API_KEY come from CI/CD Settings → Variables (masked + protected)
```

Add `GROQ_API_KEY` and `GEMINI_API_KEY` under **Settings → CI/CD → Variables**, marked *Masked*
and *Protected* so they never appear in job logs and only apply on protected branches/MRs.

---

## 7. CircleCI

`.circleci/config.yml`:

```yaml
version: 2.1

jobs:
  deepeval:
    docker:
      - image: cimg/python:3.12
    steps:
      - checkout
      - restore_cache:
          keys:
            - pip-{{ checksum "requirements.txt" }}
      - run:
          name: Install dependencies
          command: pip install --user -r requirements.txt
      - save_cache:
          key: pip-{{ checksum "requirements.txt" }}
          paths:
            - ~/.cache/pip
      - run:
          name: Run DeepEval suite
          environment:
            DEEPEVAL_TELEMETRY_OPT_OUT: "YES"
          command: |
            deepeval test run tests/evals/ \
              --identifier "circleci-${CIRCLE_SHA1}" \
              --num-processes 2 \
              --ignore-errors \
              --skip-on-missing-params

workflows:
  test-and-eval:
    jobs:
      - deepeval:
          context: deepeval-secrets   # holds GROQ_API_KEY / GEMINI_API_KEY as a CircleCI context
```

Create the secrets under **Organization Settings → Contexts → deepeval-secrets**, then reference
that context on the job so the keys are injected without living in the YAML.

---

## 8. Jenkins

`Jenkinsfile` (declarative pipeline):

```groovy
pipeline {
    agent { docker { image 'python:3.12-slim' } }

    environment {
        DEEPEVAL_TELEMETRY_OPT_OUT = 'YES'
        GROQ_API_KEY   = credentials('groq-api-key')
        GEMINI_API_KEY = credentials('gemini-api-key')
    }

    stages {
        stage('Install') {
            steps { sh 'pip install -r requirements.txt' }
        }
        stage('Run DeepEval suite') {
            steps {
                sh '''
                  deepeval test run tests/evals/ \
                    --identifier "jenkins-${BUILD_NUMBER}" \
                    --num-processes 2 \
                    --ignore-errors \
                    --skip-on-missing-params
                '''
            }
        }
    }

    post {
        always {
            echo "DeepEval run finished with status: ${currentBuild.currentResult}"
        }
    }
}
```

Register `groq-api-key` and `gemini-api-key` under **Manage Jenkins → Credentials** as *Secret
text*; `credentials(...)` injects them as masked environment variables for the run only.

---

## 9. Azure DevOps

`azure-pipelines.yml`:

```yaml
trigger:
  branches:
    include: [main]

pr:
  branches:
    include: [main]

pool:
  vmImage: "ubuntu-latest"

variables:
  DEEPEVAL_TELEMETRY_OPT_OUT: "YES"

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: "3.12"

  - script: pip install -r requirements.txt
    displayName: "Install dependencies"

  - script: |
      deepeval test run tests/evals/ \
        --identifier "azdo-$(Build.BuildId)" \
        --num-processes 2 \
        --ignore-errors \
        --skip-on-missing-params
    displayName: "Run DeepEval suite"
    env:
      GROQ_API_KEY: $(GROQ_API_KEY)
      GEMINI_API_KEY: $(GEMINI_API_KEY)
```

Add `GROQ_API_KEY` and `GEMINI_API_KEY` as **secret** pipeline variables under **Pipelines → Edit →
Variables** (toggle the lock icon) — Azure DevOps only exposes secret variables to a step when
explicitly mapped under that step's `env:`, as shown above.

---

## 10. Best practices across all providers

- **Fast PR checks, full suite on a schedule.** Run a small, high-signal subset of goldens on
  every PR (seconds to a couple minutes); run the entire dataset nightly or weekly where cost and
  time matter less.
- **Low concurrency, always.** `--num-processes` / provider concurrency should match your LLM
  provider's actual rate limits, not your CI runner's CPU count. Free-tier Groq/Gemini: stay at 2.
- **`--ignore-errors --skip-on-missing-params`** in CI. A single flaky judge call or a golden
  missing an optional field shouldn't fail the whole job — you want the aggregate pass/fail signal,
  not a crash on the first hiccup.
- **Purpose-based `--identifier`s.** Use the commit SHA or build number so failures are traceable
  back to the exact change, especially if you're also pushing to Confident AI.
- **Never commit secrets.** Every example above reads keys from the CI provider's encrypted
  secret store — `.env` files with real keys should never be committed, and `.gitignore` them.
- **Branch-protect on the eval job.** A CI check that runs but doesn't block merges is just a
  suggestion. Require it as a status check on your default branch.
- **Version your dataset with your code.** Commit `.dataset.json` (or pull it from Confident AI
  with a pinned alias/version) so a failing run is reproducible by anyone who checks out that commit.

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'deepeval.deepeval'` | `deepeval==4.0.6`'s broken CLI | pin `deepeval>=4.0.7` |
| `TypeError: Client.__init__() got an unexpected keyword argument 'model'` | `CONFIDENT_API_KEY` set alongside a Gemini judge | don't export `CONFIDENT_API_KEY` on the test-run step; sync separately (§2) |
| CI job succeeds locally but times out in CI | too much concurrency vs. free-tier rate limits | lower `--num-processes`, add a job timeout |
| Every test errors with a provider auth error | secret name typo, or secret not scoped to this branch/PR | check the exact env var name and the secret's branch/protection settings |
| Anonymous telemetry noise in CI logs | DeepEval's default telemetry | set `DEEPEVAL_TELEMETRY_OPT_OUT=YES` in the job env |
| Scores fluctuate run to run for the same input | LLM-judge non-determinism | keep judge `temperature=0`; treat near-threshold scores as "needs a human look," not a hard fail |

---

You now have the same eval suite from the notebooks in this folder running as a required check on
every pull request — the last piece of the loop from "I wrote an eval" to "a regression literally
cannot merge."
