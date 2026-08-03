# Autobench Su An Ne Yapiyor?

Bu dosya Autobench'i hic bilmeyen birine anlatmak icin yazildi. Amac kisa bir
README yazmak degil; Autobench'in bugunku halinin hangi problemi cozdunu,
hangi parcalardan olustugunu, hangi API'lerin nasil kullanildigini ve bir
uygulama benchmark'inin nasil Autobench'e tasinacagini tek dosyada, bol kod
ornegiyle gostermek.

Autobench su anda bir "benchmark script generator" degil. Daha dogru tanim su:

> Autobench, uygulama veya AI sistemi calistirirken ortaya cikan benchmark
> kanitlarini semantic, replayable ve export edilebilir bir deney kaydina
> donusturen YAML-first bir evidence framework'tur.

Bu cumledeki kelimeler onemli:

- **YAML-first:** Benchmark tanimi oncelikle insan tarafindan okunabilir bir
  YAML spec olarak dusunulur.
- **Evidence:** Sadece "score 0.82" yazmaz; case, variant, factor, metric,
  span, artifact, error, asset version ve environment bilgisini beraber saklar.
- **Semantic:** Metrikler sadece `score`, `cost`, `tokens` gibi rastgele isimler
  degildir. `quality.score`, `money.cost`, `llm.tokens.input`,
  `time.latency`, `agent.tool.argument.correctness` gibi anlam siniflarina
  baglanabilir.
- **Replayable:** Benchmark calistiktan sonra task'i tekrar import etmeden ve
  modele tekrar istek atmadan rapor, compare ve export uretilebilir.
- **General-purpose:** AI/agent benchmark'lari birinci sinif use case, ama
  framework sadece LLM eval icin yazilmadi. Latency, throughput, correctness,
  cost, domain KPI, tool kalitesi, policy compliance gibi her tur sistemi
  olcebilir.

Su an v0.1 cizgisinde olan seylerin ozeti:

- YAML benchmark spec okuma ve dogrulama
- Dataset, case, case defaults, attachment ve file-backed dataset destegi
- Case x variant matrisini deterministik kosma
- Python task runtime: sync ve async task destegi
- `RunContext` ile metric, factor, event, diagnostic, check, outcome,
  artifact ve span toplama
- Span duration'i framework tarafindan hesaplama
- Manual measurement helper: warmup, repetition, max_seconds, median, p95,
  noise vb.
- Scoring: output, pass/fail, exact, schema, python callback, expected action
- Semantic registry ve semantic alias/parent sistemi
- Token cost derivation: token + provider + model -> `money.cost`
- Paired baseline post-derivation: baseline/candidate run'larini case veya
  factor bazinda eslestirip speedup/delta uretme
- Policy checks: `must_greater_equal`, `must_less_equal`, `must_between`, vb.
- Immutable YAML RunRecord ve ExperimentRecord yazma
- Replay: task import etmeden recorded evidence'dan `ExperimentResult` kurma
- Rich CLI tablolar: validate, run, replay, report, export, compare
- Markdown/YAML/CSV export
- Asset tracking: prompt, tool, Pydantic model, dataclass, typed class,
  field/param schema, version hash, source hash, asset history
- Pydantic Evals payload adapter
- Pydantic AI usage bridge
- Trace envelope import
- Agentic metric primitives: agent/tool span'lari, expected tool action scoring
- Production sample -> case ve generated/synthetic case metadata helper'lari

Su an bilincli olarak v0.1 yuzeyinden cikarilmis seyler:

- Chart/image export
- Dashboard
- Hosted platform
- Full OTel bridge
- Full autoptimize implementation
- Causal attribution iddiasi

Autobench bugun chart engine degil. Rich terminal tablolar, Markdown,
YAML ve CSV export var; chart/image artifact tasarimi v1 sonrasi icin
bekletiliyor.

---

## 1. Neden Autobench Var?

Bir benchmark scripti genelde boyle baslar:

```python
# run_benchmark.py
for scenario in scenarios:
    for model in models:
        started = perf_counter()
        result = run_my_app(scenario, model)
        elapsed = perf_counter() - started
        score = judge(result)
        rows.append(
            {
                "scenario": scenario.name,
                "model": model,
                "score": score,
                "latency": elapsed,
                "tokens": result.usage.total_tokens,
            }
        )

write_json(rows, "summary.json")
print_table(rows)
```

Bu pratikte calisir ama hizla dagilir:

- Case'ler nerede?
- Hangi variant hangi factor'leri degistirdi?
- Metriklerin semantic anlami ne?
- Token input/output ayri mi tutuldu?
- Cost nasil hesaplandi?
- Hangi prompt veya tool version'i kullanildi?
- Bir run tekrar raporlanabilir mi?
- Task import etmeden replay yapilabilir mi?
- Baseline/candidate farki case bazinda mi aggregate bazinda mi?
- Score mu metric mi, objective mi diagnostic mi?
- Bir policy fail olursa run status nasil etkilenir?
- Degisen tool/prompt/schema benchmark sonucunu nasil etkiledi?

Autobench bu sorulari framework primitive'lerine ayirir:

```text
BenchmarkSpec
  -> Dataset / Case
  -> Variant / Factor
  -> Task
  -> RunContext
  -> Span / Observation / Artifact
  -> Scorer
  -> Deriver / PostDeriver / Policy
  -> RunRecord
  -> ExperimentRecord
  -> Replay / Report / Export / Compare
```

Yani benchmark yazarken tekrar tekrar ayni runner iskeletini yazmak yerine,
deneyi tanimlar ve framework'un evidence pipeline'ina teslim edersin.

---

## 2. En Kucuk Mental Model

Autobench'i anlamak icin su bes kavram yetiyor:

### Case

Bir test girdisi. Ornegin bir destek talebi:

```python
from autobench import Case

case = Case(
    id="refund_001",
    input={
        "subject": "Duplicate charge",
        "body": "I was billed twice yesterday.",
    },
    expected={
        "queue": "billing",
        "priority": "high",
    },
    tags=["billing", "refund"],
)
```

### Variant

Ayni case'i farkli konfigurasyonlarla kosmak icin kullanilir. Ornegin prompt
version veya model degisikligi:

```python
from autobench import FactorValue, Semantic, Variant

variant = Variant(
    id="route_v2",
    label="new routing prompt",
    factors=[
        FactorValue(
            name="prompt_version",
            value="route-v2",
            semantic_type=Semantic.PROMPT_VERSION,
            optimize=True,
        ),
        FactorValue(
            name="model",
            value="openrouter:openai/gpt-5.6-luna",
            semantic_type=Semantic.LLM_MODEL_NAME,
            optimize=True,
        ),
    ],
)
```

### Task

Autobench'in calistirdigi uygulama fonksiyonu. Signature kuralidir:

```python
def run_case(ctx, case):
    ...
```

`ctx` her zaman birinci parametre, `case` her zaman ikinci parametredir.

### Observation

Benchmark sirasinda toplanan her anlamli veri parcasidir:

- metric: `quality.score`, `time.latency`, `money.cost`
- factor: `llm.model.name`, `prompt.version`
- event: diagnostic, error, skip reason
- artifact reference: trace, raw response, generated file

### RunRecord

Bir case x variant kosusunun immutable kaydidir. Task output, observations,
scores, spans, artifacts, errors, factors ve asset versions bu kayda girer.

---

## 3. Kurulum Ve CLI

Repository icinde gelistirme kurulumu:

```bash
uv sync --extra dev
```

Autobench CLI komutlari:

```bash
uv run autobench --help
```

Mevcut komutlar:

```text
validate  Validate a YAML Autobench spec.
run       Run a YAML Autobench spec.
replay    Replay recorded YAML evidence without importing benchmark tasks.
report    Render the Rich terminal report from recorded evidence.
export    Export recorded evidence to a file and show a Rich terminal preview.
compare   Compare two recorded variants without claiming causality.
```

Tipik workflow:

```bash
uv run autobench validate autobench.yaml
uv run autobench run autobench.yaml --record runs/support-routing
uv run autobench replay runs/support-routing
uv run autobench report runs/support-routing
uv run autobench compare runs/support-routing --baseline route_v1 --candidate route_v2
uv run autobench export runs/support-routing --format yaml --path runs/support-routing/report.yaml
uv run autobench export runs/support-routing --format markdown --path runs/support-routing/report.md
uv run autobench export runs/support-routing --format csv --path runs/support-routing/runs.csv
```

`run` icin concurrency:

```bash
uv run autobench run autobench.yaml --record runs/demo --concurrency 4
```

Record istemiyorsan:

```bash
uv run autobench run autobench.yaml --no-record
```

Export formatlari su anda:

```text
csv
markdown
yaml
```

Chart/image export v0.1 yuzeyinde yoktur.

---

## 4. Ilk Tam Ornek: Support Routing Benchmark

Diyelim ki bir support ticket router'in var. Input olarak ticket aliyor,
output olarak hangi kuyruya gitmesi gerektigini soyluyor.

### 4.1 Uygulama kodu

`app/router.py`:

```python
from __future__ import annotations


def route_ticket(ticket: dict[str, str], *, prompt_version: str) -> dict[str, object]:
    text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()

    if "refund" in text or "billed" in text or "charge" in text:
        queue = "billing"
        confidence = 0.94 if prompt_version == "route-v2" else 0.86
    elif "password" in text or "login" in text:
        queue = "account"
        confidence = 0.88
    else:
        queue = "general"
        confidence = 0.72

    return {
        "queue": queue,
        "confidence": confidence,
        "matched": confidence >= 0.8,
    }
```

### 4.2 Benchmark task'i

`app/benchmarks/support.py`:

```python
from __future__ import annotations

from autobench import DurationMetricSpec, Semantic, SpanKind

from app.router import route_ticket


def run_ticket_case(ctx, case):
    prompt_version = ctx.factor("prompt_version")

    with ctx.span(
        "route_ticket",
        kind=SpanKind.WORKFLOW,
        input=case.input,
        attributes={"prompt_version": prompt_version},
        duration_metric=DurationMetricSpec(name="routing_duration"),
    ) as span:
        output = route_ticket(case.input, prompt_version=prompt_version)
        span.set_output(output)

        span.factor(
            "prompt_version",
            prompt_version,
            semantic_type=Semantic.PROMPT_VERSION,
        )
        span.metric(
            "confidence",
            output["confidence"],
            semantic_type=Semantic.QUALITY_SCORE,
        )
        span.outcome(output["matched"])

    return output
```

Burada dikkat edilecek noktalar:

- `ctx.factor("prompt_version")` aktif variant'tan factor okur.
- `ctx.span(...)` benchmark icindeki anlamli bir is parcasini kaydeder.
- `duration_metric=...` verdigin icin span bitince latency otomatik metric olur.
- `span.metric`, `span.factor`, `span.outcome` observation yazar.
- Return edilen `output`, scorer'lar tarafindan okunabilir.

### 4.3 Dataset

`datasets/support_cases.yaml`:

```yaml
cases:
  - id: refund_request
    input:
      subject: Duplicate card charge
      body: I was billed twice yesterday and need a refund.
    expected:
      queue: billing
    tags: [billing, refund]

  - id: login_problem
    input:
      subject: Cannot login
      body: My password reset link expired.
    expected:
      queue: account
    tags: [account]
```

### 4.4 Benchmark spec

`autobench.yaml`:

```yaml
benchmark:
  support-routing:
    description: Route support tickets into the right queue.

    cases: datasets/support_cases.yaml

    run:
      python: app.benchmarks.support:run_ticket_case

    variants:
      route_v1:
        label: baseline routing prompt
        factors:
          prompt_version:
            value: route-v1
            semantic: prompt.version
            optimize: true

      route_v2:
        label: improved routing prompt
        factors:
          prompt_version:
            value: route-v2
            semantic: prompt.version
            optimize: true

    score:
      matched:
        pass: output.matched
        semantic: result.success
        role: objective
        goal: maximize

      queue_correct:
        exact:
          actual: output.queue
          expected: case.expected.queue
        semantic: quality.correctness
        role: objective
        goal: maximize

      confidence:
        value: output.confidence
        semantic: quality.score
        role: diagnostic
        goal: maximize

    report:
      leaderboard:
        show:
          pass_rate:
            metric: result.success
            aggregate: ratio_true
          avg_correctness:
            metric: quality.correctness
            aggregate: mean
          avg_confidence:
            metric: quality.score
            aggregate: mean
      matrix:
        metric: quality.correctness
      compare:
        route_v1 -> route_v2:
          show:
            correctness_delta:
              metric: quality.correctness
              aggregate: mean
```

### 4.5 Calistirma

```bash
uv run autobench validate autobench.yaml
uv run autobench run autobench.yaml --record runs/support-routing
uv run autobench report runs/support-routing
uv run autobench export runs/support-routing --format markdown --path runs/support-routing/report.md
```

Bu noktada Autobench sunlari yapar:

1. YAML spec'i parse eder.
2. Dataset'i yukler.
3. Variant'lari normalize eder.
4. Case x variant matrisini kurar.
5. Her run icin `RunContext` yaratir.
6. `app.benchmarks.support:run_ticket_case` fonksiyonunu cagirir.
7. Task observation'larini toplar.
8. Scorer'lari kosar.
9. Score'lari metric observation'a cevirir.
10. Policy/derivation varsa uygular.
11. RunRecord ve ExperimentRecord yazar.
12. Rich terminal tablolarla ozet gosterir.

---

## 5. YAML Spec Tasarimi

Autobench YAML DSL iki formu destekler:

1. Daha insan dostu authoring DSL.
2. Daha structured internal payload.

Yeni benchmark yazarken authoring DSL daha okunaklidir:

```yaml
benchmark:
  my-benchmark-id:
    description: Human readable benchmark description.
    cases: datasets/cases.yaml
    run:
      python: my_package.benchmarks:run_case
    variants:
      baseline:
        factors:
          model:
            value: openrouter:openai/gpt-5.6-luna
            semantic: llm.model.name
            optimize: true
    score:
      success:
        pass: output.ok
        semantic: result.success
    report:
      leaderboard:
        show:
          success_rate:
            metric: result.success
            aggregate: ratio_true
```

Autobench tarafindan yazilan YAML'lar schema header alir. Schema cache path'i
Autobench versiyonuna baglidir:

```yaml
# yaml-language-server: $schema=/Users/you/.autobench/0.2.0/schemas/benchmark_schema.json
```

Bu sayede editor tarafinda auto-completion hedeflenir.

### 5.1 Dataset inline yazilabilir

```yaml
benchmark:
  inline-demo:
    dataset:
      cases:
        - id: case_1
          input:
            value: 10
          expected:
            doubled: 20
    run:
      python: app.tasks:double
    variants:
      default: {}
```

### 5.2 Dataset dosyadan okunabilir

```yaml
benchmark:
  file-backed-demo:
    cases: datasets/cases.yaml
    run:
      python: app.tasks:run_case
```

`datasets/cases.yaml`:

```yaml
cases:
  - id: case_1
    input:
      value: 10
    expected:
      doubled: 20
```

### 5.3 Case defaults

Case defaults ortak metadata, tags veya input/expected parcasi vermek icin
kullanilir:

```yaml
benchmark:
  defaulted-demo:
    dataset:
      defaults:
        metadata:
          owner: support-team
        tags: [smoke]
        input:
          locale: tr-TR
      cases:
        - id: refund_tr
          input:
            subject: Para iadesi
          expected:
            queue: billing
```

Autobench case defaults'i case ile merge eder.

### 5.4 Variant factors

Variant, "hangi deney kosulu" sorusunun cevabidir:

```yaml
variants:
  gpt_5_6_luna:
    label: OpenAI Luna model through OpenRouter
    factors:
      provider:
        value: openrouter
        semantic: llm.provider
      model:
        value: openrouter:openai/gpt-5.6-luna
        semantic: llm.model.name
        optimize: true
      temperature:
        value: 0.2
        semantic: llm.temperature

  gemini_flash:
    label: Gemini flash model
    factors:
      provider:
        value: google
        semantic: llm.provider
      model:
        value: gemini-3-flash-preview
        semantic: llm.model.name
        optimize: true
      temperature:
        value: 0.2
        semantic: llm.temperature
```

`optimize: true`, ileride autoptimize tarafinin "bu factor oynanabilir" diye
okuyabilecegi bir sinyaldir. Autobench tek basina optimizasyon yapmaz; ama
optimize edilebilir evidence toplar.

---

## 6. Python Task Runtime

Task hedefleri `module:function` formatindadir:

```yaml
run:
  python: app.benchmarks.support:run_ticket_case
```

Structured form:

```yaml
task:
  kind: python
  target: app.benchmarks.support:run_ticket_case
```

Task sync olabilir:

```python
def run_case(ctx, case):
    return {"ok": True}
```

Task async olabilir:

```python
async def run_case(ctx, case):
    result = await my_async_app(case.input)
    return result
```

Task exception firlatirsa Autobench:

- exception'i `ErrorRecord` olarak yakalar
- run status'u errored yapar
- onceki observation/artifact'lari korur
- sonraki run'lara devam eder

### 6.1 Context'ten factor okuma

```python
def run_case(ctx, case):
    model_name = ctx.factor("model")
    temperature = ctx.factor("temperature")
    return run_model(case.input, model=model_name, temperature=temperature)
```

Factor yoksa `KeyError` alirsin. Bu iyi bir sey; benchmark spec ve task
arasindaki sozlesme bozuldu demektir.

### 6.2 Metric yazma

```python
from autobench import Direction, ObservationRole, Semantic


def run_case(ctx, case):
    output = do_work(case.input)

    ctx.metric(
        "answer_quality",
        output["score"],
        semantic_type=Semantic.QUALITY_SCORE,
        unit=None,
        direction=Direction.MAXIMIZE,
        role=ObservationRole.OBJECTIVE,
    )

    return output
```

Metric isimleri lokal olabilir; semantic type kalici anlamdir.

### 6.3 Outcome yazma

```python
def run_case(ctx, case):
    output = do_work(case.input)
    ctx.outcome(output["ok"])
    return output
```

`ctx.outcome(True)` su anlama gelir:

- metric name: `success`
- semantic type: `result.success`
- role: objective

### 6.4 Check yazma

```python
from autobench import Semantic


def run_case(ctx, case):
    output = do_work(case.input)
    expected = case.expected["label"]

    ctx.check(
        "label_matches",
        output["label"] == expected,
        reason=f"expected={expected}, got={output['label']}",
        semantic_type=Semantic.QUALITY_CORRECTNESS,
    )

    return output
```

`check` constraint role ile metric yazar. Policy gibi daha genel gate'lerden
farkli olarak task icinde domain-specific assertion kullanmak icin pratik bir
helper'dir.

### 6.5 Artifact yazma

```python
def run_case(ctx, case):
    output = do_work(case.input)

    ctx.artifact(
        "raw_response",
        output,
        media_type="application/x-yaml",
        tags={"kind": "debug"},
    )

    return output
```

Record sirasinda artifact payload'lari `artifacts/` altina materialize edilir.
RunRecord sadece artifact reference tutar.

### 6.6 Span kullanma

```python
from autobench import DurationMetricSpec, Semantic, SpanKind


def run_case(ctx, case):
    with ctx.span(
        "retrieve_context",
        kind=SpanKind.RETRIEVER,
        input={"query": case.input["question"]},
        duration_metric=DurationMetricSpec(
            name="retrieval_latency",
            semantic_type=Semantic.TIME_LATENCY,
            unit="s",
        ),
    ) as span:
        docs = retrieve(case.input["question"])
        span.set_output({"doc_count": len(docs)})
        span.metric("retrieved_docs", len(docs), semantic_type="retrieval.docs.count")

    with ctx.span("answer", kind=SpanKind.LLM) as span:
        answer = answer_question(case.input["question"], docs)
        span.set_output(answer)

    return {"answer": answer}
```

Span kaydi sunlari tasiyabilir:

- id
- name
- kind: `agent`, `llm`, `tool`, `retriever`, `parser`, `workflow`, `custom`
- parent id
- started_at / ended_at
- duration_seconds
- input / output
- attributes
- usage
- observations
- artifacts
- error
- tags

### 6.7 Span duration framework'e ait

Task icinde sunu yazmana gerek yok:

```python
from time import perf_counter

start = perf_counter()
result = call()
latency = perf_counter() - start
```

Onun yerine:

```python
from autobench import DurationMetricSpec

with ctx.span("call", duration_metric=DurationMetricSpec(name="call_latency")):
    result = call()
```

Span bitince `duration_seconds` dolacak ve ayrica metric observation
yazilacaktir.

---

## 7. Measurement Helper

Span tek bir operasyonun suresini olcer. Bazen benchmark'ta ayni callable'i
birden fazla kez kosup median/p95/noise almak istersin. Autobench bunun icin
generic measurement helper sunar.

```python
from autobench import Semantic, measure_callable


def run_case(ctx, case):
    def candidate():
        return expensive_function(case.input["payload"])

    measurement = measure_callable(
        candidate,
        warmup=3,
        repetitions=10,
        max_seconds=30.0,
    )

    ctx.record_measurement(
        "candidate_runtime",
        measurement,
        semantic_type=Semantic.TIME_LATENCY,
        unit="ms",
        include_samples_artifact=True,
    )

    return {
        "median_ms": measurement.median_ms,
        "p95_ms": measurement.p95_ms,
        "timed_out": measurement.timed_out,
    }
```

`Measurement` uzerindeki pratik property'ler:

```python
measurement.samples_seconds
measurement.samples_ms
measurement.repetition_count
measurement.median_seconds
measurement.median_ms
measurement.mean_ms
measurement.min_ms
measurement.max_ms
measurement.p95_ms
measurement.standard_deviation_ms
measurement.range_noise_pct
measurement.is_noisy(20.0)
```

Custom timer verebilirsin:

```python
def my_timer(fn):
    started = monotonic_ns()
    fn()
    ended = monotonic_ns()
    return (ended - started) / 1_000_000_000


measurement = measure_callable(fn, repetitions=20, timer=my_timer)
```

Bu helper CUDA, browser, database veya LLM bilmez. Sadece callable olcer.
Domain-specific setup/teardown task'in icinde kalir.

---

## 8. Observation Sistemi

Observation Autobench'in en merkezi veri tiplerinden biridir:

```python
from autobench import Observation, ObservationKind, Semantic

obs = Observation(
    id="obs_1",
    name="input_tokens",
    kind=ObservationKind.METRIC,
    semantic_type=Semantic.LLM_TOKENS_INPUT,
    value=1234,
    unit="tokens",
    case_id="case_1",
    variant_id="gpt_5_6_luna",
)
```

Observation kind'lari:

```text
metric
factor
event
artifact
```

Observation role'lari:

```text
objective
constraint
diagnostic
```

Observation source'lari:

```text
task_observation
score
derived
imported
```

Bu ayrim neden onemli?

- Task metric'i ile scorer metric'i ayni sey degil.
- Derived cost ile raw token observation ayni sey degil.
- Imported trace observation'i ile local span metric'i ayni kaynak degil.
- Objective optimize edilir, diagnostic sadece yorumlanir.

Autobench reporting/projection katmani duplicate semantic metric'leri kaynak
onceligine gore ele alir. Ornegin scorer output'u ayni semantic type icin task
observation'ina gore daha guclu evidence olabilir.

---

## 9. Semantic Type Nedir?

Autobench'te metric ismi ve semantic type farkli seylerdir.

```python
ctx.metric("prompt_tokens", 1000, semantic_type="llm.tokens.input")
ctx.metric("input_tokens", 1000, semantic_type="llm.tokens.input")
ctx.metric("tokens_in", 1000, semantic_type="llm.tokens.input")
```

Bu uc metric farkli isimde olabilir ama ayni anlama gelir:

```text
llm.tokens.input
```

Built-in semantic type ornekleri:

```python
from autobench import Semantic

Semantic.LLM_TOKENS_INPUT              # "llm.tokens.input"
Semantic.LLM_TOKENS_OUTPUT             # "llm.tokens.output"
Semantic.LLM_TOKENS_TOTAL              # "llm.tokens.total"
Semantic.LLM_MODEL_NAME                # "llm.model.name"
Semantic.LLM_PROVIDER                  # "llm.provider"
Semantic.MONEY_COST                    # "money.cost"
Semantic.TIME_LATENCY                  # "time.latency"
Semantic.RESULT_SUCCESS                # "result.success"
Semantic.QUALITY_SCORE                 # "quality.score"
Semantic.QUALITY_CORRECTNESS           # "quality.correctness"
Semantic.COVERAGE_RATIO                # "coverage.ratio"
Semantic.PROMPT_VERSION                # "prompt.version"
Semantic.DATASET_VERSION               # "dataset.version"
Semantic.AGENT_TASK_COMPLETION         # "agent.task.completion"
Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS
```

Semantic registry parent/alias bilir:

```python
from autobench import DEFAULT_SEMANTIC_REGISTRY, Semantic

registry = DEFAULT_SEMANTIC_REGISTRY

assert registry.normalize("quality.answer") == Semantic.QUALITY_SCORE
assert registry.is_a("agent.tool.argument.correctness", "quality.correctness")
assert registry.is_a("optimization.cost", "money.cost")
```

Custom semantic type ekleyebilirsin:

```yaml
benchmark:
  search-demo:
    semantic_registry:
      types:
        search.ndcg:
          parent: quality.score
          shape: number
        retrieval.docs.count:
          shape: integer
      aliases:
        ndcg_at_10: search.ndcg
```

Semantic awareness'in asil faydasi:

- Farkli benchmark'larda metric isimleri degisse bile rapor ayni semantic
  type'a gore calisir.
- Cost derivation dogru token/model/provider input'larini bulabilir.
- Autoptimize ileride objective/constraint/diagnostic ayrimini anlayabilir.
- Agentic metrics domain-specific isimlerden bagimsiz normalize edilebilir.

---

## 10. Scoring

Task output'u tek basina benchmark sonucu degildir. Scoring katmani output'u,
case expected degerlerini, span'lari ve custom scorer'lari kullanarak
`ScoreRecord` uretir.

### 10.1 Output metric scorer

YAML:

```yaml
score:
  confidence:
    value: output.confidence
    semantic: quality.score
    role: diagnostic
    goal: maximize
```

Python:

```python
from autobench import OutputMetricScorer, Semantic

scorer = OutputMetricScorer(
    name="confidence",
    path="output.confidence",
    semantic_type=Semantic.QUALITY_SCORE,
)
```

`path` dotted path'tir. `output.confidence`, return edilen output dict veya
object uzerinden cozulur.

### 10.2 Pass/fail scorer

YAML:

```yaml
score:
  success:
    pass: output.ok
    semantic: result.success
```

Python:

```python
from autobench import PassFailScorer, Semantic

PassFailScorer(
    name="success",
    path="output.ok",
    semantic_type=Semantic.RESULT_SUCCESS,
)
```

Bu scorer path'teki degeri `bool(...)` olarak yorumlar.

### 10.3 Exact scorer

YAML:

```yaml
score:
  answer_correct:
    exact:
      actual: output.answer
      expected: case.expected.answer
    semantic: quality.correctness
```

Python:

```python
from autobench import ExactScorer, Semantic

ExactScorer(
    name="answer_correct",
    actual="output.answer",
    expected="case.expected.answer",
    semantic_type=Semantic.QUALITY_CORRECTNESS,
)
```

Actual ve expected esit ise value `1.0`, degilse `0.0` olur.

### 10.4 Schema scorer

YAML:

```yaml
score:
  output_shape:
    schema:
      path: output
      schema:
        type: object
        required: [answer, citations]
    semantic: agent.output.structure.validity
```

Python:

```python
from autobench import SchemaScorer, Semantic

SchemaScorer(
    name="output_shape",
    path="output",
    schema={"type": "object", "required": ["answer", "citations"]},
    semantic_type=Semantic.AGENT_OUTPUT_STRUCTURE_VALIDITY,
)
```

v0 yuzeyi object schema ve required key kontroluyle sinirlidir.

### 10.5 Python scorer

YAML:

```yaml
score:
  rubric_score:
    python: app.scorers:score_answer
    semantic: quality.score
    role: objective
```

`app/scorers.py`:

```python
from autobench import ScoreRecord, Semantic


def score_answer(call):
    output = call.output
    expected = call.case.expected

    score = 0.0
    if expected["must_mention"] in output["answer"].lower():
        score += 0.7
    if output.get("citations"):
        score += 0.3

    return ScoreRecord(
        name="rubric_score",
        semantic_type=Semantic.QUALITY_SCORE,
        value=score,
        actual_value=output,
        expected_value=expected,
    )
```

Python scorer isterse direkt value donebilir:

```python
def score_length(call):
    return min(len(call.output["answer"]) / 500, 1.0)
```

Async scorer da desteklenir:

```python
async def score_with_judge(call):
    return await judge_answer(call.output, call.case.expected)
```

`ScoringCall` ile gelenler:

```python
call.ctx
call.task_result
call.output
call.case
call.variant
call.observations
call.spans
```

### 10.6 Span selector ile component-level scoring

Bazen tum output'u degil belirli span'i score etmek istersin.

```yaml
score:
  tool_arguments:
    expected_action:
      metric: arguments
      observed_kind: tool
    span:
      kind: tool
      name: lookup_user
    semantic: agent.tool.argument.correctness
```

`span` selector alanlari:

```yaml
span:
  kind: tool
  name: lookup_user
  tag:
    phase: retrieval
  path: support_agent/lookup_user
  semantic_type: agent.tool.name
```

Selector, scorer'in sadece ilgili span'lari gormesini saglar.

---

## 11. Agentic Evidence Ve Expected Action Scoring

Agent benchmark'larinda sadece final answer score etmek yetmez. Agent hangi
tool'u cagirdi, argumanlari dogru muydu, siralama dogru muydu, bunlari da
olcmek gerekir.

### 11.1 Tool span'i kaydetme

```python
from autobench import SpanKind


def run_case(ctx, case):
    with ctx.span("support_agent", kind=SpanKind.AGENT) as agent:
        with ctx.span(
            "lookup_user",
            kind=SpanKind.TOOL,
            input={"user_id": case.input["user_id"]},
            tags={"tool": "lookup_user"},
        ) as tool:
            result = lookup_user(case.input["user_id"])
            tool.set_output(result)

        answer = build_answer(result)
        agent.set_output(answer)

    return {"answer": answer}
```

### 11.2 Case expected action

```python
from autobench import Case

case = Case(
    id="refund_gold_user",
    input={"user_id": "u_123", "message": "I need a refund"},
    expected={
        "actions": [
            {
                "id": "lookup_gold_user",
                "kind": "tool",
                "target": "lookup_user",
                "input": {"user_id": "u_123"},
                "order": 1,
                "required": True,
            }
        ]
    },
)
```

YAML:

```yaml
cases:
  - id: refund_gold_user
    input:
      user_id: u_123
      message: I need a refund
    expected:
      actions:
        - id: lookup_gold_user
          kind: tool
          target: lookup_user
          input:
            user_id: u_123
          order: 1
          required: true
```

### 11.3 ExpectedActionScorer

```python
from autobench import ExpectedActionScorer, Semantic, SpanSelector

ExpectedActionScorer(
    name="tool_arguments",
    semantic_type=Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS,
    metric="arguments",
    observed_kind="tool",
    span=SpanSelector(kind="tool"),
)
```

YAML:

```yaml
score:
  tool_selection:
    expected_action:
      metric: selection
      observed_kind: tool
    span:
      kind: tool
    semantic: agent.tool.selection.correctness

  tool_arguments:
    expected_action:
      metric: arguments
      observed_kind: tool
    span:
      kind: tool
    semantic: agent.tool.argument.correctness

  tool_sequence:
    expected_action:
      metric: sequence
      observed_kind: tool
    span:
      kind: tool
    semantic: agent.tool.sequence.correctness
```

Metric secenekleri:

```text
selection
arguments
sequence
```

Bu yuzey DeepEval tarzi agent eval ihtiyaclarina benzese de Autobench bunu
generic semantic evidence modeline koyar. Tool correctness sadece LLM degil,
herhangi bir workflow tool sistemi icin de kullanilabilir.

---

## 12. Token Cost Derivation

Autobench cost'u kendisi "bilir" gibi davranmaz. Cost, semantic input'lardan
turetilen bir metriktir:

Gerekli semantic input'lar:

```text
llm.tokens.input
llm.tokens.output
llm.provider
llm.model.name
```

Task bu degerleri observation olarak yazar:

```python
from autobench import Semantic


def run_case(ctx, case):
    result = call_llm(case.input["prompt"])
    usage = result.usage

    ctx.metric("input_tokens", usage.input_tokens, semantic_type=Semantic.LLM_TOKENS_INPUT)
    ctx.metric("output_tokens", usage.output_tokens, semantic_type=Semantic.LLM_TOKENS_OUTPUT)
    ctx.factor_observation("provider", "openrouter", semantic_type=Semantic.LLM_PROVIDER)
    ctx.factor_observation("model", "openrouter:openai/gpt-5.6-luna", semantic_type=Semantic.LLM_MODEL_NAME)

    return {"answer": result.output}
```

Pricing YAML:

```yaml
pricing:
  provider: openrouter
  source: manual
  models:
    openai/gpt-5.6-luna:
      input:
        unit: mtok
        price: 0.4
      output:
        unit: mtok
        price: 1.6
```

Tiered pricing:

```yaml
pricing:
  models:
    openai/gpt-tiered:
      input:
        unit: mtok
        price: 1.0
      output:
        unit: mtok
        tiers:
          - up_to: 500
            price: 4.0
          - price: 2.0
      cache_read:
        unit: mtok
        price: 0.2
      cache_write:
        unit: mtok
        price: 0.5
```

Benchmark spec:

```yaml
benchmark:
  llm-cost-demo:
    cases:
      - id: case_1
        input:
          prompt: Say hello
    run:
      python: app.tasks:run_case
    variants:
      default: {}
    derive:
      - kind: token_cost
        pricing: pricing/models.yaml
        output:
          name: cost
          semantic_type: money.cost
          unit: usd
        inputs:
          input_tokens: llm.tokens.input
          output_tokens: llm.tokens.output
          provider: llm.provider
          model: llm.model.name
```

Minimal formda `output` ve `inputs` verilmezse default kullanilir:

```yaml
derive:
  - kind: token_cost
    pricing: pricing/models.yaml
```

Cost bulunamazsa Autobench uydurma cost yazmaz. Diagnostic observation uretir:

- `token_cost_missing_inputs`
- `token_cost_unknown_pricing`
- `token_cost_missing_rates`

Bu tasarim onemli: Benchmark "bilmedigi seyi" tahmin etmez; eksik evidence'i
gorunur yapar.

---

## 13. Paired Baseline Post-Derivation

Normal deriver tek run icindeki observation'lardan metric uretir. Paired
baseline ise experiment bittikten sonra run'lari birbirine bakarak karsilastirir.

Klasik use case:

- baseline latency: 100 ms
- candidate latency: 50 ms
- speedup: 2.0

Task:

```python
from autobench import Semantic


def run_case(ctx, case):
    if ctx.variant.id == "baseline":
        latency_ms = case.input["baseline_ms"]
    else:
        latency_ms = case.input["candidate_ms"]

    ctx.metric(
        "median_latency",
        latency_ms,
        semantic_type=Semantic.TIME_LATENCY,
        unit="ms",
    )
    return {"latency_ms": latency_ms}
```

YAML:

```yaml
benchmark:
  latency-comparison:
    cases:
      - id: easy
        input:
          baseline_ms: 100
          candidate_ms: 50
      - id: hard
        input:
          baseline_ms: 240
          candidate_ms: 180

    run:
      python: app.tasks:run_case

    variants:
      baseline: {}
      candidate: {}

    post_derive:
      - kind: paired_baseline
        baseline_variant: baseline
        match_on:
          - kind: case_id
        metric: time.latency
        output:
          name: speedup
          semantic_type: performance.speedup
        formula: baseline_over_candidate

    report:
      leaderboard:
        show:
          avg_speedup:
            metric: performance.speedup
            aggregate: mean
      matrix:
        metric: performance.speedup
```

Formula secenekleri:

```text
baseline_over_candidate
candidate_over_baseline
candidate_minus_baseline
baseline_minus_candidate
percent_change_from_baseline
```

Eslestirme sadece case id olmak zorunda degil. Factor da kullanilabilir:

```yaml
match_on:
  - kind: case_id
  - kind: factor
    name: workload.size
```

Threshold ve verdict:

```yaml
post_derive:
  - kind: paired_baseline
    baseline_variant: baseline
    match_on:
      - kind: case_id
    metric: time.latency
    output:
      name: speedup
      semantic_type: performance.speedup
    formula: baseline_over_candidate
    threshold:
      kind: relative_noise
      pct: 2.0
    verdict:
      output:
        name: latency_verdict
        semantic_type: comparison.verdict
      threshold:
        kind: relative_noise
        pct: 2.0
```

Missing behavior:

```yaml
missing: diagnostic
zero_division: diagnostic
diagnostics_name: paired_baseline_unavailable
```

veya:

```yaml
missing: skip
zero_division: skip
```

Autobench burada causal attribution iddia etmez. Sadece evidence uretir:

> Baseline ve candidate ayni case/factor match uzerinde karsilastirildi,
> speedup su kadar.

"Bu degisime kesin olarak hangi factor sebep oldu?" sorusu autoptimize veya
daha kontrollu deney tasarimi konusudur.

---

## 14. Policy Checks

Policy, release gate gibi dusunulebilir. Bir semantic metric belirli bir sarti
saglamali.

Python:

```python
from autobench import PolicySpec, Semantic

policies = [
    PolicySpec(
        name="minimum_quality",
        metric=Semantic.QUALITY_SCORE,
        must_greater_equal=0.8,
    ),
    PolicySpec(
        name="maximum_cost",
        metric=Semantic.MONEY_COST,
        must_less_equal=0.002,
    ),
]
```

YAML:

```yaml
policies:
  - name: minimum_quality
    metric: quality.score
    must_greater_equal: 0.8

  - name: maximum_cost
    metric: money.cost
    must_less_equal: 0.002

  - name: queue_allowed
    metric: support.queue
    must_in: [billing, account, general]
```

Desteklenen requirement alanlari:

```text
must_equal
must_not_equal
must_greater
must_greater_equal
must_less
must_less_equal
must_in
must_not_in
must_between
```

`must_between`:

```yaml
policies:
  - name: confidence_band
    metric: quality.score
    must_between:
      min: 0.7
      max: 1.0
      inclusive: true
```

Policy sonucu `policy.result` semantic type'iyle derived observation olarak
eklenir. Policy fail, final run status'u etkileyebilir.

---

## 15. Reporting

Report spec su an dort ana gorunum bilir:

- leaderboard
- case matrix
- comparison
- distribution

### 15.1 Leaderboard

```yaml
report:
  leaderboard:
    show:
      pass_rate:
        metric: result.success
        aggregate: ratio_true
      avg_quality:
        metric: quality.score
        aggregate: mean
      total_cost:
        metric: money.cost
        aggregate: sum
```

Python model:

```python
from autobench import LeaderboardReportSpec, MetricAggregation, ReportSpec, Semantic

report_spec = ReportSpec(
    leaderboard=LeaderboardReportSpec(
        metrics=(
            MetricAggregation(
                name="pass_rate",
                semantic_type=Semantic.RESULT_SUCCESS,
                fn="ratio_true",
            ),
            MetricAggregation(
                name="avg_quality",
                semantic_type=Semantic.QUALITY_SCORE,
                fn="mean",
            ),
        )
    )
)
```

Aggregation fonksiyonlari:

```text
count
mean
sum
min
max
median
p95
stddev
geomean
ratio_true
```

### 15.2 Case matrix

```yaml
report:
  matrix:
    metric: quality.correctness
```

Bu case x variant matrisi uretir. Bir case hangi variant'ta iyi/kotu gitti
gorulebilir.

### 15.3 Comparison

```yaml
report:
  compare:
    baseline -> candidate:
      show:
        avg_speedup:
          metric: performance.speedup
          aggregate: mean
        avg_quality:
          metric: quality.score
          aggregate: mean
```

CLI:

```bash
uv run autobench compare runs/latency --baseline baseline --candidate candidate
```

Compare, factor delta ve metric delta gosterir ama causal claim yapmaz.

### 15.4 Distribution

Structured model:

```python
from autobench import DistributionReportSpec, ReportSpec, Semantic

ReportSpec(
    distributions=(
        DistributionReportSpec(
            name="latency_distribution",
            semantic_type=Semantic.TIME_LATENCY,
            summaries=("min", "median", "p95", "max"),
        ),
    )
)
```

YAML structured form:

```yaml
reports:
  distributions:
    - name: latency_distribution
      semantic_type: time.latency
      summaries: [min, median, p95, max]
```

### 15.5 Export

```bash
uv run autobench export runs/support-routing --format markdown --path report.md
uv run autobench export runs/support-routing --format yaml --path report.yaml
uv run autobench export runs/support-routing --format csv --path runs.csv
```

Export stdout'a raw YAML/Markdown basmaz. Dosyaya yazar, terminalde Rich preview
gosterir.

---

## 16. Recording Ve Replay

`autobench run --record runs/demo` su dosyalari yazar:

```text
runs/demo/
  experiment.yaml
  summary.yaml
  cases/
    <case_id>/
      <variant_id>/
        run.yaml
  artifacts/
    ...
```

RunRecord alanlari:

```text
record_version
run_id
experiment_id
benchmark_id
case_id
variant_id
status
evaluation_status
task_status
case
task_output
observations
scores
spans
artifacts
factors
asset_versions
parent_run_id
errors
error
```

ExperimentRecord alanlari:

```text
record_version
experiment_id
benchmark_id
plan
environment
semantic_registry
report_spec_data
spec_snapshot
spec_hash
file_hashes
run_paths
run_count
passed_count
failed_count
errored_count
skipped_count
```

Replay:

```bash
uv run autobench replay runs/demo
```

Replay task module'unu import etmez. Bu cok kritik:

- Model API key gerekmez.
- External service tekrar cagirilmaz.
- Onceki experiment reproducible raporlanir.
- Eski app kodu degismis olsa bile record okunabilir.

Programmatic replay:

```python
from pathlib import Path
from autobench import build_report, load_experiment_record, replay_experiment

record = load_experiment_record(Path("runs/demo"))
result = replay_experiment(record, root_dir=Path("runs/demo"))
report = build_report(result)
```

Programmatic record:

```python
from pathlib import Path
from autobench import collect_benchmark_source_files, record_experiment, run_benchmark_path

spec_path = Path("autobench.yaml")
result = run_benchmark_path(spec_path, concurrency_limit=2)
source_files = collect_benchmark_source_files(spec_path)
record = record_experiment(result, Path("runs/demo"), source_files=source_files)
```

Record append-only'dir. Aynı output dir'de `experiment.yaml` varsa
`RecordingError` alirsin. Bu, eski evidence'in sessizce overwrite edilmesini
engeller.

---

## 17. Python Builder API

YAML ana kaynak olmali, ama Python builder da var. Basit benchmark'lar icin
ergonomiktir.

```python
from autobench import Benchmark, Case, ExactScorer, FactorValue, PassFailScorer, Semantic, Variant

result = (
    Benchmark("builder-demo")
    .description("Small builder API demo.")
    .dataset(
        [
            Case(
                id="case_1",
                input={"text": "refund please"},
                expected={"queue": "billing"},
            )
        ]
    )
    .variants(
        [
            Variant(
                id="route_v1",
                factors=[
                    FactorValue(
                        name="prompt_version",
                        value="route-v1",
                        semantic_type=Semantic.PROMPT_VERSION,
                    )
                ],
            ),
            {
                "id": "route_v2",
                "factors": {
                    "prompt_version": {
                        "value": "route-v2",
                        "semantic_type": Semantic.PROMPT_VERSION,
                    }
                },
            },
        ]
    )
    .task("app.benchmarks.support:run_ticket_case")
    .scoring(
        [
            PassFailScorer(
                name="success",
                path="output.matched",
                semantic_type=Semantic.RESULT_SUCCESS,
            ),
            ExactScorer(
                name="queue_correct",
                actual="output.queue",
                expected="case.expected.queue",
                semantic_type=Semantic.QUALITY_CORRECTNESS,
            ),
        ]
    )
    .run(concurrency_limit=1)
)
```

Builder su anda temel spec parcalarini kurar:

- benchmark id/description
- dataset
- variants
- task
- scoring
- derive
- run/run_async

Report/post-derive/policy gibi daha gelismis yuzeylerde YAML veya Pydantic
model'leri dogrudan kullanmak daha nettir.

---

## 18. Programmatic Full Spec

Her seyi Python model'leriyle de kurabilirsin:

```python
from pathlib import Path

from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    ComparisonReportSpec,
    DatasetSpec,
    DerivedMetricOutput,
    Direction,
    FactorValue,
    LeaderboardReportSpec,
    MetricAggregation,
    ObservationRole,
    PairedBaselineDeriverSpec,
    PassFailScorer,
    PolicySpec,
    ReportSpec,
    RunMatchKey,
    Semantic,
    TaskSpec,
    TokenCostDeriverSpec,
    TokenCostInputs,
    Variant,
    run_benchmark_spec,
)


spec = BenchmarkSpec(
    benchmark=BenchmarkInfo(
        id="programmatic-demo",
        description="Fully typed benchmark spec.",
    ),
    dataset=DatasetSpec(
        cases=[
            Case(id="case_1", input={"prompt": "Say hi"}, expected={"ok": True}),
        ]
    ),
    task=TaskSpec(kind="python", target="app.tasks:run_case"),
    variants=[
        Variant(
            id="baseline",
            factors=[
                FactorValue(
                    name="model",
                    value="gpt-baseline",
                    semantic_type=Semantic.LLM_MODEL_NAME,
                )
            ],
        ),
        Variant(
            id="candidate",
            factors=[
                FactorValue(
                    name="model",
                    value="gpt-candidate",
                    semantic_type=Semantic.LLM_MODEL_NAME,
                )
            ],
        ),
    ],
    scoring=[
        PassFailScorer(
            name="success",
            path="output.ok",
            semantic_type=Semantic.RESULT_SUCCESS,
            role=ObservationRole.OBJECTIVE,
            direction=Direction.MAXIMIZE,
        )
    ],
    derive=[
        TokenCostDeriverSpec(
            pricing="pricing/models.yaml",
            output=DerivedMetricOutput(
                name="cost",
                semantic_type=Semantic.MONEY_COST,
                unit="usd",
            ),
            inputs=TokenCostInputs(),
        )
    ],
    post_derive=[
        PairedBaselineDeriverSpec(
            baseline_variant="baseline",
            match_on=(RunMatchKey(kind="case_id"),),
            metric=Semantic.TIME_LATENCY,
            output=DerivedMetricOutput(
                name="speedup",
                semantic_type="performance.speedup",
            ),
        )
    ],
    policies=[
        PolicySpec(
            name="must_succeed",
            metric=Semantic.RESULT_SUCCESS,
            must_equal=True,
        )
    ],
    reports=ReportSpec(
        leaderboard=LeaderboardReportSpec(
            metrics=(
                MetricAggregation(
                    name="success_rate",
                    semantic_type=Semantic.RESULT_SUCCESS,
                    fn="ratio_true",
                ),
                MetricAggregation(
                    name="total_cost",
                    semantic_type=Semantic.MONEY_COST,
                    fn="sum",
                ),
            )
        ),
        comparisons=(
            ComparisonReportSpec(
                baseline="baseline",
                candidate="candidate",
                metrics=(
                    MetricAggregation(
                        name="avg_speedup",
                        semantic_type="performance.speedup",
                        fn="mean",
                    ),
                ),
            ),
        ),
    ),
)
```

Async run:

```python
result = await run_benchmark_spec(spec, experiment_id="exp_programmatic")
```

Sync wrapper kullanmak istersen:

```python
import asyncio

result = asyncio.run(run_benchmark_spec(spec))
```

---

## 19. Asset Tracking

Autobench'in en onemli uzun vadeli parcalarindan biri asset tracking'dir.
Prompt, tool, output type, dataclass ve typed class gibi uygulama varliklarini
version'layabilir.

### 19.1 Prompt tracking

Text inline:

```python
from autobench import track

SYSTEM_PROMPT = track.prompt(
    name="support_router_system",
    text="Route the support ticket to billing, account, or general.",
)

print(str(SYSTEM_PROMPT))
print(SYSTEM_PROMPT.raw)
print(SYSTEM_PROMPT.version)
```

Dosyadan:

```python
from autobench import track

SYSTEM_PROMPT = track.prompt(
    name="support_router_system",
    source="./prompts/support_router.md",
)
```

`raw` property, prompt text'ine dogrudan ulasmak icindir. `str(prompt)` da
ayni metni verir.

### 19.2 Tool tracking

```python
from typing import Literal

from autobench import track

Queue = Literal["billing", "account", "general"]


@track.tool
def assign_queue(ticket_id: str, queue: Queue, priority: int = 0) -> dict[str, object]:
    """Assign a support ticket to a queue."""
    return {"ticket_id": ticket_id, "queue": queue, "priority": priority}
```

Autobench tool icin sunlari cikarmaya calisir:

- name
- qualname
- docstring
- param schema
- param annotation
- literal choices
- required/default bilgisi
- return annotation
- return type asset id
- source hash
- content hash

### 19.3 Pydantic model tracking

```python
from typing import Literal

from autobench import track
from pydantic import BaseModel, Field

Queue = Literal["billing", "account", "general"]


@track.type
class RoutingOutput(BaseModel):
    queue: Queue = Field(description="Queue selected for the ticket.")
    confidence: float = Field(ge=0, le=1, description="Router confidence.")
    reasons: list[str] = Field(default_factory=list)
```

Pydantic model icin hash JSON schema / structured fields uzerinden uretilir.
Field asset'leri sunlari tutabilir:

- field name
- annotation
- required/default/default_factory
- description
- examples
- alias
- constraints
- literal choices

### 19.4 Dataclass tracking

Iki sekilde kullanabilirsin.

Standart dataclass ustune:

```python
from dataclasses import dataclass

from autobench import track


@track.type
@dataclass(frozen=True)
class Ticket:
    id: str
    subject: str
```

Daha iyi DX icin Autobench dataclass decorator'i:

```python
from autobench import track


@track.dataclass(frozen=True, slots=True)
class Ticket:
    id: str
    subject: str
```

Bu hem dataclass'i uygular hem de type asset olarak track eder. Decorator
metadata'si de asset metadata'ya girer.

### 19.5 Generic class decorator

Kendi class decorator'ini kullanmak istersen:

```python
from attrs import define
from autobench import track


@track.decorate_type(define, frozen=True)
class AttrsTicket:
    id: str
    subject: str
```

Bu pattern su durumlarda ise yarar:

- `dataclass`
- `attrs.define`
- custom class transformer
- framework-specific typed class decorator'leri

### 19.6 Asset version'lari run'a baglama

```python
from autobench import track

ROUTER_PROMPT = track.prompt(name="router", source="./prompts/router.md")


@track.tool
def lookup_user(user_id: str) -> dict[str, str]:
    return {"tier": "gold"}


def run_case(ctx, case):
    ctx.attach_tracked_asset(ROUTER_PROMPT)
    ctx.attach_tracked_asset(lookup_user)
    return {"ok": True}
```

RunRecord icinde `asset_versions` alanina bu version'lar girer. Boylece daha
sonra "bu run hangi prompt/tool/type version'i ile kosuldu?" sorusu cevaplanir.

### 19.7 Asset registry'yi diske yazma

```python
from pathlib import Path
from autobench import track

track.write_assets(Path(".autobench/assets"))
```

Yazilan yapinin amaci:

- `index.yaml`: asset listesi ve latest version'lar
- `<asset_id>.yaml`: asset metadata, versions, diffs

Autobench version hash ve source hash tutar. Prompt gibi text assetlerde raw
icerik metadata'da tutulur. Type assetlerde structured field/schema bilgisi
hash'e girer.

---

## 20. Instrumentation

Her metric'i task icinde elle yazmak istemeyebilirsin. Autobench runtime
instrumentation ile class method'larindan metric/factor toplayabilir.

Basit servis:

```python
class LLMClient:
    def complete(self, prompt: str):
        return {
            "text": "hello",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "model": "gpt-demo",
                "provider": "openai",
            },
        }
```

Instrumentation:

```python
from autobench import InstrumentFactorSpec, InstrumentMetricSpec, Semantic, instrument_method

handle = instrument_method(
    LLMClient,
    "complete",
    metrics=[
        InstrumentMetricSpec(
            name="input_tokens",
            semantic_type=Semantic.LLM_TOKENS_INPUT,
            value_path="result.usage.input_tokens",
        ),
        InstrumentMetricSpec(
            name="output_tokens",
            semantic_type=Semantic.LLM_TOKENS_OUTPUT,
            value_factory=lambda call: call.result["usage"]["output_tokens"],
        ),
    ],
    factors=[
        InstrumentFactorSpec(
            name="model",
            semantic_type=Semantic.LLM_MODEL_NAME,
            value_path="result.usage.model",
        ),
        InstrumentFactorSpec(
            name="provider",
            semantic_type=Semantic.LLM_PROVIDER,
            value_path="result.usage.provider",
        ),
    ],
)
```

Task icinde:

```python
def run_case(ctx, case):
    client = LLMClient()
    return client.complete(case.input["prompt"])
```

Autobench pipeline calisirken active `RunContext` contextvar olarak set edilir.
Instrumented method cagrildiginda metric/factor observation'lari aktif ctx'e
yazilir.

`value_factory` Python API'deki typed callback seam'idir. `value_path` ise
mapping key, attribute ve sifir argumanli accessor zincirlerini declarative
olarak cozer. YAML tarafinda arbitrary expression calistirilmaz.

Method wrapper instance/static/class/inherited method'lari, sync/async call'lari,
generator ve async generator stream'lerini ve sync/async context manager'lari
destekler. Stream span'i iterator olusturulunca degil gercekten bittiginde,
hata verdiginde veya erken kapandiginda kapanir.

Handle kapatma:

```python
handle.close()
```

Context manager:

```python
with instrument_method(LLMClient, "complete", metrics=[...]):
    result = run_benchmark_path(Path("autobench.yaml"))
```

Not: Bu OTel degildir. Autobench'in kendi lightweight instrumentation
yuzeyidir. OTel/Logfire bridge future plan olarak dusunulur.

Reusable entegrasyonlar `InstrumentorInfo`, `Compatibility`, `Instrumentor`,
`InstrumentationRuntime` ve `InstrumentationManager` kontratini kullanir.
Manager package version'ini hook/patch kurmadan once denetler, duplicate
install'lari reference count ile birlestirir ve son handle kapandiginda native
callback'i veya exact descriptor'u restore eder. Task-local
`suppress_instrumentation("family")` sadece eslesen instrumentor/family'yi
gecici olarak susturur.

### 20.1 Native instrumentor DSL ve doctor

Pydantic AI, OpenAI client, OpenAI Agents ve HTTPX instrumentor'lari typed
Python ayarlariyla veya benchmark YAML icinden secilebilir:

```yaml
instrumentation:
  all:
    exclude: [httpx]
    strict: false
  pydantic_ai: {}
  openai: {}
  httpx:
    capture:
      path: hash
      response_headers: [x-request-id]
```

Python builder'da ortamdaki tum kurulu ve uyumlu built-in entegrasyonlar
`Benchmark.instrument_all()` ile acilir. `exclude` discovery alanini daraltir;
`strict=True` kurulamayan ilk entegrasyonda hata verir. Default mod kurulamayan
entegrasyonu atlar ve nedenini her run'a diagnostic evidence olarak yazar.
Explicit config, `false` dahil, discovery sonucunu override eder; ayni ID'ye
sahip custom runtime instrumentor da otomatik kurulumu engeller.

Tekil config `Benchmark.instrument(...)` ile verilir. Ayarlar
`BenchmarkSpec` icinde serialize edilir; custom `Instrumentor` instance'lari
ayni method'dan gecse de runtime-only kalir. Pipeline instrumentor'lari tum
matrix'ten once kurar ve hata halinde de kapatir.

`autobench instrumentation doctor` kurulu/eksik extra'lari, version
uyumlulugunu, layer/mechanism'i, sync/async/streaming capability'lerini,
semantic aileleri ve capture default'larini Rich tablolarla gosterir. Eksik
SDK import edilmez. `autobench instrumentation trace RUN_DIR` ise provider SDK
ve task import etmeden recorded ABP trace composition ve partial state'i
gosterir.

HTTPX default'u body ve header toplamaz; path hash'lenir. Secret header/body
alanlari explicit capture'da bile redact edilir ve body capture bounded'dir.
Pydantic AI -> OpenAI -> HTTPX layer'lari parent-child olarak compose edilir;
transport token/cost uretmez ve aggregate/direct accounting double count'i
engeller.

Detayli kontrat: [Native Instrumentation](native-instrumentation.md).

### 20.2 ABP trace extraction ve accounting

Instrumentor ham ABP fact'lerini toplar; extractor tamamlanmis immutable
trace'ten tekrar kullanilabilir semantic evidence uretir:

```python
from autobench import CompositeExtractor, SignalExtractor, SpanExtractor, UsageExtractor

extractor = CompositeExtractor(
    SignalExtractor(),
    SpanExtractor(),
    UsageExtractor(),
)
```

- `SignalExtractor`: measurement/event observation'larini scope, layer,
  instrumentor ve logical operation provenance'i ile geri kurar.
- `SpanExtractor`: direct span latency, operation count, max depth/fan-out,
  critical path, parallelism, incomplete work, retry/recovery, validation,
  approval, tool ve reference metric'lerini uretir.
- `UsageExtractor`: LLM request/token accounting'i ile requested model,
  response model ve provider factor'lerini uretir. Cost uretmez; cost mevcut
  pricing deriver'inin sorumlulugudur.

Parent aggregate ile child direct usage toplanmaz. Her semantic icin tek
abstraction boundary secilir; logical operation ID'si ayni olan direct
olcumler deduplicate edilir. Esdeger direct degerler cakisiyor ve unique bir
authority yoksa total uydurulmaz, `ambiguous_direct_measurement` diagnostic'i
uretilir. Aggregate deger direct toplamla uyusmazsa
`aggregate_measurement_mismatch` kaydedilir.

Extractor isim ve version'a sahiptir. `replay_extraction()` yeni bir derived
RunRecord yazar; ayni extractor'in yeni version'i onceki observation'lari yeni
derived record'da degistirir ve parent lineage'i korur.

---

## 21. Pydantic AI Usage Bridge

Pydantic AI result usage bilgisini Autobench semantic observation'a cevirmek
icin helper vardir:

```python
from autobench import PydanticAIUsage, record_pydantic_ai_usage


def run_case(ctx, case):
    result = agent.run_sync(case.input["prompt"])

    usage = result.usage()
    record_pydantic_ai_usage(
        ctx,
        PydanticAIUsage(
            requests=usage.requests,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            model_name="google/gemini-3-flash-preview",
            provider="openrouter",
        ),
    )

    return {"answer": result.output}
```

Eger span icinde kullaniyorsan:

```python
with ctx.span("agent_run", kind="agent") as span:
    result = agent.run_sync(case.input["prompt"])
    record_pydantic_ai_usage(ctx, usage, span_id=span.id)
```

Bu helper su semantic type'lari yazar:

- `llm.tokens.input`
- `llm.tokens.output`
- `llm.tokens.total`
- `llm.model.name`
- `llm.provider`

---

## 22. TraceEnvelope

Dis frameworklerden trace getirmek icin minimal bir envelope var.

```python
from datetime import UTC, datetime

from autobench import SpanRecord, TraceEnvelope, attach_trace


trace = TraceEnvelope(
    trace_id="trace_123",
    name="external_agent_run",
    spans=(
        SpanRecord(
            id="span_llm_1",
            name="llm_call",
            kind="llm",
            started_at=datetime.now(UTC),
            duration_seconds=0.42,
            usage={"input_tokens": 200, "output_tokens": 40},
            attributes={"model": "gpt-demo", "provider": "openai"},
        ),
    ),
)


def run_case(ctx, case):
    attach_trace(ctx, trace)
    return {"ok": True}
```

`trace_to_observations` LLM span usage'larini semantic observation'a cevirir:

- `prompt_tokens` veya `input_tokens` -> `llm.tokens.input`
- `completion_tokens` veya `output_tokens` -> `llm.tokens.output`
- `total_tokens` -> `llm.tokens.total`
- `model` veya `model_name` -> `llm.model.name`
- `provider` -> `llm.provider`
- span duration -> `time.latency`

Bu bir full tracing platformu degil; recorded evidence'a trace eklemek icin
hafif bir adapter yuzeyidir.

---

## 23. Pydantic Evals Payload Bridge

Autobench Pydantic Evals'i ana runtime olarak zorlamaz. Ama payload adapter
vardir.

```python
from autobench import PydanticEvalsBridge, load_benchmark_spec

spec = load_benchmark_spec("autobench.yaml")
bridge = PydanticEvalsBridge()

if bridge.is_available():
    dataset_payload = bridge.dataset_payload(spec)
    print(dataset_payload.name)
    print(dataset_payload.cases[0].inputs)
```

Case payload:

```python
from autobench import Case, PydanticEvalsBridge

bridge = PydanticEvalsBridge()
payload = bridge.case_payload(
    Case(
        id="case_1",
        input={"question": "2+2?"},
        expected={"answer": "4"},
        tags=["math"],
    )
)
```

Payload alanlari:

```text
name
inputs
expected_output
metadata
```

Pydantic Evals yuklu degilse:

```python
bridge.require_module()
```

`PydanticEvalsUnavailableError` firlatir.

---

## 24. Production Sample Ve Synthetic Case Helper'lari

Gercek production data'dan benchmark case uretmek isteyebilirsin.

```python
from datetime import datetime, UTC

from autobench import ProductionSample, SampleReason, ReviewStatus, sample_to_case

sample = ProductionSample(
    id="prod_001",
    input={"message": "refund please"},
    output={"queue": "billing"},
    expected={"queue": "billing"},
    reason=SampleReason.FAILURE_ONLY,
    review_status=ReviewStatus.CANDIDATE,
    timestamp=datetime.now(UTC),
    privacy_tags=("pii_redacted",),
)

case = sample_to_case(sample)
```

`case.metadata` icine su bilgiler girer:

- `source: production`
- `sample_reason`
- `review_status`
- `timestamp`
- `privacy_tags`
- `trace_id` varsa

Bir batch'i policy ile filtrelemek:

```python
from autobench import SamplingPolicy, SampleReason, samples_to_cases

cases = samples_to_cases(
    samples,
    policy=SamplingPolicy(
        reasons=(SampleReason.FAILURE_ONLY, SampleReason.HIGH_COST),
        max_samples=100,
    ),
)
```

Synthetic/generated case isaretleme:

```python
from autobench import Case, generated_batch_from_cases, mark_generated_case

case = Case(id="synthetic_1", input={"question": "..."}, expected={"answer": "..."})

marked = mark_generated_case(
    case,
    generator_asset_version="prompt.generator@abc123",
    model_provider="openrouter",
    model_name="openai/gpt-5.6-luna",
)

batch = generated_batch_from_cases(
    [case],
    generator_asset_version="prompt.generator@abc123",
    model_provider="openrouter",
    model_name="openai/gpt-5.6-luna",
)
```

Bu helper'lar data lineage icindir. Autobench synthetic data generator
calistirmaz; generated case'i evidence modelinde dogru isaretler.

---

## 25. Metric Packs

Metric pack, belli bir domain icin semantic registry delta, scorer factory
isimleri, default report metric'leri ve feedback extractor isimlerini paketler.

Built-in pack'ler:

```python
from autobench import DEFAULT_METRIC_PACKS

print(DEFAULT_METRIC_PACKS.names())
```

Beklenen pack id'leri:

```text
agentic
structured_output
llm_usage
performance
```

Pack okuma:

```python
from autobench import builtin_metric_pack_registry

registry = builtin_metric_pack_registry()
agentic = registry.require("agentic")

print(agentic.default_report_metrics)
print(agentic.scorer_factories)
```

Semantic registry merge:

```python
semantic_registry = registry.semantic_registry_for(["agentic", "llm_usage"])
```

Bu henuz full plugin sistemi degil. Ama ileride `autobench-ai`,
`autobench-agentic`, `autobench-performance` gibi domain paketlerinin
tasiyabilecegi sozlesmenin cekirdegi.

---

## 26. Pricing Source Helper'lari

Autobench pricing library olmak istemez. Ama LLM fiyat tablolarini tek formata
normalize etmeye yardim eden helper'lar vardir.

Manual load/dump:

```python
from pathlib import Path
from autobench import load_pricing_table, dump_pricing_table

table = load_pricing_table(Path("pricing/models.yaml"))
dump_pricing_table(table, Path("pricing/normalized.yaml"))
```

Static source:

```python
from autobench import StaticPriceSource

source = StaticPriceSource(Path("pricing/models.yaml"))
table = source.load()
```

LLMPrices / GenAIPrices source adapter'lari:

```python
from autobench import GenAIPricesSource, LLMPricesSource

genai_table = GenAIPricesSource(Path("genai-prices.json")).load()
llm_prices_table = LLMPricesSource(Path("llm-prices.json")).load()
```

Model id normalizasyonu:

```text
provider:model
provider/model
model with provider field
aliases
```

Fikir su: kullanici isterse fiyatlari elle yazar, isterse source adapter ile
normalize eder. Autobench kendi basina pricing ownership almaz.

---

## 27. Feedback Records

Autobench recorded run'dan optimization icin feedback payload'u da uretmeye
baslar.

```python
from pathlib import Path
from autobench import build_optimization_feedback_input, load_run_record

record = load_run_record(Path("runs/demo/cases/case_1/variant_1/run.yaml"))
feedback = build_optimization_feedback_input(record)
```

`FeedbackRecord` alanlari:

```text
case_id
variant_id
score_name
semantic_type
score
passed
failure_category
reason
span
```

Basarili case icin `failure_category` `None` olur. Error varsa `error`,
assertion fail varsa `assertion_failure`, low score varsa `low_score` gibi
kategoriler kullanilir.

Bu parca ileride autoptimize icin onemli olacak. Ama v0.1'de sadece evidence'dan
feedback input uretme seviyesindedir.

---

## 28. Run Status Model

Autobench farkli status katmanlarini ayirir.

Task status:

```text
passed
failed
errored
skipped
```

Evaluation status:

```text
passed
failed
errored
skipped
```

Run status:

```text
passed
failed
errored
skipped
```

Neden ayri?

- Task tamamlanabilir ama scorer fail olabilir.
- Task tamamlanabilir ama policy fail olabilir.
- Task hic tanimli degilse skipped olabilir.
- Exception varsa errored olabilir.

Bu ayrim RunRecord'da gorunur:

```yaml
run:
  status: failed
  outcome:
    evaluation: failed
    task: passed
```

Bu ornekte uygulama calismis, ama evaluation basarisiz olmustur.

---

## 29. YAML Record Ornegi

Bir run record kabaca soyle gorunur:

```yaml
record:
  type: run
  version: 4

protocol:
  name: abp
  version: 1
  semantic_registry: 1

run:
  id: run_refund_request_route_v2
  experiment: exp_support_routing_20260512T120000Z
  benchmark: support-routing
  case: refund_request
  variant: route_v2
  status: passed
  outcome:
    evaluation: passed
    task: passed

case:
  id: refund_request
  input:
    subject: Duplicate card charge
    body: I was billed twice yesterday and need a refund.
  expected:
    queue: billing
  tags:
    - billing
    - refund

variant:
  id: route_v2
  label: improved routing prompt
  factors:
    prompt_version:
      value: route-v2
      semantic: prompt.version
      optimize: true

scores:
  matched:
    value: true
    semantic: result.success
    role: objective
  queue_correct:
    value: 1.0
    semantic: quality.correctness
    actual: billing
    expected: billing

metrics:
  objectives:
    matched:
      value: true
      semantic: result.success
    queue_correct:
      value: 1.0
      semantic: quality.correctness
  diagnostics:
    confidence:
      value: 0.94
      semantic: quality.score
    routing_duration:
      value: 0.0021
      semantic: time.latency
      unit: s

spans:
  route_ticket:
    kind: workflow
    input:
      subject: Duplicate card charge
      body: I was billed twice yesterday and need a refund.
    output:
      queue: billing
      confidence: 0.94
      matched: true
    duration: 0.0021

output:
  queue: billing
  confidence: 0.94
  matched: true
```

Gercek dosyadaki alanlar daha ayrintili olabilir; ama DSL-like hedef budur:
insan okuyabilsin, makine de deserialize edebilsin.

---

## 30. End-to-End Ornek: LLM Reply Quality Benchmark

Bu ornek bir LLM reply generator'in kalite, maliyet, latency ve policy
kontrollerini ayni benchmark'ta toplar.

### 30.1 Task

`app/reply.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class FakeResult:
    output: str
    usage: FakeUsage


def generate_reply(prompt: str, *, model: str, tone: str) -> FakeResult:
    if tone == "concise":
        output = "We can help with that. Please share your order id."
    else:
        output = (
            "Thanks for reaching out. We can help with that refund request. "
            "Please share your order id so our billing team can review it."
        )
    return FakeResult(
        output=output,
        usage=FakeUsage(
            input_tokens=len(prompt.split()) + 20,
            output_tokens=len(output.split()),
            total_tokens=len(prompt.split()) + len(output.split()) + 20,
        ),
    )
```

`app/benchmarks/reply.py`:

```python
from __future__ import annotations

from autobench import DurationMetricSpec, PydanticAIUsage, Semantic, SpanKind, record_pydantic_ai_usage

from app.reply import generate_reply


def run_reply_case(ctx, case):
    model = ctx.factor("model")
    tone = ctx.factor("tone")

    with ctx.span(
        "reply_generation",
        kind=SpanKind.LLM,
        input=case.input,
        attributes={"model": model, "provider": "demo"},
        duration_metric=DurationMetricSpec(name="reply_latency"),
    ) as span:
        prompt = case.input["message"]
        result = generate_reply(prompt, model=model, tone=tone)
        span.set_output(result.output)
        span.set_usage("input_tokens", result.usage.input_tokens)
        span.set_usage("output_tokens", result.usage.output_tokens)

        record_pydantic_ai_usage(
            ctx,
            PydanticAIUsage(
                requests=1,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                total_tokens=result.usage.total_tokens,
                model_name=model,
                provider="demo",
            ),
            span_id=span.id,
        )

    contains_required_phrase = case.expected["required_phrase"] in result.output.lower()
    ctx.check(
        "required_phrase_present",
        contains_required_phrase,
        semantic_type=Semantic.QUALITY_CORRECTNESS,
    )

    return {
        "reply": result.output,
        "contains_required_phrase": contains_required_phrase,
    }
```

### 30.2 Pricing

`pricing/demo-models.yaml`:

```yaml
pricing:
  provider: demo
  source: manual
  models:
    demo/small:
      input:
        unit: mtok
        price: 0.1
      output:
        unit: mtok
        price: 0.3
    demo/large:
      input:
        unit: mtok
        price: 1.0
      output:
        unit: mtok
        price: 3.0
```

### 30.3 Spec

`reply_benchmark.yaml`:

```yaml
benchmark:
  reply-quality:
    description: Compare reply generation quality, latency, and cost.

    dataset:
      cases:
        - id: refund_case
          input:
            message: I need a refund for a duplicate charge.
          expected:
            required_phrase: order id

        - id: account_case
          input:
            message: I cannot log into my account.
          expected:
            required_phrase: help

    run:
      python: app.benchmarks.reply:run_reply_case

    variants:
      reply_v1:
        label: concise reply
        factors:
          model:
            value: demo/small
            semantic: llm.model.name
            optimize: true
          provider:
            value: demo
            semantic: llm.provider
          tone:
            value: concise

      reply_v2:
        label: fuller reply
        factors:
          model:
            value: demo/large
            semantic: llm.model.name
            optimize: true
          provider:
            value: demo
            semantic: llm.provider
          tone:
            value: helpful
            optimize: true

    score:
      success:
        pass: output.contains_required_phrase
        semantic: result.success
        role: objective
        goal: maximize

      phrase_correctness:
        exact:
          actual: output.contains_required_phrase
          expected: true
        semantic: quality.correctness
        role: objective
        goal: maximize

    derive:
      - kind: token_cost
        pricing: pricing/demo-models.yaml

    policies:
      - name: must_pass
        metric: result.success
        must_equal: true
      - name: cost_ceiling
        metric: money.cost
        must_less_equal: 0.01

    report:
      leaderboard:
        show:
          pass_rate:
            metric: result.success
            aggregate: ratio_true
          avg_correctness:
            metric: quality.correctness
            aggregate: mean
          total_cost:
            metric: money.cost
            aggregate: sum
          avg_latency:
            metric: time.latency
            aggregate: mean
      matrix:
        metric: quality.correctness
      compare:
        reply_v1 -> reply_v2:
          show:
            avg_correctness:
              metric: quality.correctness
              aggregate: mean
            total_cost:
              metric: money.cost
              aggregate: sum
```

### 30.4 Run

```bash
uv run autobench validate reply_benchmark.yaml
uv run autobench run reply_benchmark.yaml --record runs/reply-quality
uv run autobench report runs/reply-quality
uv run autobench export runs/reply-quality --format markdown --path runs/reply-quality/report.md
uv run autobench export runs/reply-quality --format csv --path runs/reply-quality/runs.csv
```

Bu benchmark'ta ayni anda:

- success
- correctness
- token usage
- derived cost
- latency
- policy result
- variant comparison
- replayable run records

toplanir.

---

## 31. Autobench Bir Uygulamaya Nasil Entegre Edilir?

Pratik migration sirasi:

1. Mevcut `run_benchmark.py` scriptindeki scenario listesini `Case` haline getir.
2. Model/prompt/tool/config kombinasyonlarini `Variant` ve `FactorValue` haline getir.
3. Eski `run_one_scenario(...)` fonksiyonunu `def run_case(ctx, case)` signature'ina uyarla.
4. Eski manuel metric dictionary'lerini `ctx.metric(...)` veya scorer spec'lerine tasi.
5. Eski timer kodunu `ctx.span(... duration_metric=...)` veya `measure_callable` ile degistir.
6. Eski cost hesaplarini raw token observation + `token_cost` deriver'a ayir.
7. Eski summary table'i `report.leaderboard` ve `report.matrix` ile tanimla.
8. Eski baseline/candidate custom compare kodunu `post_derive: paired_baseline` ile ifade et.
9. Prompt/tool/schema gibi degisen varliklari `track.prompt`, `track.tool`, `track.type` ile takip et.
10. `autobench run --record` ile immutable evidence yaz.

Eski script:

```python
for scenario in scenarios:
    result = run_app(scenario)
    rows.append(
        {
            "scenario": scenario.name,
            "success": result.ok,
            "latency": result.latency,
            "cost": result.cost,
        }
    )
```

Autobench task:

```python
from autobench import Semantic


def run_case(ctx, case):
    with ctx.span("app_run", duration_metric={"name": "latency"}) as span:
        result = run_app(case.input)
        span.set_output(result)

    ctx.metric("cost", result.cost, semantic_type=Semantic.MONEY_COST, unit="usd")
    ctx.outcome(result.ok)
    return {"ok": result.ok, "answer": result.answer}
```

Autobench YAML:

```yaml
benchmark:
  app-benchmark:
    cases: cases.yaml
    run:
      python: app.benchmarks:run_case
    variants:
      default: {}
    score:
      success:
        pass: output.ok
        semantic: result.success
    report:
      leaderboard:
        show:
          pass_rate:
            metric: result.success
            aggregate: ratio_true
          total_cost:
            metric: money.cost
            aggregate: sum
          avg_latency:
            metric: time.latency
            aggregate: mean
```

---

## 32. Hata Ve Edge Case Davranislari

### 32.1 Task import edilemezse

Spec:

```yaml
run:
  python: missing.module:run_case
```

Autobench run'u structured error ile isaretler. CLI hata detayini Rich panelde
gosterir. Replay icin task import gerekmedigi icin eski kayitlar bundan
etkilenmez.

### 32.2 Scorer path bulunamazsa

```yaml
score:
  bad:
    value: output.not_here
    semantic: quality.score
```

Scorer `ScoreRecord.error` uretir. Optional degilse evaluation fail olabilir.

Optional scorer:

```yaml
score:
  optional_debug_metric:
    value: output.debug.score
    semantic: quality.score
    optional: true
```

### 32.3 Cost input eksikse

`token_cost` deriver gerekli semantic input'lari bulamazsa cost uydurmaz:

```text
token_cost_missing_inputs
```

Diagnostic observation yazar.

### 32.4 Paired baseline match yoksa

Candidate run icin baseline match bulunamazsa:

```text
paired_baseline_unavailable
```

Diagnostic observation yazar. `missing: skip` dersen sessizce atlar.

### 32.5 Policy metric yoksa

Policy result fail olur ve reason:

```text
missing_metric
```

---

## 33. Ne Zaman Task Observation, Ne Zaman Scorer?

Kural basit:

Task observation:

- runtime'da zaten dogal olarak ortaya cikan sey
- token usage
- latency
- retrieved doc count
- selected model
- tool call count
- raw confidence
- artifact/trace/debug data

Scorer:

- output'u expected ile karsilastiran sey
- pass/fail
- exact match
- schema validity
- rubric
- judge
- expected tool action correctness

Deriver:

- var olan metric/factor'lerden hesaplanan sey
- token -> cost
- latency baseline/candidate -> speedup
- candidate-baseline -> delta

Policy:

- release gate
- "quality >= 0.8"
- "cost <= 0.002"
- "success must be true"

Bu ayrim temiz tutulursa rapor ve autoptimize icin evidence daha guvenilir olur.

---

## 34. Autobench'in Simdiki Sinirlari

Bu dosya mevcut durumu anlatiyor. Su an olmayan seyleri de net soylemek lazim:

- Chart/image export yok.
- Dashboard yok.
- Hosted report sharing yok.
- Full OpenTelemetry bridge yok.
- Logfire/DataDog export yok.
- Full autoptimize yok.
- Full pydantic-gepa optimizer pipeline Autobench core icinde yok.
- Causal attribution engine yok.
- Distributed execution yok.
- Remote dataset URL loading yok.
- YAML icinde inline Python expression yok.

Bunlarin bazilari bilerek yok. Ornegin YAML icinde:

```yaml
value_factory: len(result.output)
```

gibi expression calistirmak v0.1 yuzeyinde yoktur. YAML shareable ve guvenli
kalir. Custom logic gerekiyorsa Python scorer veya task fonksiyonu yazilir.

---

## 35. Sonuc

Autobench bugun sunu sagliyor:

```text
one-off benchmark script
  -> YAML-first experiment definition
  -> semantic observations
  -> scored/derived/policy-checked runs
  -> immutable evidence records
  -> replayable reports
  -> optimization-ready data
```

Framework'un cekirdek iddiasi sudur:

> Kullanici her uygulama icin bastan benchmark runner yazmak zorunda kalmamali.
> Case, variant, task, metric, score, derivation, policy, record ve report
> primitive'leri bir framework tarafindan saglanmali.

AI sistemleri icin bu su demek:

- model karsilastirma
- prompt version etkisi
- tool call kalitesi
- structured output validity
- token/cost tracking
- latency/cost/quality tradeoff
- replayable evidence
- ileride autoptimize icin candidate feedback

AI disi sistemler icin bu su demek:

- performans benchmark'i
- correctness regression
- algorithm variant karsilastirmasi
- policy gate
- cost/latency/throughput raporu
- baseline/candidate speedup

Autobench'in su anki hali production-ready bir platform degil; ama dogru
primitive'leri olan typed, semantic, replayable bir benchmark evidence layer'dir.
