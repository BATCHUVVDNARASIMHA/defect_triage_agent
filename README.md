# Defect Triage Agent

A multi-agent LangGraph system that triages manufacturing quality-inspection
reports, deliberately built to reproduce and fix the same class of grounding
bug found in production RAG/agent systems — where an AI trusts a
plausible-sounding but outdated document over the actual authoritative
source.

Built and tested August 2026. Deployment layer added October 2026: an HTTP
API, pgvector retrieval with metadata filtering and citations, LangSmith
tracing and evaluation, Docker, CI with an evaluation release gate, a Helm
chart, and Terraform for AWS EKS.

## Architecture

```
raw inspection note
        |
        v
  [intake_agent]            -- parses free text into a structured defect_type
        |
        +--(not a known defect)--> [unrecognized_defect] --> [human_escalation]
        |
        v
  [risk_assessment_agent]   -- retrieves evidence, determines severity tier
        |                       (this is where the grounding bug lives)
        v
  [routing_agent]           -- determines disposition route + review flag
        |
        v
  [report_agent]            -- drafts the inspector-facing report, with citation
        |
        v
  conditional edge
        |
   +----+----+
   |         |
[escalate] [finish]
   |         |
   v         v
[human_    [END]
escalation]
   |
   v
 [END]
```

Five specialized agents, a stateful graph, tool calls, and conditional edges
that route critical, review-required, or unclassifiable cases to a human gate
instead of auto-finalizing. Built with LangGraph + LangChain.

## The grounding bug (the point of the project)

`risk_assessment_agent` retrieves evidence from two sources when it decides a
defect's severity:

- **`data/severity_policy.json`** — the authoritative policy. Always
  correct, always current.
- **`data/reference_guide.json`** — a plausible-sounding new-hire reference
  doc that reads naturally but is out of date and disagrees with the policy
  on every single defect type.

Both sources are chunked, embedded, and stored in a vector store (pgvector in
deployment) with metadata: source, an `authoritative` flag, defect type, and
document version.

In **buggy mode**, retrieval is unfiltered, so both sources land in the
model's context with no instruction about which to trust. In **fixed mode**,
the fix has two layers:

1. **Retrieval:** a metadata filter (`authoritative = true`) keeps the
   outdated guide out of the context entirely.
2. **Prompt:** the model is told the policy is authoritative and must cite the
   chunk id it relied on.

This is the same shape of failure as an AI customer-support agent citing a
marketing document's warranty claim instead of the actual return-policy
tool — except this version, including the data, the graph, and the eval
harness, was built from scratch rather than following a workshop template.

## Results (reproducible — see below)

Ran the full 6-case eval set in both modes, scored against ground truth,
logged to MLflow:

| Mode  | Severity accuracy | Route accuracy | Human-review accuracy | Cites authoritative source |
|-------|-------------------|----------------|-----------------------|----------------------------|
| Buggy | 0%                | 100%           | 100%                  | 0%                         |
| Fixed | 100%              | 100%           | 100%                  | 100%                       |

The buggy mode isn't just "worse" — it's wrong on every single severity
call, because the reference guide's phrasing consistently contradicts the
real policy (it downplays a genuinely critical surface crack as "cosmetic,"
and overcorrects a minor edge nick to "always scrap"). Route and human-review
accuracy hold at 100% in both modes on this eval set — the routing and
escalation logic downstream isn't what's broken, the severity call feeding
into it is. That's deliberate: a project that shows a small improvement is
a weaker proof than one that shows the failure mode is real and the fix is
complete.

CI now enforces this: `python -m src.eval_gate` fails the build if any fixed-mode
metric drops below 100%, so a prompt, retrieval, or model change that
reintroduces the bug can't merge.

## Running it yourself

### 1. Locally, no infrastructure

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

# Mock mode (no API key needed) — deterministic:
python -m src.run_eval          # buggy vs fixed, logged to MLflow (mlflow ui)
pytest -q                       # test suite
python -m src.eval_gate         # the CI release gate

# Live mode — agents make real Claude API calls:
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.run_eval

# The API, with the in-memory vector store:
uvicorn src.api:app --reload
curl -s localhost:8000/v1/triage -H 'content-type: application/json' \
  -d '{"raw_report": "Hairline crack along the leading edge, 4mm long.", "mode": "fixed"}'
```

### 2. Docker Compose: Postgres + pgvector, ingest job, API

```bash
cp .env.example .env            # optional: ANTHROPIC_API_KEY, LANGSMITH_API_KEY
docker compose up --build
curl -s localhost:8000/readyz
curl -sN localhost:8000/v1/triage/stream -H 'content-type: application/json' \
  -d '{"raw_report": "Bluish heat tint near the blade root.", "mode": "fixed"}'
```

### 3. Local Kubernetes with kind

Needs `kind`, `kubectl`, `helm`, and Docker.

```bash
make kind-up        # cluster, image load, dev pgvector, secret, Helm release
kubectl -n defect-triage port-forward svc/defect-triage 8080:80
make kind-down
```

### 4. AWS EKS

Terraform in `deploy/terraform/aws` creates a VPC, an EKS cluster with a
managed node group, an ECR repository, an encrypted RDS Postgres 16 instance
(pgvector extension) whose password RDS keeps in Secrets Manager, and a GitHub
Actions OIDC role so no AWS keys are stored in GitHub.

**This costs money while it runs** (roughly the EKS control plane, two
t3.medium nodes, a NAT gateway, and a small RDS instance; check current AWS
pricing). Apply it, deploy, and destroy it when you're done.

```bash
cd deploy/terraform/aws
cp terraform.tfvars.example terraform.tfvars
terraform init && terraform apply
$(terraform output -raw configure_kubectl)
cd ../../..

# Secret for the app (pulls the RDS password from Secrets Manager):
ANTHROPIC_API_KEY=... LANGSMITH_API_KEY=... API_AUTH_TOKEN=... deploy/scripts/create-eks-secret.sh

# HPA needs metrics-server:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

Then in the GitHub repo settings, add the secret `AWS_ROLE_ARN`
(`terraform output github_actions_role_arn`) and the variables `AWS_REGION`,
`EKS_CLUSTER_NAME`, and `ECR_REPOSITORY`, and run the **deploy-eks** workflow.
It builds the image, pushes it to ECR, and runs `helm upgrade --install
--atomic`, which rolls back automatically if the new pods don't become ready.

Tear down: `helm -n defect-triage uninstall defect-triage`, then
`terraform -chdir=deploy/terraform/aws destroy`.

### LangSmith

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`. Every run is traced: the
LangGraph nodes, the retrieval step (which chunks reached the model, with
scores and metadata), and each Claude call (prompt, output, tokens, latency).

```bash
python -m src.langsmith_eval    # buggy and fixed as two LangSmith experiments
```

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Liveness |
| `GET /readyz` | Readiness: vector store reachable and loaded |
| `POST /v1/triage` | Full disposition with citations and trace |
| `POST /v1/triage/stream` | Same, as server-sent events, one per agent |

Set `API_AUTH_TOKEN` to require an `X-API-Key` header. Reports are length-capped,
model output is validated (one re-ask, then a 502 rather than a malformed
answer), and transient API errors are retried with backoff.

## Project structure

```
defect_triage_agent/
  data/
    severity_policy.json    # authoritative source
    reference_guide.json    # the outdated "trap" source
    eval_set.json           # 6 test cases with ground truth
  src/
    graph.py                # the LangGraph state machine
    tools.py                # policy lookups + search_knowledge() retrieval tool
    retrieval.py            # chunking, embeddings, in-memory and pgvector stores, reranking
    ingest.py               # loads both sources into pgvector (idempotent)
    llm.py                  # Claude wrapper: retries, JSON validation, mock fallback
    observability.py        # LangSmith tracing helpers
    api.py                  # FastAPI service
    evaluation.py           # shared scoring
    run_eval.py             # buggy vs fixed eval, MLflow logging
    langsmith_eval.py       # the same eval as LangSmith experiments
    eval_gate.py            # CI release gate
  tests/                    # unit, API, and pgvector integration tests
  deploy/
    helm/defect-triage/     # Helm chart (values for default, kind, EKS)
    terraform/aws/          # VPC, EKS, ECR, RDS, GitHub OIDC role
    k8s/pgvector-dev.yaml   # dev-only Postgres for kind
    kind/kind-config.yaml
    scripts/create-eks-secret.sh
  notebooks/
    databricks_deploy_notebook.py   # Databricks-ready version, Unity
                                     # Catalog + CI/CD notes
  .github/workflows/
    ci.yml                  # tests, eval gate, chart + Terraform checks, image build
    deploy-eks.yml          # manual deploy to EKS
  Dockerfile
  docker-compose.yml
  Makefile
```

## Notes on framing this honestly

This is a **personal project**, built and dated August 2026, independent of
any employer or coursework. It is not part of the Honeywell Aerospace
capstone — the domain (manufacturing defect triage) is thematically
adjacent to that project's subject matter, but the two are unrelated
systems built at different times for different purposes, and should be
described as such if asked.

The deployment layer was added in October 2026 using AI-assisted development.
The EKS stack is reproducible from `deploy/terraform/aws`, but it costs money
to run, so it is meant to be applied, exercised, and destroyed rather than
left running.

Suggested resume line:

> **Defect Triage Agent (Personal Project)** — Aug 2026
> Built a 5-agent LangGraph workflow for manufacturing defect triage with
> conditional human-escalation routing. Reproduced and fixed a real
> grounding defect where the agent trusted an outdated reference document
> over the authoritative policy tool, proving the fix with an MLflow-logged
> eval showing severity-classification accuracy improve from 0% to 100%
> across a 6-case test set.

This gives you a fully ownable, fully defensible answer to "walk me through
your LangGraph/agentic workflow experience" — one you built yourself, that
you can open the code for, that you can re-run live in an interview if
asked.
