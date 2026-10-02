# Common tasks. Run `make help` for the list.
IMAGE ?= defect-triage-agent:local
NAMESPACE ?= defect-triage
RELEASE ?= defect-triage
KIND_CLUSTER ?= defect-triage
TF_DIR := deploy/terraform/aws

.PHONY: help install test gate eval run up down kind-up kind-down tf-init tf-plan tf-apply tf-destroy

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install everything locally (including MLflow)
	pip install -r requirements.txt -r requirements-dev.txt

test: ## Run the test suite (integration tests need TEST_DATABASE_URL)
	pytest -q

gate: ## Run the evaluation release gate used in CI
	python -m src.eval_gate

eval: ## Buggy vs fixed eval logged to MLflow
	python -m src.run_eval

run: ## Run the API locally with the in-memory vector store
	uvicorn src.api:app --reload

up: ## Postgres + pgvector, ingest, and the API with Docker Compose
	docker compose up --build

down: ## Stop the Compose stack and delete its volume
	docker compose down -v

kind-up: ## Local Kubernetes: kind cluster, dev pgvector, Helm release
	kind create cluster --name $(KIND_CLUSTER) --config deploy/kind/kind-config.yaml
	docker build -t $(IMAGE) .
	kind load docker-image $(IMAGE) --name $(KIND_CLUSTER)
	kubectl create namespace $(NAMESPACE)
	kubectl -n $(NAMESPACE) apply -f deploy/k8s/pgvector-dev.yaml
	kubectl -n $(NAMESPACE) rollout status statefulset/pgvector --timeout=180s
	kubectl -n $(NAMESPACE) create secret generic defect-triage-secrets \
	  --from-literal=DATABASE_URL=postgresql://triage:triage@pgvector:5432/triage \
	  --from-literal=ANTHROPIC_API_KEY=$${ANTHROPIC_API_KEY:-} \
	  --from-literal=LANGSMITH_API_KEY=$${LANGSMITH_API_KEY:-} \
	  --from-literal=API_AUTH_TOKEN=$${API_AUTH_TOKEN:-}
	helm upgrade --install $(RELEASE) deploy/helm/defect-triage -n $(NAMESPACE) \
	  -f deploy/helm/defect-triage/values-kind.yaml --wait --timeout 10m

kind-down: ## Delete the kind cluster
	kind delete cluster --name $(KIND_CLUSTER)

tf-init: ## terraform init for the AWS stack
	terraform -chdir=$(TF_DIR) init

tf-plan: ## terraform plan for the AWS stack
	terraform -chdir=$(TF_DIR) plan

tf-apply: ## Create the AWS stack (this starts billing)
	terraform -chdir=$(TF_DIR) apply

tf-destroy: ## Tear the AWS stack down (stops billing)
	terraform -chdir=$(TF_DIR) destroy
