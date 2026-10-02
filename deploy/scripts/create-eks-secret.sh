#!/usr/bin/env bash
# Creates the Kubernetes Secret the chart reads, pulling the RDS password from
# Secrets Manager so it never touches your shell history or the repo.
# Run from the repo root after `terraform apply`:
#   ANTHROPIC_API_KEY=... LANGSMITH_API_KEY=... API_AUTH_TOKEN=... deploy/scripts/create-eks-secret.sh
set -euo pipefail

TF_DIR="deploy/terraform/aws"
NAMESPACE="${NAMESPACE:-defect-triage}"

region=$(terraform -chdir="$TF_DIR" output -raw region)
db_host=$(terraform -chdir="$TF_DIR" output -raw db_endpoint)
secret_arn=$(terraform -chdir="$TF_DIR" output -raw db_master_secret_arn)

db_password=$(aws secretsmanager get-secret-value --region "$region" --secret-id "$secret_arn" \
  --query SecretString --output text | python3 -c 'import json,sys; print(json.load(sys.stdin)["password"])')
encoded_password=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$db_password")

kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"

kubectl -n "$NAMESPACE" create secret generic defect-triage-secrets \
  --from-literal=DATABASE_URL="postgresql://triage:${encoded_password}@${db_host}:5432/triage?sslmode=require" \
  --from-literal=ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  --from-literal=LANGSMITH_API_KEY="${LANGSMITH_API_KEY:-}" \
  --from-literal=API_AUTH_TOKEN="${API_AUTH_TOKEN:-}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "secret defect-triage-secrets is set in namespace $NAMESPACE"
