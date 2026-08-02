# Databricks notebook source
# MAGIC %md
# MAGIC # Defect Triage Agent — Databricks Deployment
# MAGIC
# MAGIC This notebook deploys the Defect Triage Agent (built with LangGraph) on
# MAGIC Databricks, tracks evaluation runs in MLflow, and notes the Unity Catalog
# MAGIC governance pattern used for production deployment.
# MAGIC
# MAGIC **Architecture:** intake_agent -> risk_assessment_agent -> routing_agent ->
# MAGIC report_agent -> (conditional) human_escalation / END
# MAGIC
# MAGIC **What this demonstrates:** a real, measurable grounding bug (an agent
# MAGIC trusting an outdated reference document over the authoritative policy
# MAGIC tool) and a prompt-only fix, proven with an MLflow-tracked before/after
# MAGIC eval — 0% -> 100% severity accuracy across a 6-case eval set.

# COMMAND ----------

# MAGIC %pip install langgraph langchain-core mlflow anthropic

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog governance notes
# MAGIC
# MAGIC In a production deployment on Databricks, this agent's tool calls
# MAGIC (`get_severity_policy`, `get_reference_guide`) would be registered as
# MAGIC **Unity Catalog functions** rather than plain Python functions, so that:
# MAGIC
# MAGIC - Access to the underlying policy tables is governed by Unity Catalog
# MAGIC   permissions, not application-level code
# MAGIC - The agent runs **on-behalf-of (OBO)** the calling user, so it can only
# MAGIC   ever query the defect records that specific inspector/facility is
# MAGIC   permitted to see — the same pattern used in the Databricks Agent Apps
# MAGIC   Workshop for customer-support PII masking
# MAGIC - Every tool call is auditable in Unity Catalog's lineage view, which
# MAGIC   matters for aerospace/regulated-manufacturing environments where you
# MAGIC   need to prove *why* a part was scrapped or passed, not just *that* it was

# COMMAND ----------

import sys
import os

# When running as a Databricks Repo, the src/ package is importable directly.
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))

from src import graph as graph_module
from src.run_eval import load_eval_set, run_mode
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the MLflow experiment
# MAGIC
# MAGIC On Databricks, point this at a workspace path under your user folder,
# MAGIC e.g. `/Users/<you>/defect-triage-agent-eval`.

# COMMAND ----------

mlflow.set_experiment("/Users/<your-email>/defect-triage-agent-eval")

eval_set = load_eval_set()

results = {}
for mode in ["buggy", "fixed"]:
    summary = run_mode(mode, eval_set)
    results[mode] = summary

    with mlflow.start_run(run_name=f"defect_triage_{mode}"):
        mlflow.set_tag("mode", mode)
        mlflow.log_param("n_eval_cases", summary["n_cases"])
        mlflow.log_metric("severity_accuracy", summary["severity_accuracy"])
        mlflow.log_metric("route_accuracy", summary["route_accuracy"])
        mlflow.log_metric("human_review_accuracy", summary["human_review_accuracy"])
        mlflow.log_metric("overall_accuracy", summary["overall_accuracy"])

    print(f"{mode}: overall_accuracy = {summary['overall_accuracy']*100:.0f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## CI/CD notes — Databricks Asset Bundles
# MAGIC
# MAGIC To productionize this beyond a notebook, the project would be packaged as
# MAGIC a **Databricks Asset Bundle** (`databricks.yml`) defining:
# MAGIC
# MAGIC - A scheduled or triggered **job** that runs `src.run_eval` against a
# MAGIC   growing eval set every time the policy tables change, so a policy
# MAGIC   update can't silently break the agent's grounding
# MAGIC - A **Databricks App** deployment target for the live agent itself,
# MAGIC   serving real inspection requests rather than eval cases
# MAGIC - Environment-specific target configs (`dev` / `staging` / `prod`) so the
# MAGIC   same bundle deploys consistently instead of being copied by hand
# MAGIC
# MAGIC This mirrors the CI/CD pattern reviewed in the Databricks Agent Apps
# MAGIC Workshop, applied to a self-authored project instead of the workshop
# MAGIC template.
