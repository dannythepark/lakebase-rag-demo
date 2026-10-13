"""Databricks SDK client plus thin wrappers for the embedding and chat endpoints."""
import os

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-claude-haiku-4-5")  # rewrite + coverage verdict
# advertiser-facing answers use a stronger model: accuracy on prices and comparisons matters most there
PARTNER_LLM_ENDPOINT = os.environ.get("PARTNER_LLM_ENDPOINT", "databricks-claude-sonnet-4-6")


def embed(text: str) -> list[float]:
    resp = w.api_client.do(
        "POST", f"/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations", body={"input": [text]}
    )
    return resp["data"][0]["embedding"]


def chat(messages: list[dict], max_tokens: int = 700, temperature: float = 0.2, endpoint: str | None = None) -> str:
    resp = w.api_client.do(
        "POST",
        f"/serving-endpoints/{endpoint or LLM_ENDPOINT}/invocations",
        body={"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
    )
    return resp["choices"][0]["message"]["content"]


def vec_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
