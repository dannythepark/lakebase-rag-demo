"""Shared helpers for the Lakebase RAG demo notebooks.

Every model name, size, and resource name lives in DEFAULTS. Notebooks override
them with job parameters / widgets, so swapping the embedding model or LLM is a
config change, not a code change.
"""
import time

import psycopg
from databricks.sdk import WorkspaceClient

DEFAULTS = {
    "catalog": "demo_catalog",
    "schema": "rag_demo",
    "lakebase_project": "lakebase-rag-demo",
    "lakebase_branch": "production",
    "lakebase_endpoint": "primary",
    "pg_database": "databricks_postgres",
    "embedding_endpoint": "databricks-gte-large-en",
    "embedding_dim": "1024",
    "alt_embedding_endpoint": "databricks-qwen3-embedding-0-6b",
    "llm_endpoint": "databricks-claude-haiku-4-5",
    "partner_llm_endpoint": "databricks-claude-sonnet-4-6",
    "n_articles": "3000",
    "n_companies": "4500000",
}


def get_params(dbutils):
    """Read widgets (job parameters), falling back to DEFAULTS."""
    params = {}
    for key, default in DEFAULTS.items():
        try:
            dbutils.widgets.text(key, default)
            params[key] = dbutils.widgets.get(key) or default
        except Exception:
            params[key] = default
    return params


def endpoint_path(p):
    return (
        f"projects/{p['lakebase_project']}/branches/{p['lakebase_branch']}"
        f"/endpoints/{p['lakebase_endpoint']}"
    )


class Lakebase:
    """Minimal Lakebase REST client.

    Uses the SDK's authenticated api_client against /api/2.0/postgres directly, so the
    notebooks work with whatever databricks-sdk the serverless environment ships
    (no SDK upgrade, so no Python restart).
    """

    def __init__(self, w=None):
        self.w = w or WorkspaceClient()

    def get(self, name):
        return self.w.api_client.do("GET", f"/api/2.0/postgres/{name}")

    def exists(self, name):
        try:
            self.get(name)
            return True
        except Exception as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower() or "404" in str(e):
                return False
            raise

    def create(self, collection, id_param, id_value, body, timeout_s=1800):
        """POST a create request and wait for the long-running operation to finish."""
        op = self.w.api_client.do(
            "POST", f"/api/2.0/postgres/{collection}", query={id_param: id_value}, body=body
        )
        deadline = time.time() + timeout_s
        while not op.get("done"):
            if time.time() > deadline:
                raise TimeoutError(f"operation {op.get('name')} not done after {timeout_s}s")
            time.sleep(10)
            op = self.get(op["name"])
        if op.get("error"):
            raise RuntimeError(op["error"])
        return op.get("response")

    def host(self, endpoint):
        return self.get(endpoint)["status"]["hosts"]["host"]

    def token(self, endpoint):
        return self.w.api_client.do("POST", "/api/2.0/postgres/credentials", body={"endpoint": endpoint})["token"]


def pg_connect(p, w=None, autocommit=True):
    """Open a psycopg connection to Lakebase with a fresh OAuth token (valid 1h)."""
    w = w or WorkspaceClient()
    lb = Lakebase(w)
    ep = endpoint_path(p)
    conn = psycopg.connect(
        host=lb.host(ep),
        dbname=p["pg_database"],
        user=w.current_user.me().user_name,
        password=lb.token(ep),
        sslmode="require",
        autocommit=autocommit,
    )
    load_pgvector(conn)
    return conn


def load_pgvector(conn):
    """Touch a vector value so the pgvector library (and its hnsw.* settings) is loaded in this session."""
    try:
        conn.execute("SELECT '[1]'::vector")
    except psycopg.errors.UndefinedObject:
        pass  # extension not created yet


def vec_literal(embedding):
    """pgvector text format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


def embed(w, endpoint, texts):
    """Embed a list of strings with a Databricks-hosted embedding endpoint."""
    resp = w.api_client.do(
        "POST", f"/serving-endpoints/{endpoint}/invocations", body={"input": texts}
    )
    return [d["embedding"] for d in sorted(resp["data"], key=lambda d: d["index"])]


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000
