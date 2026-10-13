"""Lakebase connection pool with OAuth token refresh.

Deployed as a Databricks App, the platform injects PGHOST/PGUSER/PGDATABASE and
LAKEBASE_ENDPOINT (from the `postgres` app resource) and the app's service
principal credentials. Locally, set DATABRICKS_CONFIG_PROFILE and LAKEBASE_ENDPOINT
and the rest is resolved from the SDK.
"""
import os
import threading
import time

import psycopg
from psycopg_pool import ConnectionPool

from .clients import w

TOKEN_TTL_S = 45 * 60  # tokens last 1h; refresh well before expiry

ENDPOINT = os.environ.get(
    "LAKEBASE_ENDPOINT", "projects/lakebase-rag-demo/branches/production/endpoints/primary"
)

_token = {"value": None, "at": 0.0}
_token_lock = threading.Lock()


def _get_token() -> str:
    with _token_lock:
        if not _token["value"] or time.time() - _token["at"] > TOKEN_TTL_S:
            _token["value"] = w.postgres.generate_database_credential(endpoint=ENDPOINT).token
            _token["at"] = time.time()
        return _token["value"]


class OAuthConnection(psycopg.Connection):
    """Injects a current OAuth token as the password on every new connection."""

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs):
        kwargs["password"] = _get_token()
        return super().connect(conninfo, **kwargs)


def _conninfo() -> dict:
    host = os.environ.get("PGHOST") or w.postgres.get_endpoint(name=ENDPOINT).status.hosts.host
    return dict(
        host=host,
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "databricks_postgres"),
        user=os.environ.get("PGUSER") or w.current_user.me().user_name,
        sslmode="require",
    )


def _configure(conn):
    # load pgvector in each session so the hnsw.* settings on the search functions apply
    try:
        conn.execute("SELECT '[1]'::vector")
    except psycopg.errors.UndefinedObject:
        pass


pool = ConnectionPool(
    connection_class=OAuthConnection,
    kwargs={**_conninfo(), "autocommit": True},
    min_size=1,
    max_size=8,
    max_lifetime=TOKEN_TTL_S,  # recycle connections before their token would matter
    max_idle=240,  # retire idle connections before Lakebase scales to zero (300 s) and drops them
    check=ConnectionPool.check_connection,  # validate on checkout so the first request after a pause doesn't fail
    configure=_configure,
    open=False,
)
