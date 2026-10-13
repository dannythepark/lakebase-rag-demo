"""Grant the app's service principal read access to the content (and market) schemas in Lakebase.

Run once after the first `databricks bundle deploy` (the app SP's Postgres role is created when the app's
`postgres` resource is attached). The SP owns the `app` schema itself because it creates it on startup.

  python scripts/grant_app_access.py --profile <profile> --app content-assistant
"""
import argparse

import psycopg
from databricks.sdk import WorkspaceClient

ap = argparse.ArgumentParser()
ap.add_argument("--profile", default=None)
ap.add_argument("--app", default="content-assistant")
ap.add_argument("--endpoint", default="projects/lakebase-rag-demo/branches/production/endpoints/primary")
args = ap.parse_args()

w = WorkspaceClient(profile=args.profile)
sp = w.apps.get(args.app).service_principal_client_id
ep = w.postgres.get_endpoint(name=args.endpoint)
token = w.postgres.generate_database_credential(endpoint=ep.name).token

with psycopg.connect(host=ep.status.hosts.host, dbname="databricks_postgres", user=w.current_user.me().user_name,
                     password=token, sslmode="require", autocommit=True) as conn:
    roles = [r[0] for r in conn.execute("SELECT rolname FROM pg_roles WHERE rolname = %s", (sp,))]
    if not roles:
        raise SystemExit(f"Postgres role {sp} not found yet. Deploy/start the app first.")
    for schema in ("content", "market"):
        if not conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)).fetchone():
            print(f"skip {schema} (not created yet)")
            continue
        conn.execute(f'GRANT USAGE ON SCHEMA {schema} TO "{sp}"')
        conn.execute(f'GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO "{sp}"')
        conn.execute(f'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA {schema} TO "{sp}"')
        conn.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT SELECT ON TABLES TO "{sp}"')
        print(f"granted {schema} to {sp}")

    # If someone ran the app locally before deploying, `app` is owned by that user instead of the SP.
    # Grant the SP full use of it (non-destructive) rather than dropping it.
    owner = conn.execute("SELECT nspowner::regrole::text FROM pg_namespace WHERE nspname = 'app'").fetchone()
    if owner and owner[0].strip('"') != sp:
        conn.execute(f'GRANT USAGE, CREATE ON SCHEMA app TO "{sp}"')
        conn.execute(f'GRANT ALL ON ALL TABLES IN SCHEMA app TO "{sp}"')
        conn.execute(f'GRANT ALL ON ALL SEQUENCES IN SCHEMA app TO "{sp}"')
        conn.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON TABLES TO "{sp}"')
        conn.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON SEQUENCES TO "{sp}"')
        print(f"app schema owned by {owner[0]}; granted full access to {sp}")
