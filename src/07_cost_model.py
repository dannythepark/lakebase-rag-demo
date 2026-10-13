# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · What did this demo actually cost? (list price, from system tables)
# MAGIC
# MAGIC Cost transparency matters as much as features. This notebook reads
# MAGIC `system.billing.usage` and joins `system.billing.list_prices`, filtered to **only this demo's resources**:
# MAGIC
# MAGIC | Product | Filter |
# MAGIC |---|---|
# MAGIC | Lakebase | `usage_metadata.project_id` = this project's UID |
# MAGIC | Databricks App | `usage_metadata.app_name` |
# MAGIC | Jobs (serverless) and `ai_query` | `usage_metadata.job_id` in the demo jobs |
# MAGIC | Foundation Model API (app + notebooks) | `usage_metadata.endpoint_name` in the demo's endpoints, run by me or the app's SP |
# MAGIC
# MAGIC Run in your own workspace (AWS, Azure or GCP), the same notebook shows that cloud's list prices.
# MAGIC Billing data lands with a delay of a few hours.

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]>=3.2"

# COMMAND ----------

from datetime import date, timedelta

from databricks.sdk import WorkspaceClient

from rag_common import get_params, Lakebase

p = get_params(dbutils)
w = WorkspaceClient()
dbutils.widgets.text("app_name", "content-assistant")
dbutils.widgets.text("since", str(date.today() - timedelta(days=7)))
APP, SINCE = dbutils.widgets.get("app_name"), dbutils.widgets.get("since")

project_uid = Lakebase(w).get(f"projects/{p['lakebase_project']}")["uid"]
job_ids = [str(j.job_id) for j in w.jobs.list() if "Lakebase RAG Demo" in (j.settings.name or "")]
endpoints = [p["embedding_endpoint"], p["alt_embedding_endpoint"], p["llm_endpoint"], p["partner_llm_endpoint"]]
identities = [w.current_user.me().user_name]
try:
    identities.append(w.apps.get(APP).service_principal_client_id)
except Exception:
    pass
ws_id = str(w.get_workspace_id())
print(dict(project_uid=project_uid, job_ids=job_ids, endpoints=endpoints, identities=identities, workspace=ws_id))


def sql_list(xs):
    return ", ".join(f"'{x}'" for x in xs) or "''"


spark.sql(f"""
CREATE OR REPLACE TEMP VIEW demo_usage AS
SELECT u.usage_date, u.usage_start_time, u.billing_origin_product, u.sku_name, u.usage_unit, u.usage_quantity,
       u.usage_quantity * lp.pricing.default AS list_usd,
       CASE
         WHEN u.usage_metadata.project_id = '{project_uid}' THEN 'Lakebase (lakebase-rag-demo)'
         WHEN u.usage_metadata.app_name = '{APP}' THEN 'Databricks App'
         WHEN u.usage_metadata.job_id IN ({sql_list(job_ids)}) AND u.billing_origin_product = 'AI_FUNCTIONS'
              THEN 'ai_query in jobs (generation + embeddings)'
         WHEN u.usage_metadata.job_id IN ({sql_list(job_ids)}) THEN 'Serverless jobs compute'
         ELSE concat('Model serving: ', u.usage_metadata.endpoint_name)
       END AS component
FROM system.billing.usage u
LEFT JOIN system.billing.list_prices lp
  ON u.sku_name = lp.sku_name AND u.cloud = lp.cloud
 AND u.usage_start_time >= lp.price_start_time
 AND (lp.price_end_time IS NULL OR u.usage_start_time < lp.price_end_time)
WHERE u.workspace_id = '{ws_id}'
  AND u.usage_date >= '{SINCE}'
  AND (   u.usage_metadata.project_id = '{project_uid}'
       OR u.usage_metadata.app_name = '{APP}'
       OR u.usage_metadata.job_id IN ({sql_list(job_ids)})
       OR (u.usage_metadata.endpoint_name IN ({sql_list(endpoints)})
           AND coalesce(u.identity_metadata.run_as, u.identity_metadata.created_by) IN ({sql_list(identities)})))
""")

display(spark.sql("""
SELECT component, sku_name, round(sum(usage_quantity), 3) AS quantity, first(usage_unit) AS unit,
       round(sum(list_usd), 2) AS list_usd
FROM demo_usage GROUP BY component, sku_name ORDER BY list_usd DESC
"""))

# COMMAND ----------

# MAGIC %md ## One-time build vs. ongoing serving
# MAGIC Generating the synthetic articles (`ai_query` with an LLM) is a **demo-only** cost. A real publisher's content
# MAGIC already exists and would only be embedded.

# COMMAND ----------

display(spark.sql("""
SELECT CASE WHEN component LIKE 'ai_query%' OR component = 'Serverless jobs compute' THEN 'one-time build (demo data)'
            ELSE 'serving (Lakebase, app, live model calls)' END AS bucket,
       round(sum(list_usd), 2) AS list_usd
FROM demo_usage GROUP BY 1
"""))

# COMMAND ----------

# MAGIC %md ## Lakebase: scale to zero in practice
# MAGIC DBUs by hour. Hours with nothing billed are hours the compute was suspended.

# COMMAND ----------

display(spark.sql("""
SELECT date_trunc('HOUR', usage_start_time) AS hour, round(sum(usage_quantity), 4) AS dbu, round(sum(list_usd), 4) AS list_usd
FROM demo_usage WHERE component LIKE 'Lakebase%' GROUP BY 1 ORDER BY 1
"""))
