# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Market intelligence: 4.5M company/fleet records served from Lakebase
# MAGIC
# MAGIC **The pain point:** questions like *"how many companies in Texas run 50–200 vehicles?"* over about 4.5M records were slow
# MAGIC unless they ran a large serverless SQL warehouse, which gets expensive for repeated interactive lookups.
# MAGIC
# MAGIC **Pattern:** the Delta table stays the source of truth. A **Lakebase synced table** copies it into Postgres, where
# MAGIC B-tree indexes answer point and range lookups in milliseconds on a small, autoscaling, scale-to-zero compute.
# MAGIC Heavy analytics stay on the warehouse or Genie.

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]>=3.2"

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

from rag_common import get_params, pg_connect, Timer, Lakebase

p = get_params(dbutils)
w = WorkspaceClient()
lb = Lakebase(w)
CAT, SCH = p["catalog"], p["schema"]
N = int(p["n_companies"])
SRC = f"{CAT}.{SCH}.companies"

# COMMAND ----------

# MAGIC %md ## Generate synthetic companies in Delta

# COMMAND ----------

STATES = ["TX", "CA", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO",
          "MD", "WI", "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT", "IA", "NV", "AR", "MS", "KS", "NM",
          "NE", "ID", "WV", "HI", "NH", "ME", "MT", "RI", "DE", "SD", "ND", "AK", "VT", "WY"]
VERTICALS = ["Construction", "Utilities", "Field Service", "Delivery & Logistics", "Government", "Pharma & Healthcare",
             "Food & Beverage", "Waste Management", "Telecom", "Rental & Leasing", "Transit", "School Transportation",
             "Landscaping", "HVAC & Plumbing", "Oil & Gas"]

states = F.array(*[F.lit(s) for s in STATES])
verts = F.array(*[F.lit(v) for v in VERTICALS])
# skew states toward populous ones: square a uniform draw
state_idx = (F.pow(F.rand(1), 1.7) * len(STATES)).cast("int")
fleet = F.greatest(F.lit(1), F.round(F.exp(F.randn(2) * 1.3 + 2.4))).cast("int")  # log-normal: median ~11

companies = (
    spark.range(N)
    .withColumnRenamed("id", "company_id")
    .withColumn("company_name", F.concat(F.lit("Company "), F.lpad(F.col("company_id").cast("string"), 7, "0")))
    .withColumn("state", states[state_idx])
    .withColumn("vertical", verts[(F.rand(3) * len(VERTICALS)).cast("int")])
    .withColumn("fleet_size", fleet)
    .withColumn("pct_light_duty", F.round(F.rand(4) * 100).cast("int"))
    .withColumn("ev_count", F.floor(F.col("fleet_size") * F.pow(F.rand(5), 4) * 0.5).cast("int"))
    .withColumn("has_telematics", F.rand(6) < (0.25 + F.least(F.lit(0.6), F.log10(F.col("fleet_size")) / 4)))
    .withColumn("annual_revenue_musd", F.round(F.col("fleet_size") * (0.4 + F.rand(7) * 2.5), 1))
    .withColumn("employees", (F.col("fleet_size") * (2 + F.rand(8) * 8)).cast("int"))
)
companies.write.mode("overwrite").option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true").saveAsTable(SRC)
spark.sql(f"ALTER TABLE {SRC} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
display(spark.sql(f"SELECT count(*) n, percentile(fleet_size, 0.5) median_fleet, max(fleet_size) max_fleet FROM {SRC}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load into Lakebase
# MAGIC **Preferred: a Lakebase synced table** (managed pipeline, snapshot/triggered/continuous). This needs the Lakebase
# MAGIC database registered as a UC catalog, which requires `CREATE CATALOG` on the metastore. If that privilege is missing
# MAGIC (as in this demo workspace), the notebook falls back to a streaming `COPY` into the same Postgres table, so the app
# MAGIC works the same either way.

# COMMAND ----------

LB_CATALOG = "rag_lakebase"
branch = f"projects/{p['lakebase_project']}/branches/{p['lakebase_branch']}"
target = f"{LB_CATALOG}.market.companies"


def sync_with_synced_table():
    if not lb.exists(f"catalogs/{LB_CATALOG}"):
        print("Registering Lakebase database as UC catalog", LB_CATALOG)
        lb.create("catalogs", "catalog_id", LB_CATALOG,
                  {"spec": {"postgres_database": p["pg_database"], "branch": branch}})
    if lb.exists(f"synced_tables/{target}"):
        print("synced table exists:", target)
        return
    print("Creating synced table", target)
    lb.create("synced_tables", "synced_table_id", target, {
        "spec": {
            "source_table_full_name": SRC,
            "primary_key_columns": ["company_id"],
            "scheduling_policy": "SNAPSHOT",
            "branch": branch,
            "postgres_database": p["pg_database"],
            "create_database_objects_if_missing": True,
            "new_pipeline_spec": {"storage_catalog": CAT, "storage_schema": SCH},
        }
    }, timeout_s=3600)
    # wait until rows land in Postgres
    with pg_connect(p, w) as conn:
        for _ in range(120):
            try:
                n = conn.execute("SELECT count(*) FROM market.companies").fetchone()[0]
            except Exception:
                n = 0
            print("rows in Lakebase:", n)
            if n >= N:
                return
            time.sleep(30)


COLS = ["company_id", "company_name", "state", "vertical", "fleet_size", "pct_light_duty", "ev_count",
        "has_telematics", "annual_revenue_musd", "employees"]


def load_with_copy():
    with pg_connect(p, w) as conn, Timer() as t:
        conn.execute("""
          CREATE SCHEMA IF NOT EXISTS market;
          DROP TABLE IF EXISTS market.companies;
          CREATE TABLE market.companies (
            company_id BIGINT PRIMARY KEY, company_name TEXT, state TEXT, vertical TEXT, fleet_size INT,
            pct_light_duty INT, ev_count INT, has_telematics BOOLEAN, annual_revenue_musd DOUBLE PRECISION,
            employees INT)""")
        with conn.cursor() as cur, cur.copy(f"COPY market.companies ({', '.join(COLS)}) FROM STDIN") as cp:
            for i, r in enumerate(spark.table(SRC).select(*COLS).toLocalIterator(prefetchPartitions=True)):
                cp.write_row(tuple(r))
                if i % 500_000 == 0:
                    print(f"{i:,} rows")
    print(f"COPY loaded {N:,} rows in {t.ms/1000:.0f}s")


try:
    sync_with_synced_table()
    LOAD_METHOD = "synced table"
except Exception as e:
    if "CREATE CATALOG" not in str(e) and "PERMISSION" not in str(e).upper():
        raise
    print(f"Synced table unavailable ({str(e)[:160]}...). Falling back to COPY.")
    load_with_copy()
    LOAD_METHOD = "COPY fallback"
print("load method:", LOAD_METHOD)

# COMMAND ----------

# MAGIC %md ## Index for lookups (indexes are allowed on synced tables too)

# COMMAND ----------

with pg_connect(p, w) as conn, Timer() as t:
    # covering indexes -> index-only scans for count/sum lookups (no heap reads)
    conn.execute("CREATE INDEX IF NOT EXISTS companies_state_fleet_cov ON market.companies "
                 "(state, fleet_size) INCLUDE (vertical, ev_count, has_telematics)")
    conn.execute("CREATE INDEX IF NOT EXISTS companies_vert_state_fleet_cov ON market.companies "
                 "(vertical, state, fleet_size) INCLUDE (ev_count, has_telematics)")
    conn.execute("CREATE INDEX IF NOT EXISTS companies_vert_fleet_cov ON market.companies "
                 "(vertical, fleet_size) INCLUDE (state, ev_count, has_telematics)")  # any-state lookups
    conn.execute("CREATE INDEX IF NOT EXISTS companies_fleet_cov ON market.companies "
                 "(fleet_size) INCLUDE (state, vertical, ev_count, has_telematics)")  # fleet-size-only lookups
    conn.execute("VACUUM ANALYZE market.companies")  # sets the visibility map so index-only scans skip the heap
print(f"indexes built in {t.ms/1000:.1f}s")

# COMMAND ----------

# MAGIC %md ## Benchmark: Lakebase vs. the same query on Delta

# COMMAND ----------

LOOKUPS = [("TX", None, 50, 200), ("CA", "Construction", 10, 50), ("FL", "Delivery & Logistics", 100, 1000),
           ("OH", None, 1, 5), ("NY", "Utilities", 20, 500)]

def pct(xs, k):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * k))], 1)

with pg_connect(p, w) as conn:
    lb_ms = []
    for _ in range(20):
        for st, v, lo, hi in LOOKUPS:
            with Timer() as t:
                conn.execute(
                    "SELECT count(*), sum(fleet_size), sum(ev_count) FROM market.companies "
                    f"WHERE state = %s {'AND vertical = %s' if v else ''} AND fleet_size BETWEEN %s AND %s",
                    (st, v, lo, hi) if v else (st, lo, hi)).fetchone()
            lb_ms.append(t.ms)

spark_ms = []
for st, v, lo, hi in LOOKUPS:
    with Timer() as t:
        spark.sql(f"""SELECT count(*), sum(fleet_size), sum(ev_count) FROM {SRC}
                      WHERE state = '{st}' {f"AND vertical = '{v}'" if v else ''} AND fleet_size BETWEEN {lo} AND {hi}""").collect()
    spark_ms.append(t.ms)

print(f"Lakebase  p50 {pct(lb_ms, .5)} ms   p95 {pct(lb_ms, .95)} ms   (n={len(lb_ms)})")
print(f"Delta/Spark (serverless notebook) p50 {pct(spark_ms, .5)} ms")
