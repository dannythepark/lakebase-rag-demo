# Teardown

Everything billable this demo creates, and how to remove it. **The Lakebase delete is permanent.**

```bash
P=<profile>
cd lakebase-rag-demo

# 1. App, jobs (and their runs), uploaded bundle files
databricks bundle destroy --profile $P

# 2. Synced table, Lakebase UC catalog registration (if 06 ran)
databricks postgres delete-synced-table synced_tables/rag_lakebase.market.companies --profile $P
databricks postgres delete-catalog catalogs/rag_lakebase --profile $P

# 3. Lakebase project (all branches, computes and data)
databricks postgres delete-project projects/lakebase-rag-demo --profile $P

# 4. Unity Catalog schema with the Delta tables
databricks schemas delete demo_catalog.rag_demo --force --profile $P
```

If you only want to stop spend between rehearsals, nothing is required. The Lakebase compute scales to zero after 5 idle minutes, and the app can be stopped with `databricks apps stop content-assistant --profile <profile>`.
