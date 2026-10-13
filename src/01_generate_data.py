# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Generate synthetic trade-media content
# MAGIC
# MAGIC Creates two Delta tables in Unity Catalog, which stays the **source of truth**:
# MAGIC
# MAGIC | Table | What | Rows |
# MAGIC |---|---|---|
# MAGIC | `articles` | Articles, news, blogs, webcast and videocast recaps across 8 **fictional** brand sites | ~3,000 |
# MAGIC | `campaigns` | Media-kit offerings and past sponsor campaign case studies | ~150 |
# MAGIC
# MAGIC Metadata (brand, topic, tags, dates) is generated deterministically in Python, and the text is
# MAGIC written by a Databricks-hosted LLM with `ai_query`. Topics deliberately repeat across brands and
# MAGIC years so "has this already been covered?" returns real hits.

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]>=3.2"

# COMMAND ----------

import random
from datetime import date, timedelta

from rag_common import get_params

p = get_params(dbutils)
CAT, SCH = p["catalog"], p["schema"]
N_ARTICLES = int(p["n_articles"])
LLM = p["llm_endpoint"]
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CAT}.{SCH}")
print(p)

# COMMAND ----------

# MAGIC %md ## Brands, topics, tags

# COMMAND ----------

BRANDS = {
    "FleetPulse": ("fleetpulse.example", "fleet management for corporate and government fleets"),
    "Upfit & Haul": ("upfithaul.example", "work trucks, vocational vehicles and upfitting"),
    "Transit Forward": ("transitforward.example", "public transit agencies and bus operations"),
    "Yellow Route": ("yellowroute.example", "school transportation and pupil transport"),
    "Dealer Lot Report": ("dealerlot.example", "auto dealership operations and F&I"),
    "Bay & Bench": ("baybench.example", "automotive aftermarket, collision and repair shops"),
    "Charge Depot": ("chargedepot.example", "fleet electrification and EV infrastructure"),
    "SafeMiles": ("safemiles.example", "fleet safety, risk and regulatory compliance"),
}

# (topic, brands that cover it, tags)
TOPICS = [
    ("EV charging infrastructure for depot-based fleets", ["Charge Depot", "FleetPulse", "Transit Forward"], ["ev", "charging", "depot", "infrastructure"]),
    ("Total cost of ownership: EV vs ICE light-duty fleets", ["Charge Depot", "FleetPulse"], ["ev", "tco", "fleet-cost", "acquisition"]),
    ("Utility partnerships and make-ready programs for fleet charging", ["Charge Depot", "Transit Forward"], ["ev", "charging", "utilities", "grants"]),
    ("Battery-electric school buses and V2G pilots", ["Yellow Route", "Charge Depot"], ["ev", "school-bus", "v2g", "grants"]),
    ("School bus driver shortage and recruitment", ["Yellow Route"], ["driver-shortage", "recruitment", "school-bus", "workforce"]),
    ("Student tracking and routing software", ["Yellow Route"], ["routing", "software", "school-bus", "safety"]),
    ("Stop-arm violations and camera enforcement", ["Yellow Route", "SafeMiles"], ["safety", "school-bus", "cameras", "regulation"]),
    ("Telematics ROI for small and mid-size fleets", ["FleetPulse", "SafeMiles", "Upfit & Haul"], ["telematics", "roi", "small-fleet", "data"]),
    ("AI dash cams and driver coaching", ["SafeMiles", "FleetPulse"], ["dash-cams", "ai", "driver-coaching", "safety"]),
    ("Distracted driving policy and enforcement", ["SafeMiles", "FleetPulse"], ["safety", "distracted-driving", "policy", "risk"]),
    ("Nuclear verdicts and fleet liability insurance", ["SafeMiles", "FleetPulse"], ["insurance", "liability", "risk", "legal"]),
    ("FMCSA and DOT compliance updates", ["SafeMiles", "Upfit & Haul"], ["regulation", "fmcsa", "compliance", "eld"]),
    ("Fuel cards, fuel price volatility and fuel management", ["FleetPulse", "Upfit & Haul"], ["fuel", "fleet-cost", "fuel-cards", "data"]),
    ("Vehicle acquisition delays and OEM order banks", ["FleetPulse", "Dealer Lot Report", "Upfit & Haul"], ["acquisition", "oem", "supply-chain", "remarketing"]),
    ("Remarketing and used vehicle values", ["FleetPulse", "Dealer Lot Report"], ["remarketing", "used-vehicles", "residuals", "acquisition"]),
    ("Upfitting lead times and chassis availability", ["Upfit & Haul"], ["upfitting", "supply-chain", "vocational", "oem"]),
    ("Electric work trucks and vocational EVs", ["Upfit & Haul", "Charge Depot"], ["ev", "vocational", "upfitting", "work-truck"]),
    ("Preventive maintenance programs and uptime", ["FleetPulse", "Bay & Bench", "Upfit & Haul"], ["maintenance", "uptime", "fleet-cost", "data"]),
    ("Technician shortage in collision and repair", ["Bay & Bench"], ["technician-shortage", "workforce", "collision", "training"]),
    ("ADAS calibration in collision repair", ["Bay & Bench", "SafeMiles"], ["adas", "collision", "calibration", "safety"]),
    ("Right to repair legislation", ["Bay & Bench", "Dealer Lot Report"], ["right-to-repair", "regulation", "aftermarket", "data"]),
    ("Parts supply chain and aftermarket pricing", ["Bay & Bench"], ["parts", "supply-chain", "aftermarket", "pricing"]),
    ("Dealership F&I product trends", ["Dealer Lot Report"], ["f-and-i", "dealership", "pricing", "consumer"]),
    ("Digital retailing and online vehicle sales", ["Dealer Lot Report"], ["digital-retail", "dealership", "consumer", "software"]),
    ("Dealer service department profitability", ["Dealer Lot Report", "Bay & Bench"], ["service", "dealership", "technician-shortage", "pricing"]),
    ("Zero-emission bus procurement and FTA grants", ["Transit Forward", "Charge Depot"], ["ev", "transit", "grants", "procurement"]),
    ("Transit ridership recovery and service redesign", ["Transit Forward"], ["ridership", "transit", "planning", "data"]),
    ("Microtransit and on-demand transit", ["Transit Forward"], ["microtransit", "transit", "software", "planning"]),
    ("Transit operator assaults and safety", ["Transit Forward", "SafeMiles"], ["safety", "transit", "workforce", "policy"]),
    ("Fleet sustainability reporting and emissions goals", ["FleetPulse", "Charge Depot"], ["sustainability", "emissions", "reporting", "ev"]),
    ("Data and analytics for fleet decision-making", ["FleetPulse", "SafeMiles"], ["data", "analytics", "ai", "telematics"]),
    ("Generative AI in fleet operations", ["FleetPulse", "Dealer Lot Report", "SafeMiles"], ["ai", "software", "data", "automation"]),
    ("Tire management and retreading", ["FleetPulse", "Bay & Bench", "Upfit & Haul"], ["tires", "maintenance", "fleet-cost", "safety"]),
    ("Fleet leasing vs owning", ["FleetPulse", "Dealer Lot Report"], ["leasing", "acquisition", "fleet-cost", "tco"]),
    ("Vehicle cybersecurity and connected-car data privacy", ["SafeMiles", "Dealer Lot Report"], ["cybersecurity", "data", "privacy", "telematics"]),
    ("Hydrogen and alternative fuels for heavy fleets", ["Upfit & Haul", "Transit Forward"], ["hydrogen", "alt-fuel", "emissions", "vocational"]),
    ("Winter operations and severe weather readiness", ["Upfit & Haul", "Yellow Route", "Transit Forward"], ["weather", "operations", "safety", "maintenance"]),
    ("Fleet manager career paths and leadership", ["FleetPulse"], ["leadership", "workforce", "careers", "training"]),
    ("Autonomous vehicles and driver-assist in commercial fleets", ["SafeMiles", "Transit Forward", "FleetPulse"], ["autonomous", "adas", "safety", "ai"]),
    ("Mobile service vans and field service fleets", ["Upfit & Haul", "FleetPulse"], ["field-service", "upfitting", "vocational", "telematics"]),
]

ANGLES = [
    "a case study of one named (fictional) organization",
    "a how-to guide with practical steps",
    "a news report on a recent announcement",
    "a survey/benchmark data story with numbers",
    "an opinion column from an industry veteran",
    "a Q&A interview with a fleet leader",
    "a buyer's guide comparing approaches",
    "a regulatory/policy explainer",
    "a lessons-learned retrospective",
    "a trend outlook for the coming year",
]

CONTENT_TYPES = [("article", 0.55), ("news", 0.2), ("blog", 0.12), ("webcast", 0.08), ("videocast", 0.05)]

rng = random.Random(42)

def pick_type():
    r, acc = rng.random(), 0.0
    for t, w in CONTENT_TYPES:
        acc += w
        if r <= acc:
            return t
    return "article"

start, end = date(2019, 1, 1), date(2026, 9, 30)
span = (end - start).days

rows = []
for i in range(N_ARTICLES):
    topic, brands, tags = rng.choice(TOPICS)
    # mostly the topic's home brands, sometimes another brand covers it too
    brand = rng.choice(brands) if rng.random() < 0.85 else rng.choice(list(BRANDS))
    # skew toward recent years
    d = start + timedelta(days=int(span * (rng.random() ** 0.6)))
    extra = rng.sample([t for _, _, ts in TOPICS for t in ts if t not in tags], k=rng.randint(0, 2))
    article_id = f"A{i+1:05d}"
    slug_brand = BRANDS[brand][0]
    rows.append(
        dict(
            article_id=article_id,
            brand=brand,
            brand_domain=slug_brand,
            topic=topic,
            angle=rng.choice(ANGLES),
            content_type=pick_type(),
            published_at=d,
            tags=sorted(set(tags[: rng.randint(2, 4)] + extra)),
            author=rng.choice(["J. Alvarez", "M. Chen", "R. Okafor", "S. Patel", "K. Novak", "T. Brooks", "L. Haddad", "D. Kim", "A. Russo", "P. Mehta", "E. Walsh", "C. Dubois"]),
        )
    )

meta = spark.createDataFrame(rows)
meta.createOrReplaceTempView("article_meta")
display(meta.groupBy("brand").count())

# COMMAND ----------

# MAGIC %md ## Write article text with `ai_query`
# MAGIC The model returns `TITLE: …` on the first line, followed by the body. Rows the model fails on are dropped.

# COMMAND ----------

brand_desc_map = ", ".join(f"'{b}', '{d}'" for b, (_, d) in BRANDS.items())

spark.sql(f"""
CREATE OR REPLACE TABLE {CAT}.{SCH}.articles_raw AS
WITH m AS (
  SELECT *, map({brand_desc_map})[brand] AS brand_desc FROM article_meta
)
SELECT m.*,
  ai_query(
    '{LLM}',
    concat(
      'You write for "', brand, '", a B2B trade publication covering ', brand_desc, '. ',
      'Write a ', content_type,
      CASE WHEN content_type IN ('webcast','videocast') THEN ' recap (summarize what the panelists discussed, with short quotes)' ELSE '' END,
      ' published in ', date_format(published_at, 'MMMM yyyy'), '. ',
      'Topic: ', topic, '. Angle: ', angle, '. ',
      'Use realistic but entirely FICTIONAL organizations, people and figures. Be specific and concrete; ',
      'mention the year context naturally. 350-650 words, plain paragraphs, no markdown headings, no bullet lists. ',
      'Output format exactly: first line "TITLE: <headline>", then a blank line, then the body.'
    ),
    modelParameters => named_struct('max_tokens', 1100, 'temperature', 0.9),
    failOnError => false
  ) AS gen
FROM m
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CAT}.{SCH}.articles
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS
SELECT
  article_id,
  brand,
  brand_domain,
  regexp_replace(regexp_extract(gen.result, '^\\\\s*TITLE:[ \\\\t]*([^\\\\n]*)', 1), '[*"]|^\\\\s+|\\\\s+$', '') AS title,
  regexp_replace(regexp_replace(gen.result, '^(\\\\s*(TITLE|BODY):[^\\\\n]*\\\\n)+', ''), '^\\\\s*(BODY:|DESCRIPTION:|CASE STUDY:)?\\\\s*|\\\\s+$', '') AS body,
  topic,
  angle,
  content_type,
  published_at,
  tags,
  author,
  concat('https://', brand_domain, '/', content_type, '/',
         trim(BOTH '-' FROM lower(regexp_replace(regexp_replace(regexp_extract(gen.result, '^\\\\s*TITLE:[ \\\\t]*([^\\\\n]*)', 1), '[*"]|^\\\\s+|\\\\s+$', ''), '[^A-Za-z0-9]+', '-'))),
         '-', lower(article_id)) AS url
FROM {CAT}.{SCH}.articles_raw
WHERE gen.errorMessage IS NULL
  AND gen.result RLIKE '^\\\\s*TITLE:'
  AND length(gen.result) > 800
""")
spark.sql(f"DROP TABLE IF EXISTS {CAT}.{SCH}.articles_raw")
display(spark.sql(f"SELECT count(*) n, round(avg(size(split(body, ' ')))) avg_words FROM {CAT}.{SCH}.articles"))
display(spark.table(f"{CAT}.{SCH}.articles").limit(5))

# COMMAND ----------

# MAGIC %md ## Campaigns and media kit
# MAGIC Two kinds of records: **offerings** (what an advertiser can buy) and **case studies** (past sponsor campaigns and their results).

# COMMAND ----------

PACKAGES = [
    ("Newsletter takeover", "$$", "8,000-15,000"),
    ("Sponsored webinar", "$$$", "18,000-35,000"),
    ("Custom content series", "$$$", "25,000-60,000"),
    ("Research report sponsorship", "$$$", "30,000-75,000"),
    ("Dedicated email blast", "$", "3,500-7,500"),
    ("Run-of-site display", "$", "2,000-10,000 / month"),
    ("Podcast sponsorship", "$$", "6,000-14,000 / quarter"),
    ("Lead generation program", "$$$", "20,000-50,000"),
    ("Event and conference sponsorship", "$$$", "15,000-90,000"),
    ("Awards program sponsorship", "$$", "10,000-25,000"),
]
SPONSORS = [
    "telematics provider", "EV charging provider", "commercial truck OEM", "tire manufacturer",
    "fuel card company", "fleet insurance carrier", "upfitter", "fleet maintenance software vendor",
    "AI dash cam vendor", "fleet leasing company", "school bus manufacturer", "transit software vendor",
    "collision repair equipment maker", "dealer management system vendor",
]
AUDIENCES = {
    "FleetPulse": "corporate and public-sector fleet managers (avg fleet 500+ vehicles)",
    "Upfit & Haul": "vocational fleet operators, upfitters and work truck buyers",
    "Transit Forward": "transit agency executives, planners and maintenance chiefs",
    "Yellow Route": "school district transportation directors and contractors",
    "Dealer Lot Report": "dealer principals, GMs and F&I managers",
    "Bay & Bench": "collision and mechanical repair shop owners and technicians",
    "Charge Depot": "fleet electrification leads, utilities and charging buyers",
    "SafeMiles": "fleet safety, risk and compliance managers",
}

crng = random.Random(7)
camps = []
for i in range(150):
    kind = "offering" if i < 40 else "case_study"
    pkg, tier, price = PACKAGES[i % len(PACKAGES)] if kind == "offering" else crng.choice(PACKAGES)
    brands = crng.sample(list(BRANDS), k=crng.randint(1, 3))
    camps.append(
        dict(
            campaign_id=f"C{i+1:04d}",
            record_type=kind,
            package_type=pkg,
            price_tier=tier,
            price_range_usd=price,
            sponsor_category=None if kind == "offering" else crng.choice(SPONSORS),
            brands=brands,
            target_audience="; ".join(AUDIENCES[b] for b in brands),
            year=None if kind == "offering" else crng.randint(2021, 2026),
        )
    )
spark.createDataFrame(camps).createOrReplaceTempView("campaign_meta")

spark.sql(f"""
CREATE OR REPLACE TABLE {CAT}.{SCH}.campaigns
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS
WITH g AS (
  SELECT *,
    ai_query(
      '{LLM}',
      CASE WHEN record_type = 'offering' THEN concat(
        'Write a media-kit entry for a B2B trade-media publisher advertising product: "', package_type,
        '" offered across these publications: ', array_join(brands, ', '), '. Audience: ', target_audience,
        '. Price range USD ', price_range_usd, '. Describe what is included, typical deliverables, audience reach ',
        '(fictional but plausible numbers), lead times, and who it is best for. 120-200 words, plain prose. ',
        'Output format exactly: first line "NAME: <offering name>", blank line, then the description.')
      ELSE concat(
        'Write a past campaign case study for a B2B trade-media publisher. Sponsor: a FICTIONAL ',
        sponsor_category, ' (invent a company name). Package: ', package_type, ' in ', year,
        ' across: ', array_join(brands, ', '), '. Audience: ', target_audience,
        '. Include the sponsor goal, what ran, and concrete results (impressions, registrations, MQLs, CTR or pipeline - ',
        'fictional but plausible). 120-200 words, plain prose. ',
        'Output format exactly: first line "NAME: <campaign name>", blank line, then the case study.')
      END,
      modelParameters => named_struct('max_tokens', 600, 'temperature', 0.8),
      failOnError => false
    ) AS gen
  FROM campaign_meta
)
SELECT
  campaign_id, record_type,
  regexp_replace(regexp_extract(gen.result, '^\\\\s*NAME:[ \\\\t]*([^\\\\n]*)', 1), '[*"]|^\\\\s+|\\\\s+$', '') AS name,
  package_type, sponsor_category, brands, target_audience, price_tier, price_range_usd, year,
  regexp_replace(regexp_replace(gen.result, '^(\\\\s*(NAME|BODY):[^\\\\n]*\\\\n)+', ''), '^\\\\s*(BODY:|DESCRIPTION:|CASE STUDY:)?\\\\s*|\\\\s+$', '') AS description
FROM g
WHERE gen.errorMessage IS NULL AND gen.result RLIKE '^\\\\s*NAME:'
""")
display(spark.sql(f"SELECT record_type, count(*) FROM {CAT}.{SCH}.campaigns GROUP BY 1"))
display(spark.table(f"{CAT}.{SCH}.campaigns").limit(5))
