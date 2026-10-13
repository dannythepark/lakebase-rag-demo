"""Call the Partner Assistant from OUTSIDE Databricks (a laptop, a website backend, Azure API Management).

Auth options (handled by the Databricks SDK's unified auth):
  * Service principal, OAuth M2M (production):
        export DATABRICKS_HOST=https://<workspace>
        export DATABRICKS_CLIENT_ID=<sp-application-id>
        export DATABRICKS_CLIENT_SECRET=<sp-oauth-secret>
    The SP needs CAN_USE on the app.
  * A CLI profile (demo):  python external_client.py --profile <profile>

Usage:
  python external_client.py --app-url https://content-assistant-<id>.aws.databricksapps.com \
      "We sell AI dash cams to mid-size fleets. What sponsorship options reach safety managers?"
"""
import argparse
import json
import time

import requests
from databricks.sdk.core import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--app-url", required=True)
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    cfg = Config(profile=args.profile) if args.profile else Config()
    headers = cfg.authenticate()  # {"Authorization": "Bearer <oauth token>"}
    headers["Content-Type"] = "application/json"
    base = args.app_url.rstrip("/")

    sid = requests.post(f"{base}/api/partner/sessions", headers=headers, json={"title": "external client"}).json()["session_id"]
    t0 = time.perf_counter()
    r = requests.post(f"{base}/api/partner/chat", headers=headers, json={"session_id": sid, "message": args.question})
    r.raise_for_status()
    body = r.json()
    print(body["answer"])
    print("\nsources:")
    for c in body["citations"]:
        print(f"  [{c['n']}] {c['kind']}: {c['title']}")
    print("\ncache:", json.dumps(body["cache"]), "| server timings ms:", json.dumps(body["timings_ms"]),
          f"| round trip {1000 * (time.perf_counter() - t0):.0f} ms")


if __name__ == "__main__":
    main()
