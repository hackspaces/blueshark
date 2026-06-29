"""Mine the tender DBs for India-context training signal.

Two outputs from the real Indian procurement scrape (~3.2M tender records):
  1. india_text.txt  — cleaned, deduped Indian-English domain prose (work
     descriptions, titles, org names) for the pretrain mix's India slice.
  2. india_seeds.json — real values (INR amounts, 6-digit PIN codes, financial
     years, org names) to GROUND the datagen verified-task engine in authentic
     records instead of purely synthetic ones.

  python pipeline/extract_india_corpus.py --db ~/Downloads/tenders_vps.db --max 50000
"""
import argparse
import html
import json
import os
import re
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "india")
TEXT_FIELDS = ["Tender Title", "Work Description", "Organisation Name",
               "Organisation Type", "Product Category", "Product Sub-Category", "Location"]

INR_RE = re.compile(r"₹\s*([\d,]+)")
PIN_RE = re.compile(r"\b([1-9]\d{5})\b")
FY_RE = re.compile(r"\b(20\d{2})[-/](\d{2})\b")


def clean(s):
    if not isinstance(s, str):
        return ""
    s = html.unescape(s).replace("&ampamp#x0d", " ").replace("#x0d", " ")
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--max", type=int, default=50000, help="rows to scan (0 = all)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    con = sqlite3.connect(args.db)
    cur = con.execute("SELECT details_json FROM tender_details"
                      + (f" LIMIT {args.max}" if args.max else ""))

    seen_desc = set()
    inr, pins, fys, orgs = set(), set(), set(), set()
    n_rows = n_text = chars = 0
    tf = open(os.path.join(OUT, "india_text.txt"), "w", encoding="utf-8")
    for (raw,) in cur:
        n_rows += 1
        try:
            d = json.loads(raw)
        except Exception:
            continue
        # text slice (dedup on work description, which is the substantive prose)
        desc = clean(d.get("Work Description", ""))
        if desc and len(desc) > 40 and desc not in seen_desc:
            seen_desc.add(desc)
            parts = [clean(d.get(f, "")) for f in TEXT_FIELDS]
            block = " | ".join(p for p in parts if p)
            tf.write(block + "\n")
            n_text += 1
            chars += len(block)
        # real seeds for datagen
        for v in (d.get("EMD", ""), d.get("Tender Fee", "")):
            m = INR_RE.search(str(v))
            if m:
                inr.add(int(m.group(1).replace(",", "")))
        for m in PIN_RE.finditer(clean(d.get("Address", ""))):
            pins.add(m.group(1))
        for fld in ("Tender Title", "Work Description", "ePublished Date"):
            m = FY_RE.search(clean(d.get(fld, "")))
            if m:
                fys.add(f"{m.group(1)}-{m.group(2)}")
        o = clean(d.get("Organisation Name", ""))
        if o:
            orgs.add(o)
    tf.close()
    con.close()

    seeds = {"inr_amounts": sorted(x for x in inr if x > 0)[:5000],
             "pin_codes": sorted(pins)[:5000],
             "financial_years": sorted(fys),
             "org_names": sorted(orgs)[:5000]}
    json.dump(seeds, open(os.path.join(OUT, "india_seeds.json"), "w"), ensure_ascii=False, indent=1)

    print(f"scanned {n_rows:,} rows")
    print(f"india_text.txt: {n_text:,} unique blocks, {chars/1e6:.1f}MB (~{chars//4/1e6:.1f}M tokens)")
    print(f"seeds: {len(seeds['inr_amounts'])} INR amounts, {len(seeds['pin_codes'])} PINs, "
          f"{len(seeds['financial_years'])} FYs, {len(seeds['org_names'])} orgs")
    print(f"  -> {OUT}/")


if __name__ == "__main__":
    main()
