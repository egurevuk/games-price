"""
BIK Sanctions Screener
======================
Streamlit app that takes a Russian BIK (Bank Identification Code / БИК),
resolves the bank's official name via OpenSanctions' `ru_cbr_banks` dataset
(sourced from the Central Bank of Russia), and screens it against global
sanctions lists via the OpenSanctions /match API.

Author: built for Stape Online Ltd
"""

from __future__ import annotations

import os
import re
import json
import time
from datetime import datetime
from html import unescape
from typing import Any

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OS_BASE = "https://api.opensanctions.org"
BANKS_DATASET = "ru_cbr_banks"          # CBR-sourced Russian banking registry
SANCTIONS_SCOPE = "sanctions"           # /match endpoint scope (sanctions only)
DEFAULT_SCOPE = "default"               # /match endpoint scope (everything)
ALGORITHM = "logic-v2"                  # current recommended scoring algo
REQUEST_TIMEOUT = 30
SANCTION_SCORE_THRESHOLD = 0.70         # below this we don't alert
STRONG_SCORE_THRESHOLD = 0.85           # above this we treat as confirmed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_bik(raw: str) -> str:
    """Strip non-digits and left-pad to 9 digits.

    Russian BIKs are formally 9 digits, but operators frequently drop the
    leading zero (e.g. `44525700` -> `044525700`).
    """
    digits = re.sub(r"\D", "", str(raw or ""))
    if not digits:
        return ""
    if len(digits) == 8:
        digits = "0" + digits
    return digits


def is_valid_bik(bik: str) -> bool:
    return bool(re.fullmatch(r"\d{9}", bik))


def os_headers(api_key: str | None) -> dict:
    """Build Authorization header for OpenSanctions if a key is set."""
    if api_key:
        return {"Authorization": f"ApiKey {api_key}"}
    return {}


def get_api_key() -> str:
    """Return the OpenSanctions API key.

    Resolution order:
      1. ``st.secrets["OPENSANCTIONS_API_KEY"]`` — recommended.
         Stored in ``.streamlit/secrets.toml`` locally, or in the Streamlit
         Cloud "Secrets" panel in deployment. Never committed to git.
      2. ``OPENSANCTIONS_API_KEY`` environment variable — useful for
         Docker / CI / quick local runs.

    Returns an empty string if neither is set, so the caller can show a
    friendly message rather than crash.
    """
    # 1. Streamlit secrets. Accessing st.secrets when no secrets.toml exists
    #    raises StreamlitSecretNotFoundError; accessing a missing key raises
    #    KeyError. Both are non-fatal — we fall through to the env var.
    try:
        val = st.secrets.get("OPENSANCTIONS_API_KEY", "")
        if val:
            return str(val).strip()
    except Exception:
        pass
    # 2. Environment variable
    return os.environ.get("OPENSANCTIONS_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# OpenSanctions API calls
# ---------------------------------------------------------------------------

def _entity_contains_bik(entity: dict, bik: str) -> bool:
    """Strict check: does this entity actually carry this exact BIK?

    We accept a match only if the BIK appears as:
      • the source ID ``ru-bik-{BIK}`` in ``referents``, or
      • the literal value in any indexed identifier property
        (``registrationNumber``, ``bikCode``, ``innCode``, ``ogrnCode``,
        ``taxNumber``, ``identifiers``).
    """
    refs = [str(x) for x in (entity.get("referents") or [])]
    if f"ru-bik-{bik}" in refs:
        return True
    props = entity.get("properties", {}) or {}
    for key in ("registrationNumber", "bikCode", "innCode", "ogrnCode",
                "taxNumber", "identifiers"):
        for v in (props.get(key) or []):
            if str(v) == bik:
                return True
    return False


@st.cache_data(show_spinner=False, ttl=3600)
def lookup_bank_in_cbr_registry(bik: str, api_key: str) -> dict | None:
    """Resolve a Russian BIK to an OpenSanctions entity — strict, no guessing.

    The original implementation used ``q={BIK}`` which is *full-text*
    search — ElasticSearch happily returned the top-scoring hit even when
    the BIK didn't actually appear in it. That produced both false
    positives (random sanctioned bank surfaced for an unrelated BIK) and
    false negatives (branch BIKs whose parent legal entity is indexed
    under a different BIK weren't reached).

    The fix:

    1. Try precise field-scoped lookups in order. The most reliable is the
       source-ID lookup ``referents:"ru-bik-{BIK}"`` on the ``default``
       scope, which catches both standalone entities in the CBR registry
       AND deduplicated sanctioned entities that carry the BIK as a
       source ID. We then fall back to identifier-field lookups, and
       finally a scoped text query inside the CBR registry.
    2. *Strictly* validate every candidate via :func:`_entity_contains_bik`.
       Never return an entity that doesn't actually contain this BIK.
    3. No fuzzy ``results[0]`` fallback — better to report "unknown" than
       to mis-attribute sanctions to the wrong institution.
    """
    attempts: list[tuple[str, str]] = [
        # (scope, lucene query) — most specific first
        ("default",     f'referents:"ru-bik-{bik}"'),
        ("default",     f'identifiers:"{bik}"'),
        (BANKS_DATASET, f'identifiers:"{bik}"'),
        (BANKS_DATASET, bik),  # text-mode fallback inside the CBR dataset only
    ]

    for scope, q in attempts:
        url = f"{OS_BASE}/search/{scope}"
        try:
            resp = requests.get(
                url,
                params={"q": q, "limit": 10},
                headers=os_headers(api_key),
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException:
            continue
        results = resp.json().get("results", []) or []
        for r in results:
            if _entity_contains_bik(r, bik):
                return r
        # else: every result for this query failed strict validation → try next

    return None


# ---------------------------------------------------------------------------
# External CBR directory fallback (bankirsha.com)
# ---------------------------------------------------------------------------
#
# OpenSanctions indexes a `ru-bik-{BIK}` identifier on the parent legal entity
# for each *head-office* BIK that appears on a sanctions list (Sberbank,
# VTB, Alfa-Bank, Tinkoff/T-Bank, etc.). It does NOT index branch BIKs of
# those banks — e.g. 044030653 (Sberbank's Severo-Zapadny territorial
# branch) and 044525411 (VTB's Tsentralny Moscow branch) have no entity in
# OpenSanctions even though their parent legal entity is heavily
# sanctioned.
#
# To screen those correctly we need a complete BIK directory. bankirsha.com
# mirrors the CBR's BIK register and explicitly lists, for every branch
# BIK, the parent's head-office BIK. We scrape that page and, if a head
# office BIK is found, do a second OpenSanctions lookup against it.

DIRECTORY_URL = "https://bankirsha.com/bik.{bik}.html"
DIRECTORY_UA = "Mozilla/5.0 (BIK-Screener; +github.com/anthropics/claude-code)"


@st.cache_data(show_spinner=False, ttl=86400)  # cache a full day
def resolve_bik_via_directory(bik: str) -> dict | None:
    """Fetch BIK metadata from the bankirsha.com CBR directory mirror.

    Returns a dict like::

        {
            "bik": "044030653",
            "name":          "СЕВЕРО-ЗАПАДНЫЙ БАНК ПАО СБЕРБ",
            "fullName":      "СЕВЕРО-ЗАПАДНЫЙ БАНК ПАО СБЕРБАНК",
            "type":          "Территориальные управления Сбербанка (ТУСБ)",
            "swift":         "SABRRU2PXXX",
            "regNumber":     "1481/1309",
            "address":       "191124, САНКТ-ПЕТЕРБУРГ, УЛ КРАСНОГО ТЕКСТИЛЬЩИКА, 2",
            "headOfficeBik": "044525225",      # only present for branches
            "_source":       "bankirsha.com",
        }

    or ``None`` if the page can't be reached or doesn't contain valid data.
    Parsing is regex-based against the page's stable two-column table; if
    bankirsha.com ever restructures the page we just fall back to "unknown".
    """
    url = DIRECTORY_URL.format(bik=bik)
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": DIRECTORY_UA, "Accept-Language": "ru,en"},
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    html = resp.text
    # Safety: the page must contain the BIK we asked for
    if bik not in html:
        return None

    def _grab(pattern: str) -> str | None:
        m = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        return unescape(m.group(1)).strip() if m else None

    info: dict[str, str | None] = {"bik": bik, "_source": "bankirsha.com"}

    # Standard table fields. The page uses <td>Label</td><td>Value</td>.
    # Values sometimes wrap in <a>/<span> — strip inner tags afterwards.
    raw_fields = {
        "name":      r"Наименование банка\s*</td>\s*<td[^>]*>(.+?)</td>",
        "fullName":  r"Полное название\s*</td>\s*<td[^>]*>(.+?)</td>",
        "type":      r"Тип организации\s*</td>\s*<td[^>]*>(.+?)</td>",
        "regNumber": r"Регистрационный номер[^<]*</td>\s*<td[^>]*>(.+?)</td>",
        "swift":     r"SWIFT[^<]*\)\s*</td>\s*<td[^>]*>(.+?)</td>",
        "address":   r"Юридический адрес\s*</td>\s*<td[^>]*>(.+?)</td>",
    }
    for key, pat in raw_fields.items():
        raw = _grab(pat)
        if raw:
            # Strip nested HTML tags from the captured cell
            info[key] = re.sub(r"<[^>]+>", "", raw).strip() or None

    # Head office BIK (only present for branch/division entries).
    # The "Головной офис" section links to the parent's BIK page.
    head_match = re.search(
        r"Головной офис[\s\S]{0,600}?bik\.(\d{9})\.html", html
    )
    if head_match and head_match.group(1) != bik:
        info["headOfficeBik"] = head_match.group(1)

    if not (info.get("fullName") or info.get("name")):
        return None
    return info


def _synthetic_entity_from_directory(bik: str, info: dict) -> dict:
    """Build an OpenSanctions-shaped entity from directory data.

    Used when the BIK exists in the CBR directory but neither it nor its
    head office is indexed by OpenSanctions. The synthetic entity feeds
    /match/sanctions, which can still find the parent legal entity via
    name + SWIFT BIC fuzzy matching.
    """
    name = info.get("fullName") or info.get("name") or f"Bank with BIK {bik}"
    props: dict[str, list[str]] = {
        "name": [name],
        "jurisdiction": ["Russia"],
        "country": ["ru"],
        "registrationNumber": [bik],
    }
    if info.get("swift"):
        # Strip optional 3-char branch suffix (e.g. SABRRU2PXXX → SABRRU2P)
        sw = re.sub(r"X{1,3}$", "", info["swift"])
        props["swiftBic"] = [sw] if sw else [info["swift"]]
    if info.get("address"):
        props["address"] = [info["address"]]
    return {
        "id": None,
        "schema": "Company",
        "properties": props,
        "referents": [f"ru-bik-{bik}"],
        "datasets": [],
        "target": False,
        "_synthetic": True,
        "_source": info.get("_source"),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_full_entity(entity_id: str, api_key: str) -> dict | None:
    """Get a fully-nested entity record by ID.

    `nested=true` causes the API to inline related Sanction objects under
    ``properties.sanctions`` (and other relationships like ownership), each
    one carrying its own authority, program, dates, and reason.
    """
    url = f"{OS_BASE}/entities/{entity_id}"
    resp = requests.get(
        url,
        params={"nested": "true"},
        headers=os_headers(api_key),
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False, ttl=3600)
def match_against_sanctions(
    bank_entity: dict, api_key: str, scope: str = SANCTIONS_SCOPE
) -> dict:
    """Run OpenSanctions /match using the bank entity as query-by-example.

    We pass every known identifier (BIK, INN, OGRN, etc.) plus every name
    variant, so the scoring algorithm has maximum signal. The endpoint
    returns up to 5 candidate matches each with a score.
    """
    url = f"{OS_BASE}/match/{scope}"
    props = (bank_entity or {}).get("properties", {}) or {}

    # Pick out the properties that make sense to forward
    query_props: dict[str, list[str]] = {}
    for key in (
        "name", "alias", "previousName", "weakAlias",
        "innCode", "ogrnCode", "registrationNumber",
        "taxNumber", "swiftBic", "bikCode",
        "address", "mainCountry", "jurisdiction",
        "website", "email", "phone",
    ):
        vals = props.get(key)
        if vals:
            query_props[key] = [str(v) for v in vals]

    # Always force jurisdiction=Russia if not present (helps disambiguate)
    query_props.setdefault("jurisdiction", ["Russia"])

    payload = {"queries": {"q1": {"schema": "Company", "properties": query_props}}}
    params = {"algorithm": ALGORITHM}

    resp = requests.post(
        url,
        params=params,
        headers={**os_headers(api_key), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def is_sanctioned_topic(topics: list[str] | None) -> bool:
    """Topics list contains 'sanction' if the entity is on a sanctions list.

    OpenSanctions topics include things like: sanction, sanction.linked,
    sanction.counter, role.pep, crime, debarment, export.control. We treat
    a direct `sanction` tag as the strong signal. `sanction.linked` (an
    entity linked to but not itself sanctioned) is flagged for review.
    """
    if not topics:
        return False
    t = [str(x).lower() for x in topics]
    return "sanction" in t


# Datasets that are clear sanctions sources — used as a fallback signal
# when topics aren't populated on a /search response.
SANCTIONS_DATASETS_HINTS = {
    "sanctions", "default",
    "us_ofac_sdn", "us_ofac_cons",
    "eu_fsf", "eu_sanctions_map", "eu_travel_bans",
    "gb_hmt_sanctions", "gb_fcdo_sanctions",
    "ca_dfatd_sema_sanctions",
    "ch_seco_sanctions",
    "au_dfat_sanctions",
    "ja_mof_sanctions",
    "nz_russia_sanctions",
    "ua_nsdc_sanctions", "ua_war_sanctions",
    "fr_ga_sanctions", "be_fod_sanctions",
    "sg_terrorists",
    "mc_freezes",
}


def classify(bank_entity: dict, match_response: dict) -> dict:
    """Combine entity-level evidence and /match evidence into a verdict.

    Returns dict with:
        status:    "sanctioned" | "clear" | "review"
        reasons:   list[str] human-readable evidence
        top_hits:  list[dict] sanctions candidates with scores
    """
    reasons: list[str] = []
    top_hits: list[dict] = []
    status = "clear"

    bank_entity = bank_entity or {}
    props = bank_entity.get("properties", {}) or {}

    # 1) The bank registry entity is itself marked as a sanctions target.
    #    OpenSanctions sets `target=True` on entities that are designated
    #    on any watchlist. Because the CBR registry is deduplicated against
    #    sanctions lists, this is the most reliable signal.
    if bank_entity.get("target") is True:
        status = "sanctioned"
        reasons.append("Bank entity is flagged as a sanctions `target` in OpenSanctions.")

    # 2) Topic-based check (belt-and-braces; covers entities where the
    #    search response includes topics but not target=True).
    topics = props.get("topics") or bank_entity.get("topics") or []
    if is_sanctioned_topic(topics) and status != "sanctioned":
        status = "sanctioned"
        reasons.append(f"Bank entity carries sanctions topic ({', '.join(topics)}).")

    # 3) Dataset-membership check — if the entity belongs to any dataset
    #    other than `ru_cbr_banks` AND that dataset looks sanctions-related,
    #    flag as sanctioned. (Pure debarment/PEP datasets won't trigger.)
    datasets = bank_entity.get("datasets") or []
    other_ds = [d for d in datasets if d != BANKS_DATASET]
    sanction_ds_hits = [d for d in other_ds if d in SANCTIONS_DATASETS_HINTS
                        or "sanction" in d.lower() or "ofac" in d.lower()]
    if sanction_ds_hits and status != "sanctioned":
        status = "sanctioned"
        reasons.append(f"Bank appears in sanctions dataset(s): {', '.join(sanction_ds_hits)}.")

    # 4) Independent verification via /match
    q1 = (match_response or {}).get("responses", {}).get("q1", {})
    for hit in q1.get("results", []) or []:
        score = float(hit.get("score") or 0)
        hit_topics = (hit.get("properties", {}) or {}).get("topics") or \
                     hit.get("topics") or []
        top_hits.append({
            "id": hit.get("id"),
            "caption": hit.get("caption"),
            "score": score,
            "match": bool(hit.get("match")),
            "target": bool(hit.get("target")),
            "topics": hit_topics,
            "datasets": hit.get("datasets", []),
            "schema": hit.get("schema"),
        })
        if hit.get("match") and (is_sanctioned_topic(hit_topics) or hit.get("target")):
            if status != "sanctioned":
                status = "sanctioned"
            reasons.append(
                f"/match returned a confirmed match: "
                f"\"{hit.get('caption')}\" score={score:.2f}"
                + (f" topics=[{', '.join(hit_topics)}]" if hit_topics else "")
            )

    # 5) High-score but unconfirmed → REVIEW
    if status == "clear" and top_hits:
        best = max(top_hits, key=lambda x: x["score"])
        if best["score"] >= SANCTION_SCORE_THRESHOLD:
            status = "review"
            reasons.append(
                f"Top candidate \"{best['caption']}\" scored {best['score']:.2f} "
                f"(≥ {SANCTION_SCORE_THRESHOLD:.2f} threshold). Manual review recommended."
            )

    if status == "clear" and not reasons:
        reasons.append("No matches above the alert threshold across sanctions lists.")

    return {"status": status, "reasons": reasons, "top_hits": top_hits}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_status_badge(status: str) -> None:
    colors = {
        "sanctioned": ("#b00020", "❌  SANCTIONED"),
        "review":     ("#b07a00", "⚠️  REVIEW REQUIRED"),
        "clear":      ("#0a7a2f", "✅  CLEAR"),
        "error":      ("#444444", "⚠  LOOKUP ERROR"),
        "unknown":    ("#444444", "❔ UNKNOWN BIK"),
    }
    color, label = colors.get(status, ("#444444", status.upper()))
    st.markdown(
        f"<div style='display:inline-block;padding:6px 14px;"
        f"background:{color};color:white;border-radius:6px;"
        f"font-weight:600;font-size:0.95rem'>{label}</div>",
        unsafe_allow_html=True,
    )


def first(props: dict, key: str, default: str = "—") -> str:
    vals = props.get(key) or []
    return str(vals[0]) if vals else default


def render_bank_card(bank: dict) -> None:
    props = bank.get("properties", {}) or {}
    name = first(props, "name", "(no name)")
    inn = first(props, "innCode")
    ogrn = first(props, "ogrnCode")
    address = first(props, "address")
    status = first(props, "status")
    incorp = first(props, "incorporationDate")
    aliases = props.get("alias") or []
    swift = first(props, "swiftBic")

    st.markdown(f"### {name}")
    cols = st.columns([1, 1, 1])
    cols[0].metric("INN", inn)
    cols[1].metric("OGRN", ogrn)
    cols[2].metric("Status", status)

    with st.expander("More bank details", expanded=False):
        if address != "—":
            st.write(f"**Address:** {address}")
        if incorp != "—":
            st.write(f"**Incorporated:** {incorp}")
        if swift != "—":
            st.write(f"**SWIFT/BIC:** `{swift}`")
        if aliases:
            st.write("**Aliases:** " + " · ".join(aliases[:10]))
        eid = bank.get("id")
        if eid:
            st.write(
                f"**OpenSanctions entity:** "
                f"[`{eid}`](https://www.opensanctions.org/entities/{eid}/)"
            )
        if bank.get("_synthetic"):
            st.info(
                "ℹ️ This bank isn't directly indexed by OpenSanctions — "
                "its identity was derived from "
                f"`{bank.get('_source', 'external directory')}` and the "
                "sanctions check below was performed by fuzzy match on "
                "name + SWIFT against OpenSanctions' /match endpoint."
            )


# Map from OpenSanctions referent prefixes to the issuing authority. Each
# entity's `referents` list contains source IDs like `ofac-12345` or
# `gb-hmt-15016` — we use these to enumerate which sanctions lists the bank
# appears on. Order matters: longest prefix wins (so `ofac-pr-` is matched
# before `ofac-`).
SOURCE_PREFIX_MAP: list[tuple[str, str]] = [
    ("ofac-pr-",      "🇺🇸 US OFAC press release"),
    ("ofac-",         "🇺🇸 US OFAC SDN list"),
    ("usgsa-",        "🇺🇸 US GSA debarment"),
    ("gb-hmt-",       "🇬🇧 UK HM Treasury (OFSI)"),
    ("gb-fcdo-",      "🇬🇧 UK FCDO sanctions"),
    ("gb-invban-",    "🇬🇧 UK Investment Ban"),
    ("gb-coh-psc-",   "🇬🇧 UK Companies House (PSC)"),
    ("gb-coh-",       "🇬🇧 UK Companies House"),
    ("eu-fsf-",       "🇪🇺 EU Financial Sanctions Files"),
    ("eu-sancmap-",   "🇪🇺 EU Sanctions Map"),
    ("eu-oj-",        "🇪🇺 EU Official Journal"),
    ("ca-sema-",      "🇨🇦 Canada SEMA"),
    ("ch-seco-",      "🇨🇭 Switzerland SECO"),
    ("au-dfat-",      "🇦🇺 Australia DFAT"),
    ("ja-mof-",       "🇯🇵 Japan MoF"),
    ("nz-",           "🇳🇿 New Zealand sanctions"),
    ("ua-nsdc-",      "🇺🇦 Ukraine NSDC"),
    ("ua-ws-",        "🇺🇦 Ukraine 'War & Sanctions'"),
    ("fr-ga-",        "🇫🇷 France national freeze"),
    ("be-fod-",       "🇧🇪 Belgium FOD"),
    ("mc-freezes-",   "🇲🇨 Monaco freeze list"),
    ("sg-",           "🇸🇬 Singapore sanctions"),
    ("tw-shtc-",      "🇹🇼 Taiwan SHTC"),
    ("ru-bik-",       "🇷🇺 CBR banking registry"),
    ("ru-inn-",       "🇷🇺 Russia Federal Tax Service"),
    ("lei-",          "GLEIF (LEI)"),
    ("bic-",          "SWIFT BIC reference"),
    ("icijol-",       "ICIJ Offshore Leaks"),
    ("gem-",          "Global Energy Monitor"),
    ("permid-",       "LSEG PermID"),
]


def label_referent(ref: str) -> str | None:
    """Return a human-readable authority label for a source ID prefix."""
    for prefix, label in SOURCE_PREFIX_MAP:
        if ref.startswith(prefix):
            return label
    return None


def render_sanctions_panel(bank_entity: dict) -> None:
    """Render the SANCTIONED detail panel: programs, dates, sources, link."""
    bank_entity = bank_entity or {}
    props = bank_entity.get("properties", {}) or {}
    eid = bank_entity.get("id")
    name = first(props, "name", "(unnamed)")

    # ----- Top-of-panel: prominent OpenSanctions link --------------------
    if eid:
        os_url = f"https://www.opensanctions.org/entities/{eid}/"
        st.markdown(
            f"#### 🚨 {name} is on one or more sanctions lists\n\n"
            f"**Full record on OpenSanctions:** "
            f"[opensanctions.org/entities/{eid}/]({os_url})  \n"
            f"_The page above is the canonical source — it lists every "
            f"designation, the original wording, related entities, and links "
            f"to each authority's primary record._"
        )
        st.link_button("🔗 Open OpenSanctions record ↗", os_url, type="primary")

    # ----- Sanction designations (from nested Sanction sub-entities) -----
    # When fetched with ?nested=true, properties.sanctions contains one
    # Sanction object per designation, each with authority/program/dates.
    sanction_objs = props.get("sanctions") or []
    rows: list[dict] = []
    for s in sanction_objs:
        if not isinstance(s, dict):
            continue
        sp = s.get("properties", {}) or {}
        rows.append({
            "Authority":   first(sp, "authority"),
            "Country":     first(sp, "country"),
            "Program":     first(sp, "program"),
            "Listed":      first(sp, "listingDate", first(sp, "startDate")),
            "Start":       first(sp, "startDate"),
            "End":         first(sp, "endDate", "—"),
            "Reason":      (first(sp, "reason", "") or first(sp, "summary", ""))[:300],
            "Source URL":  first(sp, "sourceUrl", ""),
        })

    if rows:
        st.markdown("##### Sanctions designations")
        # Date-aware sort — newest first
        rows.sort(key=lambda r: str(r.get("Listed") or r.get("Start") or ""), reverse=True)
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Source URL": st.column_config.LinkColumn(
                    "Source", display_text="open ↗"
                ),
                "Reason": st.column_config.TextColumn("Reason / note", width="large"),
            },
        )
        st.caption(f"{len(rows)} designation(s) recorded by OpenSanctions.")
    else:
        st.info(
            "Designation details aren't inlined in the response — open the "
            "OpenSanctions page above for the per-program breakdown."
        )

    # ----- Source authorities derived from referents --------------------
    referents = bank_entity.get("referents") or []
    authorities = {}
    for ref in referents:
        lbl = label_referent(str(ref))
        if not lbl:
            continue
        # Skip pure-reference sources that aren't sanctions/debarment lists
        if lbl in (
            "🇷🇺 CBR banking registry",
            "🇷🇺 Russia Federal Tax Service",
            "GLEIF (LEI)",
            "SWIFT BIC reference",
            "🇬🇧 UK Companies House",
            "🇬🇧 UK Companies House (PSC)",
            "ICIJ Offshore Leaks",
            "Global Energy Monitor",
            "LSEG PermID",
        ):
            continue
        authorities[lbl] = authorities.get(lbl, 0) + 1

    if authorities:
        st.markdown("##### Listed on")
        chip_html = " ".join(
            f"<span style='display:inline-block;margin:3px 6px 3px 0;"
            f"padding:4px 10px;background:#fdecea;color:#b00020;"
            f"border-radius:12px;font-size:0.85rem;font-weight:500'>"
            f"{lbl}{' ×' + str(n) if n > 1 else ''}</span>"
            for lbl, n in sorted(authorities.items())
        )
        st.markdown(chip_html, unsafe_allow_html=True)

    # ----- Topic chips --------------------------------------------------
    topics = props.get("topics") or bank_entity.get("topics") or []
    if topics:
        TOPIC_LABELS = {
            "sanction":          "🚫 Sanctioned",
            "sanction.linked":   "🔗 Sanction-linked",
            "sanction.counter":  "↩ Counter-sanction",
            "debarment":         "⛔ Debarred",
            "export.control":    "📦 Export-controlled",
            "role.pep":          "👤 PEP",
            "role.rca":          "👥 PEP relative/associate",
            "corp.public":       "🏢 Public company",
            "fin.bank":          "🏦 Bank",
        }
        chip_html = " ".join(
            f"<span style='display:inline-block;margin:3px 6px 3px 0;"
            f"padding:4px 10px;background:#f0f2f6;color:#222;"
            f"border-radius:12px;font-size:0.85rem'>"
            f"{TOPIC_LABELS.get(t, t)}</span>"
            for t in topics
        )
        st.markdown("##### Topics")
        st.markdown(chip_html, unsafe_allow_html=True)

    # ----- Narrative summary / notes ------------------------------------
    summary = first(props, "summary", "")
    notes = props.get("notes") or props.get("description") or []
    notes_text = " ".join(str(n) for n in notes) if isinstance(notes, list) else str(notes)
    if summary and summary != "—":
        st.markdown("##### Summary")
        st.write(summary)
    if notes_text and notes_text != "—":
        with st.expander("Official rationale / notes from designating authorities"):
            st.write(notes_text[:3000] + ("…" if len(notes_text) > 3000 else ""))


def render_match_table(top_hits: list[dict]) -> None:
    if not top_hits:
        st.info("No candidate matches returned by /match.")
        return
    rows = []
    for h in top_hits:
        rows.append({
            "Score": round(h["score"], 3),
            "Confirmed match": "✅" if h["match"] else "—",
            "Entity": h["caption"],
            "Topics": ", ".join(h["topics"] or []),
            "Datasets": ", ".join(h["datasets"] or []),
            "OpenSanctions link": f"https://www.opensanctions.org/entities/{h['id']}/",
        })
    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "OpenSanctions link": st.column_config.LinkColumn(
                "OpenSanctions link", display_text="open ↗"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Per-BIK pipeline
# ---------------------------------------------------------------------------

def screen_bik(bik_raw: str, api_key: str, scope: str) -> dict:
    """Run the full pipeline for a single BIK and return a result dict.

    Resolution strategy:
      1. Strict OpenSanctions index lookup (covers head-office BIKs of
         sanctioned banks, e.g. 044525974 → Tinkoff).
      2. If miss → fetch BIK metadata from the external CBR directory
         (bankirsha.com). If it's a branch BIK with a head-office BIK
         pointer (e.g. 044030653 → 044525225), re-query OpenSanctions
         against the head office. That's the common case for branches of
         sanctioned banks (Sberbank, VTB, etc.).
      3. If even the head office isn't indexed → build a synthetic entity
         from the directory data and let /match find the parent legal
         entity by fuzzy name + SWIFT BIC.
      4. If the BIK isn't in the CBR directory at all → unknown.
    """
    bik = normalize_bik(bik_raw)
    result: dict[str, Any] = {
        "bik_input": bik_raw,
        "bik": bik,
        "status": "error",
        "bank": None,
        "match": None,
        "verdict": None,
        "error": None,
        "resolved_via": None,
        "directory_info": None,
    }
    if not is_valid_bik(bik):
        result["status"] = "error"
        result["error"] = "BIK must be 8 or 9 digits."
        return result

    # ---- Stage 1: strict OpenSanctions index lookup -----------------------
    try:
        bank = lookup_bank_in_cbr_registry(bik, api_key)
    except requests.HTTPError as exc:
        result["error"] = f"OpenSanctions lookup failed: {exc.response.status_code} {exc.response.text[:200]}"
        return result
    except Exception as exc:
        result["error"] = f"OpenSanctions lookup failed: {exc}"
        return result

    if bank:
        result["resolved_via"] = "OpenSanctions index (direct BIK match)"

    # ---- Stage 2: external CBR directory fallback for branch BIKs --------
    if not bank:
        info = resolve_bik_via_directory(bik)
        if info:
            result["directory_info"] = info
            # 2a. Branch with a head-office BIK pointer → re-query OpenSanctions
            head_bik = info.get("headOfficeBik")
            if head_bik:
                try:
                    head_bank = lookup_bank_in_cbr_registry(head_bik, api_key)
                except Exception:
                    head_bank = None
                if head_bank:
                    bank = head_bank
                    result["resolved_via"] = (
                        f"CBR directory → head-office BIK {head_bik} "
                        f"(branch \"{info.get('fullName') or info.get('name')}\" "
                        f"of this legal entity)"
                    )
            # 2b. Otherwise build a synthetic entity from the directory data
            if not bank:
                bank = _synthetic_entity_from_directory(bik, info)
                result["resolved_via"] = (
                    "CBR directory (not indexed by OpenSanctions; "
                    "/match performed by name + SWIFT)"
                )

    if not bank:
        result["status"] = "unknown"
        result["error"] = (
            "Couldn't resolve this BIK. Possible reasons:\n\n"
            "• The BIK is invalid or a typo (must be 9 digits starting with 04).\n"
            "• The bank's licence was recently revoked and the entry was "
            "dropped from both OpenSanctions and the external CBR directory.\n"
            "• The directory mirror (bankirsha.com) is temporarily "
            "unreachable from your network.\n\n"
            "Cross-check the BIK on opensanctions.org and on the CBR's own "
            "registry (cbr.ru) before acting on this result."
        )
        return result

    result["bank"] = bank

    # Pull the fully-hydrated entity (includes datasets + sanction relationships)
    full = None
    if bank.get("id"):
        try:
            full = fetch_full_entity(bank["id"], api_key)
        except Exception:
            pass
    result["bank_full"] = full or bank

    try:
        match_response = match_against_sanctions(bank, api_key, scope=scope)
    except requests.HTTPError as exc:
        result["error"] = f"/match failed: {exc.response.status_code} {exc.response.text[:200]}"
        return result
    except Exception as exc:
        result["error"] = f"/match failed: {exc}"
        return result
    result["match"] = match_response

    verdict = classify(result["bank_full"], match_response)
    result["verdict"] = verdict
    result["status"] = verdict["status"]
    return result


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BIK Sanctions Screener",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 BIK Sanctions Screener")
st.caption(
    "Resolve a Russian Bank Identification Code (БИК) to its issuer and screen "
    "it against global sanctions lists via the OpenSanctions API."
)

# --- Sidebar config -------------------------------------------------------
api_key = get_api_key()

with st.sidebar:
    st.header("Configuration")

    # API key status (no input — the key lives in st.secrets / env)
    if api_key:
        st.success("OpenSanctions API key loaded ✓", icon="🔑")
        with st.expander("Where is the key loaded from?"):
            try:
                in_secrets = bool(st.secrets.get("OPENSANCTIONS_API_KEY"))
            except Exception:
                in_secrets = False
            in_env = bool(os.environ.get("OPENSANCTIONS_API_KEY"))
            st.markdown(
                f"* `st.secrets`: {'✅ set' if in_secrets else '— not set'}\n"
                f"* `OPENSANCTIONS_API_KEY` env var: {'✅ set' if in_env else '— not set'}"
            )
    else:
        st.error("OpenSanctions API key not configured", icon="🔒")
        with st.expander("How to add the key", expanded=True):
            st.markdown(
                "Create **`.streamlit/secrets.toml`** in the project root:\n\n"
                "```toml\n"
                'OPENSANCTIONS_API_KEY = "your_key_here"\n'
                "```\n\n"
                "…or export it as an env var before launching Streamlit:\n\n"
                "```bash\n"
                'export OPENSANCTIONS_API_KEY="your_key_here"\n'
                "streamlit run app.py\n"
                "```\n\n"
                "Get a free key at "
                "[opensanctions.org/account](https://www.opensanctions.org/account/)."
            )

    scope = st.radio(
        "Screening scope",
        options=[SANCTIONS_SCOPE, DEFAULT_SCOPE],
        index=0,
        help=(
            "`sanctions` — only checks against sanctions lists (recommended). "
            "`default` — broader scope incl. PEPs, crime, debarment."
        ),
        format_func=lambda s: {
            SANCTIONS_SCOPE: "Sanctions only (recommended)",
            DEFAULT_SCOPE: "All (sanctions + PEPs + crime)",
        }[s],
    )
    st.divider()
    st.markdown(
        "**About**\n\n"
        "* BIK resolved via OpenSanctions `ru_cbr_banks` dataset "
        "(Central Bank of Russia registry, refreshed daily).\n"
        "* Sanctions match via `/match` endpoint with `logic-v2` scoring.\n"
        "* The CBR dataset and many sanctions lists are deduplicated by "
        "OpenSanctions, so a sanctioned bank shows up as a single entity "
        "tagged with the `sanction` topic."
    )

# --- Main pane ------------------------------------------------------------
tab_single, tab_batch = st.tabs(["Single BIK", "Batch screening"])

# ---------- Tab 1: Single BIK --------------------------------------------
with tab_single:
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        bik_input = st.text_input(
            "Enter a BIK (8 or 9 digits)",
            placeholder="e.g. 044525700",
            label_visibility="collapsed",
        )
    with col_btn:
        run = st.button("Screen", type="primary", use_container_width=True)

    # Sample chips for convenience — verified against opensanctions.org
    st.caption("Try a sample BIK:")
    samples = [
        ("044525974", "sanctioned"),  # Tinkoff — direct hit in OpenSanctions
        ("044030653", "sanctioned"),  # Sberbank Severo-Zapadny branch — via head office
        ("044525411", "sanctioned"),  # VTB Tsentralny branch — via head office
        ("044525593", "sanctioned"),  # Alfa-Bank — direct
        ("044525104", "sanctioned"),  # Bank Tochka — direct
        ("044525700", "clear"),
        ("046577904", "clear"),
        ("046015762", "clear"),
        ("040349556", "clear"),
    ]
    # Layout in two rows: sanctioned on top, clear below
    sanctioned = [s for s in samples if s[1] == "sanctioned"]
    clear = [s for s in samples if s[1] == "clear"]
    for row, label in [(sanctioned, "Expected SANCTIONED"), (clear, "Expected CLEAR")]:
        st.caption(label)
        cols = st.columns(len(row))
        for col, (s, lab) in zip(cols, row):
            with col:
                icon = "❌" if lab == "sanctioned" else "✅"
                if st.button(f"{icon} {s}", key=f"sample_{s}",
                             use_container_width=True,
                             help=f"expected: {lab}"):
                    st.session_state["_prefill"] = s
                    st.rerun()

    if "_prefill" in st.session_state and not bik_input:
        bik_input = st.session_state.pop("_prefill")
        run = True

    if run:
        if not api_key:
            st.warning(
                "OpenSanctions API key is not configured. Add it to "
                "`.streamlit/secrets.toml` as "
                "`OPENSANCTIONS_API_KEY = \"...\"` (see sidebar) and reload.",
                icon="🔒",
            )
        elif not bik_input.strip():
            st.warning("Please enter a BIK.")
        else:
            with st.spinner(f"Screening BIK {normalize_bik(bik_input)}…"):
                res = screen_bik(bik_input, api_key, scope)

            st.divider()
            top = st.columns([1, 3])
            with top[0]:
                render_status_badge(res["status"])
                st.write(f"**BIK:** `{res['bik']}`")
                if res.get("resolved_via"):
                    st.caption(f"🔍 Resolved via: {res['resolved_via']}")
            with top[1]:
                if res["status"] == "error":
                    st.error(res["error"])
                elif res["status"] == "unknown":
                    st.warning(res["error"])
                else:
                    for r in res["verdict"]["reasons"]:
                        st.write(f"• {r}")

            if res.get("bank"):
                st.divider()
                render_bank_card(res["bank_full"] or res["bank"])

            # Detailed sanctions panel — only when this bank is actually sanctioned
            if res.get("status") == "sanctioned":
                st.divider()
                render_sanctions_panel(res["bank_full"] or res["bank"])

            if res.get("verdict"):
                st.divider()
                st.subheader("Sanctions match candidates")
                st.caption(
                    "Independent verification: results of querying "
                    "`POST /match/sanctions` with the bank's properties. "
                    "A score ≥ 0.85 with `match=true` is a confirmed hit."
                )
                render_match_table(res["verdict"]["top_hits"])

            with st.expander("Raw API responses (debug)"):
                st.write("**Bank (search):**")
                st.json(res.get("bank") or {})
                st.write("**Bank (full entity):**")
                st.json(res.get("bank_full") or {})
                st.write("**/match response:**")
                st.json(res.get("match") or {})

# ---------- Tab 2: Batch screening ---------------------------------------
with tab_batch:
    st.write(
        "Paste one BIK per line (or comma/space-separated). "
        "Up to ~50 at a time is comfortable."
    )
    bulk = st.text_area(
        "BIKs",
        height=140,
        placeholder="040813713\n044030653\n046577904\n046015762\n044525700",
        label_visibility="collapsed",
    )
    run_batch = st.button("Screen all", type="primary")

    if run_batch:
        if not api_key:
            st.warning(
                "OpenSanctions API key is not configured. Add it to "
                "`.streamlit/secrets.toml` and reload.",
                icon="🔒",
            )
        else:
            raw_biks = re.split(r"[\s,;]+", bulk.strip())
            biks = [b for b in (normalize_bik(x) for x in raw_biks) if b]
            biks = list(dict.fromkeys(biks))  # de-dupe, preserve order
            if not biks:
                st.warning("No valid BIKs found.")
            else:
                progress = st.progress(0.0)
                status_line = st.empty()
                results = []
                for i, b in enumerate(biks, 1):
                    status_line.write(f"Screening {b} ({i}/{len(biks)})…")
                    results.append(screen_bik(b, api_key, scope))
                    progress.progress(i / len(biks))
                    # gentle rate-limit; OpenSanctions free tier ~60 rpm
                    time.sleep(0.2)
                status_line.empty()
                progress.empty()

                # Build a results dataframe
                rows = []
                for r in results:
                    props = (r.get("bank") or {}).get("properties", {}) or {}
                    bank_id = (r.get("bank") or {}).get("id")
                    rows.append({
                        "BIK": r["bik"],
                        "Status": r["status"],
                        "Bank": first(props, "name"),
                        "INN": first(props, "innCode"),
                        "OGRN": first(props, "ogrnCode"),
                        "Top hit": (
                            r["verdict"]["top_hits"][0]["caption"]
                            if r.get("verdict") and r["verdict"]["top_hits"] else "—"
                        ),
                        "Top score": (
                            round(r["verdict"]["top_hits"][0]["score"], 3)
                            if r.get("verdict") and r["verdict"]["top_hits"] else None
                        ),
                        "Notes": (
                            r["verdict"]["reasons"][0]
                            if r.get("verdict") and r["verdict"]["reasons"]
                            else (r.get("error") or "")
                        ),
                        "OpenSanctions": (
                            f"https://www.opensanctions.org/entities/{bank_id}/"
                            if bank_id else ""
                        ),
                    })
                df = pd.DataFrame(rows)

                # Color status — Styler.map (pandas ≥ 2.1; Styler.applymap is deprecated)
                def _color(s):
                    return {
                        "sanctioned": "background-color:#fdecea;color:#b00020;font-weight:600",
                        "review":     "background-color:#fff4e0;color:#b07a00;font-weight:600",
                        "clear":      "background-color:#e9f7ee;color:#0a7a2f;font-weight:600",
                    }.get(s, "")
                styled = df.style.map(_color, subset=["Status"]) \
                    if hasattr(df.style, "map") else df.style.applymap(_color, subset=["Status"])
                st.divider()
                st.dataframe(
                    styled,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "OpenSanctions": st.column_config.LinkColumn(
                            "OpenSanctions", display_text="open ↗"
                        ),
                    },
                )

                # Export
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    csv,
                    file_name=f"bik_screening_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

                # Per-BIK detail expanders
                st.divider()
                st.subheader("Details")
                for r in results:
                    label = f"{r['bik']} — {first((r.get('bank') or {}).get('properties', {}) or {}, 'name')} — {r['status'].upper()}"
                    with st.expander(label):
                        if r["status"] in ("error", "unknown"):
                            st.error(r.get("error") or "")
                            continue
                        for line in r["verdict"]["reasons"]:
                            st.write(f"• {line}")
                        if r["status"] == "sanctioned":
                            st.divider()
                            render_sanctions_panel(r["bank_full"] or r["bank"])
                        st.divider()
                        render_match_table(r["verdict"]["top_hits"])

st.divider()
st.caption(
    f"Data: OpenSanctions ({BANKS_DATASET} + sanctions collections). "
    "Not legal advice — confirm matches with your compliance officer."
)
