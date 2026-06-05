"""OPTIONAL literature-context module — fully isolated from the ML core.

Given an SGCA variant, retrieve what the *published literature* says about that specific
residue/variant via Europe PMC. Hard rules:
  * It only ever returns what the API returns — it NEVER fabricates a finding or citation.
  * It imports nothing from the model/feature code; the science core stands alone.
  * Offline / fixture mode does not invent results: it returns the exact query it *would*
    run and an explicit empty result set.

Europe PMC REST: https://www.ebi.ac.uk/europepmc/webservices/rest/search
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cadinho.config import Config
from cadinho.normalize.hgvs import ParsedVariant, parse_variant

EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


@dataclass
class LiteratureResult:
    online: bool
    query: str
    results: list[dict] = field(default_factory=list)
    note: str = ""
    error: str | None = None


def variant_aliases(pv: ParsedVariant) -> list[str]:
    """Search terms for a variant: HGVS plus common short forms (e.g. R77C, Arg77Cys)."""
    aliases: list[str] = []
    if pv.hgvs_p:
        aliases.append(pv.hgvs_p)                       # p.Arg77Cys
        body = pv.hgvs_p[2:].strip("()")
        if body:
            aliases.append(body)                        # Arg77Cys
    if pv.aaref and pv.protein_position and pv.aaalt:
        aliases.append(f"{pv.aaref}{pv.protein_position}{pv.aaalt}")  # R77C
    if pv.hgvs_c:
        aliases.append(pv.hgvs_c)                       # c.229C>T
    # de-dup, preserve order
    seen: set[str] = set()
    return [a for a in aliases if not (a in seen or seen.add(a))]


def build_query(gene: str, aliases: list[str]) -> str:
    if aliases:
        terms = " OR ".join(f'"{a}"' for a in aliases)
        return f'{gene} AND ({terms})'
    return gene


def search_europepmc(query: str, page_size: int = 10, timeout: int = 30) -> list[dict]:
    """Live Europe PMC search. Returns ONLY real records. Raises on network failure."""
    import requests

    resp = requests.get(EUROPEPMC_SEARCH, params={
        "query": query, "format": "json", "pageSize": page_size,
        "resultType": "lite"}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    out = []
    for r in payload.get("resultList", {}).get("result", []):
        pmid = r.get("pmid")
        url = (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid
               else f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}")
        out.append({
            "title": r.get("title"), "authors": r.get("authorString"),
            "year": r.get("pubYear"), "journal": r.get("journalTitle"),
            "pmid": pmid, "doi": r.get("doi"), "source": r.get("source"),
            "id": r.get("id"), "url": url,
        })
    return out


def literature_for_variant(cfg: Config, hgvs_text: str, gene: str | None = None,
                           online: bool | None = None) -> LiteratureResult:
    pv = parse_variant(hgvs_text, cfg.variant_class.splice_canonical_offset)
    gene = gene or pv.gene or cfg.genes.focus
    aliases = variant_aliases(pv)
    query = build_query(gene, aliases)

    # Default: do NOT go online in fixture mode (the variant is synthetic) or unless asked.
    if online is None:
        online = cfg.data_sources.mode != "fixture"
    if not online:
        return LiteratureResult(
            online=False, query=query, results=[],
            note=("No live query performed (offline/fixture mode). This is the exact "
                  "Europe PMC query that WOULD run. No findings are fabricated. Re-run "
                  "with online=True (and network access) for real citations."))
    try:
        results = search_europepmc(query)
        note = (f"{len(results)} Europe PMC record(s) for the query. Citations are real; "
                "relevance to this specific residue is not guaranteed — verify each source.")
        return LiteratureResult(online=True, query=query, results=results, note=note)
    except Exception as e:  # network blocked / API error -> honest, no fabrication
        return LiteratureResult(
            online=True, query=query, results=[], error=str(e),
            note="Live query failed (see error). No findings fabricated.")


def to_markdown(res: LiteratureResult, hgvs_text: str, gene: str) -> str:
    L = [f"## Literature context — {gene} {hgvs_text}", "",
         f"- Query: `{res.query}`", f"- {res.note}"]
    if res.error:
        L.append(f"- ⚠️ error: `{res.error}`")
    if res.results:
        L += ["", "| year | title | authors | link |", "|---|---|---|---|"]
        for r in res.results:
            link = r["url"] or ""
            title = (r["title"] or "").replace("|", "/")
            authors = (r["authors"] or "")[:60].replace("|", "/")
            L.append(f"| {r['year'] or ''} | {title} | {authors} | {link} |")
    elif res.online and not res.error:
        L.append("\n_No Europe PMC records matched._")
    return "\n".join(L) + "\n"
