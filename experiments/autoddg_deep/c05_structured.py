"""C05: structure-preserving serialization of the exact F2 profile and facts.

The original UFD prompt carries the table evidence in three natural-language blocks,
each in its own prompt slot:

    basic profile       "**col**: Data is of type T. There are N unique values. Coverage spans from a to b."
    semantic profile    structural profile lines ("**Temporal coverage** for columns ...") + semantic lines
    verified facts      "- Slot title: col: fact; col: fact" (render.format_slots)

``structured_block`` re-serializes the same content as one key-value block grouped by
column (one block per column, one table-level group), with the source of every value
kept as a key (profile / semantic / facts) because the sources can disagree (sample vs
full table). Nothing is recomputed: every value is parsed from the cached F2 strings or
taken from the cached factset render texts, and ``check_lossless`` verifies per column
that every number and value string of the original reaches the column's block.

Field values >= DICT_MIN_CHARS that recur in >= 2 places are written once in a VALUES
dictionary and referenced as &1, &2, ...
"""

from __future__ import annotations

import re
from collections import Counter

DICT_MIN_CHARS = 40  # fixed before the run: shorter shared values save nothing and only add indirection

FACT_KEYS = {
    "table_size": "size",
    "temporal_span": "span",
    "temporal_resolution": "spacing",
    "spatial_places": "places",
    "spatial_level": "level",
    "spatial_bbox": "bbox",
    "measure_range": "range",
    "measure_unit": "unit",
    "category_values": "top_values",
    "missingness": "missing",
    "observation_unit": "observation_unit",
}
SLOTS = ("identity", "temporal_scope", "geographic_scope", "observation_unit",
         "measures_and_units", "representative_categories", "limitations")

HEADER = (
    "Structured profile, one block per column. Keys: profile = data profile, "
    "semantic = semantic profile of the column, facts = facts computed from the full table "
    "(values separated by \" | \"). TABLE holds table-level entries. &n refers to VALUES."
)
SEMANTIC_POINTER = "(Given as the \"semantic\" entries of each column in the structured profile above.)"
UFD_FACTS_POINTER = (
    "(Given as the \"facts\" entries of TABLE and of each column in the structured profile above.)"
)

LINE = re.compile(r"^\*\*(.+?)\*\*:\s?(.*)$")


# --- parsing the cached natural-language blocks ------------------------------------

def _sentences(body: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", body.strip()) if s.strip()]


def parse_basic(text: str) -> tuple[list[str], dict[str, list[tuple[str, str]]]]:
    """basic profile -> column order, column -> [(key, value)]; unknown sentences kept as ``note``."""
    order, out = [], {}
    for line in text.split("\n"):
        m = LINE.match(line.strip())
        if not m:
            if line.strip() and not line.startswith("The key data profile"):
                raise ValueError(f"unparsed basic profile line: {line!r}")
            continue
        col, body = m.group(1), m.group(2)
        order.append(col)
        fields = []
        for s in _sentences(body):
            if (mm := re.fullmatch(r"Data is of type (.+)\.", s)):
                fields.append(("type", mm.group(1)))
            elif (mm := re.fullmatch(r"There (?:are|is) (\S+) unique values?\.", s)):
                fields.append(("unique_values", mm.group(1)))
            elif (mm := re.fullmatch(r"Coverage spans from (.+) to (.+)\.", s)):
                fields.append(("coverage", f"{mm.group(1)} to {mm.group(2)}"))
            else:
                fields.append(("note", s))
        out[col] = fields
    return order, out


SEM_PATTERNS = (
    (r"Represents (.+)\.", "represents"),
    (r"Contains temporal data \(resolution: (.+)\)\.", "temporal_resolution"),
    (r"Contains spatial data \(resolution: (.+)\)\.", "spatial_resolution"),
    (r"Domain-specific type: (.+)\.", "domain"),
    (r"Function/Usage context: (.+)\.", "usage"),
)


def parse_semantic(text: str) -> dict[str, list[tuple[str, str]]]:
    out = {}
    for line in text.split("\n"):
        m = LINE.match(line.strip())
        if not m:
            if line.strip() and not line.startswith("The key semantic"):
                raise ValueError(f"unparsed semantic line: {line!r}")
            continue
        fields = []
        for s in _sentences(m.group(2)):
            for pat, key in SEM_PATTERNS:
                if (mm := re.fullmatch(pat, s)):
                    fields.append((key, mm.group(1)))
                    break
            else:
                fields.append(("note", s))
        out[m.group(1)] = fields
    return out


def _split_columns(spec: str, columns: list[str]) -> list[str] | None:
    """'A, B' -> [A, B] using the known column names (names may contain commas)."""
    spec = spec.strip().rstrip(",").strip()
    found, rest = [], spec
    while rest:
        for c in sorted(columns, key=len, reverse=True):
            if rest == c or rest.startswith(c + ", "):
                found.append(c)
                rest = rest[len(c):].lstrip(",").strip()
                break
        else:
            return None
    return found


def parse_structural(text: str, columns: list[str]):
    """structural profile -> column -> [(key, value)], table-level [(key, value)]."""
    per_col: dict[str, list] = {}
    table: list = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if (m := re.fullmatch(r"\*\*Temporal coverage\*\* for columns (.+?) with resolution (.+?), "
                              r"covering from (.+) to (.+)\.", line)):
            cols = _split_columns(m.group(1), columns)
            val = f"resolution {m.group(2)}, {m.group(3)} to {m.group(4)}"
            key = "temporal_coverage"
        elif (m := re.fullmatch(r"\*\*Spatial coverage\*\* for columns (.+), with type (.+)\.", line)):
            cols = _split_columns(m.group(1), columns)
            val = f"type {m.group(2)}"
            key = "spatial_coverage"
        else:
            table.append(("profile_note", line))
            continue
        if cols and len(cols) == 1:
            per_col.setdefault(cols[0], []).append((key, val))
        elif cols:
            table.append((key, f"columns [{', '.join(cols)}], {val}"))
        else:
            table.append(("profile_note", line))
    return per_col, table


# --- facts --------------------------------------------------------------------------

def fact_line(f: dict, target: str) -> str:
    """The exact line render.slots writes for this fact (render._ufd_line / _sfd_line)."""
    text = f["render"]["ufd"]
    if target == "ufd":
        hedged = f["kind"] == "spatial_places" and not (f.get("value") or {}).get("confirmed")
        return f"likely {text}" if f["status"] == "uncertain" and not hedged else text
    terms = f["render"]["sfd_terms"]
    return text + (f" ({', '.join(terms)})" if terms else "")


def original_fact_block(factset: dict, target: str) -> str:
    """render.format_slots(render.slots(fs, target)) rebuilt from the cached dict (checked against the cache)."""
    chosen = set(factset["budget"].get(f"selected_{target}", []))
    lines: dict[str, list[str]] = {s: [] for s in SLOTS}
    for f in factset["facts"]:
        if f["fact_id"] not in chosen or not f.get("render"):
            continue
        lines[f["render"]["slot"]].append(fact_line(f, target))
        if f["render"].get("limitation"):
            lines["limitations"].append(f["render"]["limitation"])
    titles = {s: s.replace("_", " ").capitalize() for s in SLOTS}
    return "\n".join(f"- {titles[s]}: {'; '.join(v)}" for s, v in lines.items() if v)


def fact_entries(factset: dict, target: str, columns: list[str]):
    """Selected facts -> column -> [(key, text without the 'col: ' prefix)], table [(key, text)]."""
    chosen = set(factset["budget"].get(f"selected_{target}", []))
    per_col: dict[str, list] = {}
    table: list = []
    for f in factset["facts"]:
        if f["fact_id"] not in chosen or not f.get("render"):
            continue
        key = FACT_KEYS.get(f["kind"], f["kind"])
        line = fact_line(f, target)
        cols = f.get("columns") or []
        if len(cols) == 1 and cols[0] in columns:
            col = cols[0]
            prefix = "likely " if line.startswith("likely ") and not f["render"]["ufd"].startswith("likely ") else ""
            body = line[len(prefix):]
            if body.startswith(col + ": "):
                body = body[len(col) + 2:]
            per_col.setdefault(col, []).append((key, prefix + body))
        else:
            table.append((key, line))
        if f["render"].get("limitation"):
            table.append(("limitation", f["render"]["limitation"]))
    return per_col, table


# --- serialization -------------------------------------------------------------------

def _kv(fields: list[tuple[str, str]], sep: str = "; ") -> str:
    return sep.join(f"{k}={v}" for k, v in fields)


def _dictionary(blocks: list[list[tuple[str, str]]]) -> dict[str, str]:
    counts = Counter(v for fields in blocks for _, v in fields if len(v) >= DICT_MIN_CHARS)
    refs, n = {}, 0
    for v, c in counts.items():  # first-occurrence order
        if c >= 2:
            n += 1
            refs[v] = f"&{n}"
    return refs


def structured_block(basic: str, structural: str, semantic_details: str, factset: dict | None,
                     target: str = "ufd", with_profile: bool = True) -> tuple[str, dict]:
    """Return (text, stats). ``with_profile=False`` = facts only (SFD fact slot)."""
    order, basic_f = parse_basic(basic)
    sem_f = parse_semantic(semantic_details) if with_profile else {}
    struct_f, struct_t = parse_structural(structural, order) if with_profile else ({}, [])
    fact_f, fact_t = fact_entries(factset, target, order) if factset else ({}, [])
    unknown = [c for c in list(sem_f) + list(struct_f) + list(fact_f) if c not in order]
    if unknown:
        raise ValueError(f"columns outside the basic profile: {unknown}")

    groups = []  # (column or None, [(source, fields)])
    table = []
    if fact_t:
        table.append(("facts", fact_t))
    if struct_t:
        table.append(("profile", struct_t))
    groups.append((None, table))
    for col in order:
        g = []
        prof = (basic_f.get(col, []) + struct_f.get(col, [])) if with_profile else []
        if prof:
            g.append(("profile", prof))
        if sem_f.get(col):
            g.append(("semantic", sem_f[col]))
        if fact_f.get(col):
            g.append(("facts", fact_f[col]))
        if g or with_profile:
            groups.append((col, g))

    refs = _dictionary([fields for _, g in groups for _, fields in g])
    lines = [] if not with_profile else [HEADER]
    if refs:
        lines.append("VALUES")
        lines.extend(f"  {r} = {v}" for v, r in refs.items())
    for col, g in groups:
        if col is None and not g:
            continue
        lines.append("TABLE" if col is None else f"[{col}]")
        for source, fields in g:
            fields = [(k, refs.get(v, v)) for k, v in fields]
            sep = " | " if source == "facts" else "; "
            lines.append(f"  {source}: {_kv(fields, sep)}")
    stats = {"dict_refs": len(refs), "columns": len(order), "columns_with_facts": len(fact_f),
             "table_fact_entries": len(fact_t), "table_profile_entries": len(struct_t)}
    return "\n".join(lines), stats


def expand_refs(text: str) -> str:
    """Drop the VALUES section and resolve &n references (for the lossless check only)."""
    refs, body, in_values = {}, [], False
    for line in text.split("\n"):
        if line == "VALUES":
            in_values = True
            continue
        if in_values and (m := re.fullmatch(r"  (&\d+) = (.*)", line)):
            refs[m.group(1)] = m.group(2)
            continue
        in_values = False
        body.append(line)
    out = "\n".join(body)
    for r in sorted(refs, key=len, reverse=True):
        out = re.sub(re.escape(r) + r"(?!\d)", lambda _: refs[r], out)
    return out


# --- lossless check -------------------------------------------------------------------

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _atoms(text: str) -> Counter:
    """Numbers and alphabetic words (lower case), the units of information the check counts."""
    nums = NUM.findall(text)
    words = re.findall(r"[a-z][a-z'&]+", NUM.sub(" ", text.lower()))
    return Counter(nums) + Counter(words)


BOILER = set("""data is of type there are unique values coverage spans from to represents contains temporal spatial
resolution domain specific function usage context the key profile information for this dataset includes semantic
columns with covering identity scope geographic measures and units representative categories limitations""".split())


def _blocks(structured: str) -> dict[str, str]:
    blocks, cur = {}, "TABLE"
    for line in expand_refs(structured).split("\n"):
        if line == "TABLE":
            cur = "TABLE"
        elif (m := re.fullmatch(r"\[(.+)\]", line)):
            cur = m.group(1)
        blocks[cur] = blocks.get(cur, "") + line + "\n"
    return blocks


def _need(piece: str) -> Counter:
    a = _atoms(piece)
    for k in [k for k in a if k in BOILER]:
        del a[k]
    return a


def check_lossless(basic: str, structural: str, semantic_details: str, factset: dict | None,
                   structured: str, target: str = "ufd", with_profile: bool = True) -> dict:
    """Per column: atoms of every original line about the column that are absent from that
    column's block (multi-column / table-level lines: from the TABLE block).

    Returns {"missing": {column|TABLE: [atoms]}, "n_atoms": int, "n_missing": int}.
    """
    blocks = _blocks(structured)
    order, _ = parse_basic(basic)
    want: dict[str, Counter] = {}
    if with_profile:
        for src in (basic, semantic_details):
            for line in src.split("\n"):
                if (m := LINE.match(line.strip())):
                    want.setdefault(m.group(1), Counter()).update(_need(m.group(2)))
        for line in structural.split("\n"):
            if not line.strip():
                continue
            m = re.match(r"\*\*\w+ coverage\*\* for columns (.+?)(?: with resolution|, with type)", line)
            cols = _split_columns(m.group(1), order) if m else None
            tgt = cols[0] if cols and len(cols) == 1 else "TABLE"
            piece = re.sub(r"\*\*.*?\*\*", "", line)
            if tgt != "TABLE":
                piece = piece.replace(f"columns {tgt}", "columns", 1)  # the name is the block header
            want.setdefault(tgt, Counter()).update(_need(piece))
    if factset:
        chosen = set(factset["budget"].get(f"selected_{target}", []))
        for f in factset["facts"]:
            if f["fact_id"] in chosen and f.get("render"):
                cols = f.get("columns") or []
                tgt = cols[0] if len(cols) == 1 and cols[0] in order else "TABLE"
                line = fact_line(f, target)
                if tgt != "TABLE":  # the name is the block header
                    line = re.sub(r"^(likely )?" + re.escape(tgt) + ": ", r"\1", line)
                want.setdefault(tgt, Counter()).update(_need(line))
                if f["render"].get("limitation"):
                    want.setdefault("TABLE", Counter()).update(_need(f["render"]["limitation"]))
    missing, n_atoms = {}, 0
    for col, atoms in want.items():
        have = _atoms(blocks.get(col, "")) + _atoms(col)
        lost = sorted(a for a in atoms if have[a] < atoms[a])
        n_atoms += sum(atoms.values())
        if lost:
            missing[col] = lost
    return {"missing": missing, "n_atoms": n_atoms, "n_missing": sum(len(v) for v in missing.values())}


# --- position-aligned variant ---------------------------------------------------------
# Each evidence block stays in its own original prompt slot with the original slot text
# around it; only the lines about columns are rewritten from sentences to key=value lines
# grouped by column. No source keys (one source per slot), no VALUES dictionary, no header.

def _group_lines(groups: list[tuple[str | None, list[tuple[str, str]]]], sep: str = "; ") -> list[str]:
    out = []
    for col, fields in groups:
        if fields:
            out += ["TABLE" if col is None else f"[{col}]", "  " + _kv(fields, sep)]
    return out


def aligned_profile(basic: str) -> str:
    """Basic-profile slot: non-column lines verbatim, each column line -> [col] + key=value line."""
    _, fields = parse_basic(basic)
    out = []
    for line in basic.split("\n"):
        m = LINE.match(line.strip())
        out += _group_lines([(m.group(1), fields[m.group(1)])]) if m else [line]
    return "\n".join(out)


def aligned_semantic(basic: str, structural: str, semantic_details: str) -> str:
    """Semantic-profile slot (= structural profile + semantic details in the original):
    the semantic header verbatim, then TABLE (multi-column structural lines) and one
    group per column with its structural and semantic entries."""
    order, _ = parse_basic(basic)
    sem = parse_semantic(semantic_details)
    st_col, st_table = parse_structural(structural, order)
    head = [l for l in semantic_details.split("\n") if l.strip() and not LINE.match(l.strip())]
    groups = [(None, st_table)] + [(c, st_col.get(c, []) + sem.get(c, [])) for c in order]
    body = _group_lines(groups)
    # columns that the semantic profile lists with no content stay listed, as in the original
    listed = set(sem) | set(st_col)
    for c in order:
        if c in listed and not (st_col.get(c) or sem.get(c)):
            body.append(f"[{c}]")
    return "\n".join(head + body)


def aligned_facts(factset: dict, target: str, columns: list[str]) -> str:
    """Fact slot: TABLE + one group per column, facts separated by ' | '."""
    fc, ft = fact_entries(factset, target, columns)
    return "\n".join(_group_lines([(None, ft)] + [(c, fc.get(c, [])) for c in columns], " | "))
