"""Compare the OpenAPI documents of a Python chapkit service and a chapkit-rs service.

    python3 scripts/compare_openapi.py http://127.0.0.1:8001 http://127.0.0.1:8000

Reports, for the chapkit routes both serve: routes only one side has, status codes that differ,
and request/response body fields that differ. Exits 1 when a route chap-core relies on is missing
or differs, so it can gate CI.
"""

import json
import sys
import urllib.request

# Routes chap-core calls (chap_core/models/chapkit_rest_api_wrapper.py). These must match.
CHAP_CORE_ROUTES = {
    ("get", "/health"),
    ("get", "/api/v1/info"),
    ("get", "/api/v1/configs/$schema"),
    ("get", "/api/v1/configs"),
    ("post", "/api/v1/configs"),
    ("delete", "/api/v1/configs/{id}"),
    ("post", "/api/v1/ml/$train"),
    ("post", "/api/v1/ml/$predict"),
    ("get", "/api/v1/jobs/{id}"),
    ("get", "/api/v1/artifacts/{id}"),
}
METHODS = {"get", "post", "put", "delete", "patch"}


def load(url):
    with urllib.request.urlopen(url.rstrip("/") + "/openapi.json", timeout=10) as r:
        return json.load(r)


def norm(path):
    """Normalize path parameter names: {entity_id}, {job_id}, {id} all become {id}."""
    parts = [("{id}" if p.startswith("{") else p) for p in path.split("/")]
    return "/".join(parts)


def resolve(doc, schema, depth=0):
    if not isinstance(schema, dict) or depth > 8:
        return {}
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return resolve(doc, doc.get("components", {}).get("schemas", {}).get(name, {}), depth + 1)
    if "allOf" in schema:
        merged = {}
        for s in schema["allOf"]:
            merged.update(resolve(doc, s, depth + 1).get("properties", {}))
        return {"properties": merged}
    for key in ("anyOf", "oneOf"):
        if key in schema:
            alts = [resolve(doc, s, depth + 1) for s in schema[key]]
            alts = [a for a in alts if a and a.get("type") != "null"]
            return alts[0] if len(alts) == 1 else {"alternatives": alts}
    if schema.get("type") == "array" and "items" in schema:
        return {"array_of": resolve(doc, schema["items"], depth + 1)}
    return schema


def fields(doc, schema):
    s = resolve(doc, schema)
    if "alternatives" in s:
        return sorted((json.dumps(fields(doc, a)) for a in s["alternatives"]))
    if "array_of" in s:
        return {"[]": sorted(s["array_of"].get("properties", {}).keys())}
    return sorted(s.get("properties", {}).keys())


def routes(doc):
    out = {}
    for path, item in doc.get("paths", {}).items():
        for method, op in item.items():
            if method not in METHODS:
                continue
            body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            responses = {}
            for code, r in op.get("responses", {}).items():
                content = r.get("content", {})
                schema = (content.get("application/json") or next(iter(content.values()), {})).get("schema")
                responses[code] = fields(doc, schema) if schema else None
            out[(method, norm(path))] = {"request": fields(doc, body) if body else None, "responses": responses}
    return out


def main():
    py_url, rs_url = sys.argv[1], sys.argv[2]
    py, rs = routes(load(py_url)), routes(load(rs_url))
    failures = 0
    print(f"python chapkit: {len(py)} operations, chapkit-rs: {len(rs)} operations\n")
    only_py = sorted(set(py) - set(rs))
    only_rs = sorted(set(rs) - set(py))
    if only_py:
        print("Only in Python chapkit:")
        for m, p in only_py:
            flag = "  REQUIRED BY CHAP-CORE" if (m, p) in CHAP_CORE_ROUTES else ""
            failures += bool(flag)
            print(f"  {m.upper():6} {p}{flag}")
        print()
    if only_rs:
        print("Only in chapkit-rs:")
        for m, p in only_rs:
            print(f"  {m.upper():6} {p}")
        print()
    print("Shared operations:")
    for key in sorted(set(py) & set(rs)):
        a, b = py[key], rs[key]
        notes = []
        if a["request"] != b["request"] and a["request"] is not None and b["request"] is not None:
            notes.append(f"request fields py={a['request']} rs={b['request']}")
        ok_a = {c: f for c, f in a["responses"].items() if c.startswith("2")}
        ok_b = {c: f for c, f in b["responses"].items() if c.startswith("2")}
        if set(ok_a) != set(ok_b):
            notes.append(f"success codes py={sorted(ok_a)} rs={sorted(ok_b)}")
        for code in set(ok_a) & set(ok_b):
            fa, fb = ok_a[code], ok_b[code]
            if fa and fb and fa != fb:
                notes.append(f"{code} fields py={fa} rs={fb}")
        critical = key in CHAP_CORE_ROUTES
        mark = "ok  " if not notes else ("DIFF" if not critical else "FAIL")
        failures += critical and bool(notes)
        print(f"  [{mark}] {key[0].upper():6} {key[1]}{'  (chap-core)' if critical else ''}")
        for n in notes:
            print(f"           {n}")
    print(f"\n{'FAILED' if failures else 'OK'}: {failures} chap-core route problem(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
