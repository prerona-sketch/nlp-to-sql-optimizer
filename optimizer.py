
import re
import time
import sqlite3
import sqlglot
from sqlglot import exp


# =====================================================
# EXPLAIN QUERY PLAN
# =====================================================
def get_query_plan(conn, sql):

    cur = conn.cursor()
    cur.execute("EXPLAIN QUERY PLAN " + sql)

    rows = []
    for r in cur.fetchall():
        # SQLite returns (id, parent, notused, detail)
        if len(r) >= 4:
            rows.append({"id": r[0], "parent": r[1], "detail": r[3]})
        else:
            rows.append({"id": r[0], "parent": r[1], "detail": r[-1]})

    return rows


# =====================================================
# DETECT SLOW OPERATIONS
# =====================================================
def detect_slow_ops(plan_rows):

    issues = []

    for row in plan_rows:
        detail = row["detail"]
        up = detail.upper()

        # SEARCH ... USING INDEX / USING COVERING INDEX -> indexed, good
        if up.startswith("SEARCH") and "USING" in up and "INDEX" in up:
            continue

        # SCAN <table> with no index -> full table scan
        if up.startswith("SCAN"):

            m = re.match(r"SCAN\s+(?:TABLE\s+)?(\w+)", detail, re.IGNORECASE)
            table = m.group(1) if m else None

            issue_type = "full_table_scan"

            # child of another row -> inner side of a join
            if row["parent"] and row["parent"] != 0:
                issue_type = "nested_loop_without_index"

            issues.append({
                "table": table,
                "issue": issue_type,
                "detail": detail
            })
            continue

        # SEARCH with no index usage (rare, but flag it)
        if up.startswith("SEARCH") and "INDEX" not in up:
            m = re.match(r"SEARCH\s+(?:TABLE\s+)?(\w+)", detail, re.IGNORECASE)
            table = m.group(1) if m else None

            issues.append({
                "table": table,
                "issue": "search_without_index",
                "detail": detail
            })

    return issues


# =====================================================
# RESOLVE TABLE ALIASES
# =====================================================
def _from_tables(from_node):
    # sqlglot changed where tables live under From across versions.
    # Newer: from_.this is a Table; Older: from_.expressions is [Table, ...].
    tables = []
    if from_node is None:
        return tables
    this = getattr(from_node, "this", None)
    if isinstance(this, exp.Table):
        tables.append(this)
    exprs = getattr(from_node, "expressions", None) or []
    for e in exprs:
        if isinstance(e, exp.Table):
            tables.append(e)
    return tables


def _build_alias_map(expr):

    alias_map = {}

    for t in _from_tables(expr.find(exp.From)):
        name = t.name
        alias = t.alias or name
        alias_map[alias] = name
        alias_map[name] = name

    for j in expr.find_all(exp.Join):
        t = j.this
        if isinstance(t, exp.Table):
            name = t.name
            alias = t.alias or name
            alias_map[alias] = name
            alias_map[name] = name

    return alias_map


# =====================================================
# EXTRACT INDEX CANDIDATES
# =====================================================
def extract_index_candidates(sql):

    candidates = []
    seen = set()

    def _add(table, col):
        if not table or not col:
            return
        key = (table.lower(), col.lower())
        if key in seen:
            return
        seen.add(key)
        candidates.append((table, col))

    try:
        expr = sqlglot.parse_one(sql)
    except Exception:
        return _regex_fallback(sql)

    alias_map = _build_alias_map(expr)

    default_table = None
    from_tables = _from_tables(expr.find(exp.From))
    if from_tables:
        default_table = from_tables[0].name

    sargable_ops = (
        exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.In, exp.Between
    )

    where = expr.find(exp.Where)
    if where:
        for op in where.find_all(*sargable_ops):
            for col in op.find_all(exp.Column):
                tbl = col.table
                resolved = alias_map.get(tbl, tbl) if tbl else default_table
                _add(resolved, col.name)

        # LIKE only if pattern is prefix, e.g. 'abc%' (sargable in SQLite)
        for like in where.find_all(exp.Like):
            cols = list(like.find_all(exp.Column))
            lit = like.args.get("expression")
            if lit is not None and isinstance(lit, exp.Literal):
                val = lit.this or ""
                if not val.startswith("%"):
                    for col in cols:
                        tbl = col.table
                        resolved = alias_map.get(tbl, tbl) if tbl else default_table
                        _add(resolved, col.name)

    # JOIN ON columns
    for j in expr.find_all(exp.Join):
        on = j.args.get("on")
        if on is None:
            continue
        for col in on.find_all(exp.Column):
            tbl = col.table
            resolved = alias_map.get(tbl, tbl) if tbl else default_table
            _add(resolved, col.name)

    return candidates


# =====================================================
# REGEX FALLBACK (only if sqlglot parse fails)
# =====================================================
def _regex_fallback(sql):

    candidates = []
    seen = set()

    def _add(table, col):
        key = ((table or "").lower(), col.lower())
        if key in seen:
            return
        seen.add(key)
        candidates.append((table, col))

    m_from = re.search(r"FROM\s+(\w+)", sql, re.IGNORECASE)
    default_table = m_from.group(1) if m_from else None

    where_m = re.search(
        r"WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|\s+LIMIT|$)",
        sql, re.IGNORECASE
    )
    if where_m:
        where_text = where_m.group(1)
        for m in re.finditer(
            r"(?:(\w+)\.)?(\w+)\s*(?:=|>|<|>=|<=|IN|BETWEEN)",
            where_text, re.IGNORECASE
        ):
            _add(m.group(1) or default_table, m.group(2))

    for m in re.finditer(
        r"JOIN\s+\w+.*?ON\s+(?:(\w+)\.)?(\w+)\s*=\s*(?:(\w+)\.)?(\w+)",
        sql, re.IGNORECASE
    ):
        _add(m.group(1), m.group(2))
        _add(m.group(3), m.group(4))

    return candidates


# =====================================================
# EXISTING INDEXES (skip columns already indexed)
# =====================================================
def _existing_indexed_columns(conn, table):

    indexed = set()
    cur = conn.cursor()

    try:
        cur.execute(f"PRAGMA index_list({table})")
        idx_rows = cur.fetchall()
    except sqlite3.Error:
        return indexed

    for idx in idx_rows:
        idx_name = idx[1]
        try:
            cur.execute(f"PRAGMA index_info({idx_name})")
            for info in cur.fetchall():
                indexed.add(info[2].lower())
        except sqlite3.Error:
            continue

    return indexed


def _table_exists(conn, table):
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name=?",
        (table,)
    )
    return cur.fetchone() is not None


# =====================================================
# RECOMMEND INDEXES
# =====================================================
def recommend_indexes(conn, sql):

    candidates = extract_index_candidates(sql)

    recommendations = []
    seen_stmt = set()

    for table, col in candidates:

        if not _table_exists(conn, table):
            continue

        already = _existing_indexed_columns(conn, table)
        if col.lower() in already:
            continue

        idx_name = f"idx_{table}_{col}"
        stmt = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({col})"

        if stmt in seen_stmt:
            continue
        seen_stmt.add(stmt)

        recommendations.append({
            "table": table,
            "column": col,
            "index_name": idx_name,
            "statement": stmt
        })

    return recommendations


# =====================================================
# TIME A QUERY
# =====================================================
def time_query(conn, sql, runs=50):

    times = []

    for _ in range(runs):
        cur = conn.cursor()
        t0 = time.perf_counter()
        cur.execute(sql)
        cur.fetchall()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    avg_ms = sum(times) / len(times)
    min_ms = min(times)

    return {"avg_ms": avg_ms, "min_ms": min_ms}


# =====================================================
# BENCHMARK: BEFORE vs AFTER
# =====================================================
def benchmark_with_indexes(
    conn, sql, recommendations, runs=50, keep_indexes=True
):

    before = time_query(conn, sql, runs)

    created = []
    for rec in recommendations:
        try:
            conn.execute(rec["statement"])
            created.append(rec["index_name"])
        except sqlite3.Error:
            pass

    conn.commit()

    after = time_query(conn, sql, runs)

    speedup = (
        before["avg_ms"] / after["avg_ms"]
        if after["avg_ms"] > 0 else float("inf")
    )

    if not keep_indexes:
        for name in created:
            try:
                conn.execute(f"DROP INDEX IF EXISTS {name}")
            except sqlite3.Error:
                pass
        conn.commit()
        created = []

    return {
        "before_ms": before["avg_ms"],
        "before_min_ms": before["min_ms"],
        "after_ms": after["avg_ms"],
        "after_min_ms": after["min_ms"],
        "speedup": speedup,
        "created_indexes": created
    }


# =====================================================
# TOP-LEVEL PIPELINE
# =====================================================
def analyze_performance(conn, sql, runs=50, keep_indexes=True):

    plan_before = get_query_plan(conn, sql)
    issues = detect_slow_ops(plan_before)
    recommendations = recommend_indexes(conn, sql)

    bench = benchmark_with_indexes(
        conn, sql, recommendations,
        runs=runs, keep_indexes=keep_indexes
    )

    plan_after = get_query_plan(conn, sql)

    return {
        "sql": sql,
        "plan_before": plan_before,
        "issues": issues,
        "recommended_indexes": [r["statement"] for r in recommendations],
        "recommendations": recommendations,
        "benchmark": bench,
        "plan_after": plan_after
    }


# =====================================================
# PRETTY PRINTER
# =====================================================
def print_performance_report(report):

    print("=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)
    print(f"SQL: {report['sql']}")
    print()

    print("-- Plan (before) --")
    for r in report["plan_before"]:
        print(f"  [{r['id']}|{r['parent']}] {r['detail']}")
    print()

    print("-- Issues detected --")
    if not report["issues"]:
        print("  (none)")
    for iss in report["issues"]:
        print(f"  [{iss['issue']}] table={iss['table']} :: {iss['detail']}")
    print()

    print("-- Recommended indexes --")
    if not report["recommended_indexes"]:
        print("  (none)")
    for s in report["recommended_indexes"]:
        print(f"  {s};")
    print()

    b = report["benchmark"]
    print("-- Benchmark --")
    print(f"  Before  : avg {b['before_ms']:.4f} ms  "
          f"(min {b['before_min_ms']:.4f} ms)")
    print(f"  After   : avg {b['after_ms']:.4f} ms  "
          f"(min {b['after_min_ms']:.4f} ms)")
    print(f"  Speedup : {b['speedup']:.2f}x")
    print(f"  Created : {b['created_indexes']}")
    print()

    print("-- Plan (after) --")
    for r in report["plan_after"]:
        print(f"  [{r['id']}|{r['parent']}] {r['detail']}")
    print("=" * 60)