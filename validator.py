import re
import sqlite3
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import sqlglot
from sqlglot import exp

MODEL_NAME = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def encode_text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def cosine_sim(a, b):
    return F.cosine_similarity(a, b).item()

def classify_query_intent(question):
    labels = {
        "COUNT": "how many total number count quantity",
        "AGGREGATE": "highest lowest maximum minimum average mean sum total",
        "SELECT": "show list display records details names data"
    }

    q_emb = encode_text_to_embedding(question)

    best_label = None
    best_score = -1

    for label, text in labels.items():
        l_emb = encode_text_to_embedding(text)
        score = cosine_sim(q_emb, l_emb)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label

def parse_database_schema(schema_text):
    schema = {}
    table_patterns = re.findall(r'(\w+)\((.*?)\)', schema_text)
    for table, cols in table_patterns:
        schema[table] = [c.strip() for c in cols.split(",")]
    return schema

def detect_tables_from_question(question, schema, top_k=3, threshold=0.45):
    q_emb = encode_text_to_embedding(question)
    scores = []
    for t in schema:
        t_emb = encode_text_to_embedding(t)
        scores.append((t, cosine_sim(q_emb, t_emb)))
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, s in scores if s >= threshold][:top_k]
    return selected if selected else [scores[0][0]]

def detect_columns_from_question(question, columns):
    q_emb = encode_text_to_embedding(question)
    out = []
    for c in columns:
        c_emb = encode_text_to_embedding(c.replace("_", " "))
        if cosine_sim(q_emb, c_emb) > 0.45:
            out.append(c)
    return out if out else ["*"]

def extract_value_filters(question, columns, column_values):
    filters = []
    q_emb = encode_text_to_embedding(question)
    for col, vals in column_values.items():
        for v in vals:
            if str(v).isnumeric():
                continue
            v_emb = encode_text_to_embedding(str(v))
            if str(v).lower() in question.lower() or cosine_sim(q_emb, v_emb) > 0.55:
                filters.append(f"{col} = '{v}'")
    return filters

def extract_numeric_filters(question, columns):
    filters = []
    for col in columns:
        m = re.search(rf"{col.lower()}\s*(>=|<=|>|<|=)\s*(\d+)", question.lower())
        if m:
            filters.append(f"{col} {m.group(1)} {m.group(2)}")
    return filters

def extract_joins_from_schema(schema):
    joins = []
    tables = list(schema.keys())

    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            t1, t2 = tables[i], tables[j]

            for c1 in schema[t1]:
                for c2 in schema[t2]:
                    if c1 == c2 and "id" in c1:
                        joins.append(f"{t1}.{c1} = {t2}.{c2}")

    return list(set(joins))

def fetch_column_values(conn, table, columns, limit=50):
    out = {}
    cur = conn.cursor()
    for c in columns:
        try:
            cur.execute(f"SELECT DISTINCT {c} FROM {table} WHERE {c} IS NOT NULL LIMIT {limit}")
            out[c] = [str(r[0]) for r in cur.fetchall()]
        except:
            out[c] = []
    return out

def infer_numeric_columns(conn, table, columns):
    numeric = []
    cur = conn.cursor()
    for c in columns:
        try:
            cur.execute(f"SELECT {c} FROM {table} LIMIT 5")
            for r in cur.fetchall():
                if r[0] is not None and isinstance(r[0], (int, float)):
                    numeric.append(c)
                    break
        except:
            pass
    return numeric

def extract_components_from_question(question, schema_text, conn):
    schema = parse_database_schema(schema_text)

    tables = detect_tables_from_question(question, schema)

    all_columns = []
    for t in tables:
        all_columns += schema[t]

    components = {
        "tables": tables,
        "columns": detect_columns_from_question(question, all_columns),
        "filters": [],
        "joins": extract_joins_from_schema(schema)
    }

    column_values = {}
    for t in tables:
        column_values.update(fetch_column_values(conn, t, schema[t]))

    components["filters"] += extract_value_filters(question, all_columns, column_values)
    components["filters"] += extract_numeric_filters(question, all_columns)

    return components

def build_sql_query(components, intent):
    tables = components["tables"]

    select_cols = components["columns"]
    if intent == "COUNT":
        select = "COUNT(*)"
    elif intent == "AGGREGATE":
        select = f"MAX({select_cols[0]})" if select_cols != ["*"] else "MAX(*)"
    else:
        select = ", ".join(
            [f"{tables[0]}.{c}" if c != "*" else f"{tables[0]}.*" for c in select_cols]
        )

    query = f"SELECT {select} FROM {tables[0]}"

    for i in range(1, len(tables)):
        join = components["joins"][i - 1] if i - 1 < len(components["joins"]) else None
        if join:
            query += f" JOIN {tables[i]} ON {join}"

    if components["filters"]:
        query += " WHERE " + " AND ".join(components["filters"])

    return query

def is_safe_sql(query):
    bad = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
    return not any(b in query.upper() for b in bad)

def validate_and_rewrite_query(question, schema_text, old_sql, conn):
    intent = classify_query_intent(question)

    components = extract_components_from_question(question, schema_text, conn)

    sql = build_sql_query(components, intent)

    if not is_safe_sql(sql):
        return {"error": "unsafe sql"}

    return {
        "question": question,
        "old_sql": old_sql,
        "intent": intent,
        "corrected_sql": sql,
        "components": components
    }
