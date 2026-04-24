import re
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModel
import sqlglot
from sqlglot import exp


MODEL_NAME = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def encode_text_to_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[:, 0, :]


def classify_query_intent(question, schema=""):
    question_lower = question.lower()

    if "how many" in question_lower or "count" in question_lower:
        return "COUNT"

    if "max" in question_lower or "highest" in question_lower or "average" in question_lower:
        return "AGGREGATE"

    return "SELECT"


def parse_sql_to_ast(sql):
    return sqlglot.parse_one(sql)


def extract_sql_components(sql):
    result = {}

    try:
        expr = parse_sql_to_ast(sql)
    except:
        return result

    select_clause = expr.find(exp.Select)
    if select_clause:
        result["select"] = [e.sql() for e in select_clause.expressions]

    from_clause = expr.find(exp.From)
    if from_clause:
        result["from"] = from_clause.this.sql()

    where_clause = expr.find(exp.Where)
    if where_clause:
        result["where"] = where_clause.this.sql()

    joins = []
    for join in expr.find_all(exp.Join):
        joins.append({
            "type": join.args.get("kind", "INNER"),
            "table": join.this.sql(),
            "on": join.args.get("on").sql() if join.args.get("on") else None
        })

    if joins:
        result["joins"] = joins

    group_clause = expr.find(exp.Group)
    if group_clause:
        result["group_by"] = [e.sql() for e in group_clause.expressions]

    having_clause = expr.find(exp.Having)
    if having_clause:
        result["having"] = having_clause.this.sql()

    order_clause = expr.find(exp.Order)
    if order_clause:
        result["order_by"] = [e.sql() for e in order_clause.expressions]

    return result


def parse_database_schema(schema_text):
    schema = {}

    table_patterns = re.findall(r'(\w+)\((.*?)\)', schema_text)

    for table_name, columns_text in table_patterns:
        schema[table_name] = [col.strip() for col in columns_text.split(",")]

    return schema


def fetch_distinct_column_values(conn, table, columns, limit=100):
    values = {}
    cursor = conn.cursor()

    for column in columns:
        try:
            cursor.execute(
                f"SELECT DISTINCT {column} FROM {table} "
                f"WHERE {column} IS NOT NULL LIMIT {limit}"
            )

            distinct_values = [str(row[0]) for row in cursor.fetchall()]
            values[column] = distinct_values

        except:
            values[column] = []

    return values


def infer_numeric_columns(conn, table, columns):
    numeric_columns = []
    cursor = conn.cursor()

    for column in columns:
        try:
            cursor.execute(f"SELECT {column} FROM {table} LIMIT 5")
            rows = cursor.fetchall()

            for row in rows:
                if row[0] is not None:
                    if isinstance(row[0], (int, float)):
                        numeric_columns.append(column)
                    break
        except:
            pass

    return numeric_columns


def extract_components_from_question(question, schema_text, conn):
    question_lower = question.lower()
    schema = parse_database_schema(schema_text)

    components = {
        "table": None,
        "columns": [],
        "filters": []
    }

    table_name = detect_table_from_question(question_lower, schema)
    components["table"] = table_name or list(schema.keys())[0]

    table = components["table"]
    all_columns = schema[table]

    detected_columns = detect_columns_from_question(question_lower, all_columns)
    components["columns"] = detected_columns if detected_columns else ["*"]

    column_values = fetch_distinct_column_values(conn, table, all_columns)

    value_based_filters = extract_value_filters(question_lower, all_columns, column_values)
    components["filters"].extend(value_based_filters)

    numeric_filters = extract_numeric_filters(question_lower, all_columns)
    components["filters"].extend(numeric_filters)

    fallback_numeric_filter = extract_fallback_numeric_filter(
        question_lower, 
        conn, 
        table, 
        all_columns,
        components["filters"]
    )
    if fallback_numeric_filter:
        components["filters"].append(fallback_numeric_filter)

    return components


def detect_table_from_question(question_lower, schema):
    for table_name in schema:
        singular_form = table_name[:-1] if table_name.endswith("s") else table_name

        if table_name.lower() in question_lower or singular_form.lower() in question_lower:
            return table_name

    return None


def detect_columns_from_question(question_lower, columns):
    detected_columns = []

    for column in columns:
        if column.lower() in question_lower:
            detected_columns.append(column)

    if "names" in question_lower and "name" in columns and "name" not in detected_columns:
        detected_columns.append("name")

    return detected_columns


def extract_value_filters(question_lower, columns, column_values):
    filters = []

    for column, values in column_values.items():
        for value in values:
            if str(value).lower() in question_lower:
                if str(value).isnumeric():
                    continue

                filters.append(f"{column} = '{value}'")

    return filters


def extract_numeric_filters(question_lower, columns):
    filters = []

    for column in columns:
        pattern = rf"{column.lower()}\s*(>=|<=|>|<|=)\s*(\d+)"
        match = re.search(pattern, question_lower)

        if match:
            operator = match.group(1)
            numeric_value = match.group(2)
            filters.append(f"{column} {operator} {numeric_value}")

    return filters


def extract_fallback_numeric_filter(question_lower, conn, table, columns, existing_filters):
    numeric_patterns = re.findall(r'(>=|<=|>|<)\s*(\d+)', question_lower)

    numeric_columns = infer_numeric_columns(conn, table, columns)

    if numeric_patterns and len(numeric_columns) == 1:
        for operator, value in numeric_patterns:
            candidate_filter = f"{numeric_columns[0]} {operator} {value}"

            if candidate_filter not in existing_filters:
                return candidate_filter

    return None


def build_sql_query(components, intent):
    if not components:
        return None

    table = components["table"]

    if intent == "COUNT":
        query = f"SELECT COUNT(*) FROM {table}"
    else:
        select_clause = ", ".join(components["columns"])
        query = f"SELECT {select_clause} FROM {table}"

    if components["filters"]:
        where_clause = " AND ".join(components["filters"])
        query += f" WHERE {where_clause}"

    return query


def validate_and_rewrite_query(question, schema, old_sql, conn):
    intent = classify_query_intent(question, schema)

    old_sql_components = extract_sql_components(old_sql)

    question_components = extract_components_from_question(question, schema, conn)

    final_components = {
        "table": question_components["table"] or old_sql_components.get("from"),
        "columns": question_components["columns"] if question_components["columns"] else old_sql_components.get("select", ["*"]),
        "filters": question_components["filters"]
    }

    corrected_sql = build_sql_query(final_components, intent)

    return {
        "question": question,
        "old_sql": old_sql,
        "old_parts": old_sql_components,
        "question_parts": question_components,
        "intent": intent,
        "corrected_sql": corrected_sql
    }
