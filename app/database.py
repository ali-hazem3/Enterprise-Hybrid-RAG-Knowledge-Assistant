import pyodbc

from app.config import (
    SQL_SERVER,
    SQL_PORT,
    SQL_DATABASE,
    SQL_USERNAME,
    SQL_PASSWORD,
    SQL_DRIVER,
)


def get_connection():
    connection_string = (
        f"DRIVER={{{SQL_DRIVER}}};"
        f"SERVER={SQL_SERVER},{SQL_PORT};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USERNAME};"
        f"PWD={SQL_PASSWORD};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=yes;"
    )

    return pyodbc.connect(connection_string)


def execute_select_query(query: str):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(query)

        columns = [
            column[0]
            for column in cursor.description
        ]

        rows = cursor.fetchall()

        return [
            dict(zip(columns, row))
            for row in rows
        ]

    finally:
        cursor.close()
        connection.close()


def get_database_schema():
    query = """
    SELECT
        TABLE_SCHEMA,
        TABLE_NAME,
        COLUMN_NAME,
        DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    ORDER BY
        TABLE_SCHEMA,
        TABLE_NAME,
        ORDINAL_POSITION;
    """

    rows = execute_select_query(query)

    schema = {}

    for row in rows:
        table_name = (
            f"{row['TABLE_SCHEMA']}."
            f"{row['TABLE_NAME']}"
        )

        if table_name not in schema:
            schema[table_name] = []

        schema[table_name].append(
            {
                "column_name": row["COLUMN_NAME"],
                "data_type": row["DATA_TYPE"],
            }
        )

    return schema

def get_plan_names():
    query = """
    SELECT plan_name
    FROM dbo.plans
    ORDER BY plan_name;
    """

    rows = execute_select_query(query)

    return [
        row["plan_name"]
        for row in rows
    ]