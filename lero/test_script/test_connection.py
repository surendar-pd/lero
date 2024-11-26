import psycopg2

CONNECTION_STR = "dbname=tpch user=postgres password='Sshprd@007' host=localhost port=5432"

try:
    conn = psycopg2.connect(CONNECTION_STR)
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
