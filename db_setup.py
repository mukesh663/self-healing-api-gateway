import psycopg2

def initialize_pgvector_db():
    # Load the SQL file and execute it
    with open("db_script.sql", "r") as f:
        sql_commands = f.read()
    
    conn = psycopg2.connect(
        host="localhost",
        dbname="your_db",
        user="your_user",
        password="your_password",
        port=5432
    )
    cur = conn.cursor()
    
    cur.execute(sql_commands)
    conn.commit()
    cur.close()
    conn.close()

    print("✅ Database and table initialized successfully.")

if __name__ == "__main__":
    initialize_pgvector_db()
