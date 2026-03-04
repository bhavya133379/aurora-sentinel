import sqlite3

conn = sqlite3.connect("aurora_reports.db")
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE reports ADD COLUMN priority TEXT")
    print("✅ priority column added successfully")
except sqlite3.OperationalError as e:
    print("⚠️", e)

conn.commit()
conn.close()