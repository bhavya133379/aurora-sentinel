import sqlite3

conn = sqlite3.connect("aurora_reports.db")
cursor = conn.cursor()

# Try adding image_paths column safely
try:
    cursor.execute("ALTER TABLE reports ADD COLUMN image_paths TEXT")
    print("✅ image_paths column added successfully")
except Exception as e:
    print("⚠ Column may already exist:", e)

conn.commit()
conn.close()