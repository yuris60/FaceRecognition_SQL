import sqlite3
conn = sqlite3.connect('database.db')
c = conn.cursor()
sql = """
DROP TABLE IF EXISTS users;
CREATE TABLE users (
           id integer unique primary key autoincrement,
           nim text, 
           nama text,
           kelas text
);
"""
c.executescript(sql)
conn.commit()
conn.close()