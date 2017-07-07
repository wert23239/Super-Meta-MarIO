import sqlite3

conn= sqlite3.connect('tutorial.db')

c= conn.cursor()

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS ReplayMemory(unix REAL,image TEXT, value REAL)')


def data_entry():
    c.execute("INSERT INTO ReplayMemory VALUES(145123542,'[2,3,4,5]',3.5)")
    conn.commit()
    c.close()
    conn.close()
create_table()
data_entry()    


