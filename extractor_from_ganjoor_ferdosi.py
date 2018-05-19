# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:49:17 2018

@author: Morteza
"""
import datetime
import sqlite3
from sqlite3 import Error
 
 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None
 
 
 
 
def select_verse_by_poem_id(conn, start_poem_id, end_poem_id):
    """
 
    """
    cur = conn.cursor()
    cur.execute("SELECT text FROM verse WHERE poem_id between ? and ?", (start_poem_id,end_poem_id))
 
    rows = cur.fetchall()
#    print(rows[0][0] + '\\t' + rows[1][0])
    
#    my_string = ' '.join(map(str, rows))
#    print(my_string)
    
    s = ''
    for i in range(0,len(rows)-1,2):
         s += str(rows[i][0]) + '\t' + str(rows[i+1][0]) + '\n'
    s = s.replace('\\u200c', ' ')
         
    print(len(rows))
    #print(s)
    cancat_file = './ferdosi_poem_1.txt'
    with open(cancat_file, 'w', encoding='utf8') as cf:
         cf.write(str(s))


def select_verse_by_poem_id_2(conn, start_poem_id, end_poem_id):
    """
 
    """
    cur = conn.cursor()
    cur.execute("SELECT text FROM verse WHERE poem_id between ? and ?", (start_poem_id,end_poem_id))
 
    rows = cur.fetchall()
#    print(rows[0][0] + '\\t' + rows[1][0])
    
#    my_string = ' '.join(map(str, rows))
#    print(my_string)
    
    s = ''
    for i in range(0,len(rows)-2,2):
         s += str(rows[i][0]) + '\t' + str(rows[i+1][0]) + '\n'
         s += str(rows[i+1][0]) + '\t' + str(rows[i+2][0]) + '\n'
    s = s.replace('\\u200c', ' ')
         
    print(len(rows))
    #print(s)
    cancat_file = './ferdosi_poem_2.txt'
    with open(cancat_file, 'w', encoding='utf8') as cf:
         cf.write(str(s))
 
def main():
    database = ".\ganjoor.s3db"
 
    # create a database connection
    conn = create_connection(database)
    with conn:
        print("1. Query task by priority:")
        select_verse_by_poem_id_2(conn, 1321, 1940)
 
#        print("2. Query all tasks")
        #select_all_tasks(conn)
 
 
if __name__ == '__main__':
    main()
    
