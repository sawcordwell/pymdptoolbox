# -*- coding: utf-8 -*-

import sqlite3

class DatabaseManager(object):
    """"""
    
    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()
        self.cur.execute("SELECT value FROM info WHERE name='states'")
        try:
            self.S = self.cur.fetchone()[0]
        except TypeError:
            raise ValueError("Cannot determine number of states from database. "
                             "There is no name 'states' in table 'info'.")
        self.cur.execute("SELECT value FROM info WHERE name='actions'")
        try:
            self.A = self.cur.fetchone()[0]
        except TypeError:
            raise ValueError("Cannot determine number of actions from database. "
                             "There is no name 'actions' in table 'info'.")
        self._initQ()
    
    def _initQ(self):
        #tbl = ", ".join("q%s REAL" % a for a in range(self.A))
        #self.cur.execute("CREATE TABLE Q (" + tbl + ")")
        # I don't think there is a way to do max() and argmax() if I use the
        # above schema.
        self.cur.execute("CREATE TABLE Q (state INTEGER, action INTEGER, value REAL)")
    
    def initValue(self, V=0):
        self.cur.execute("CREATE TABLE results (policy INTEGER, value REAL)")
        cmd = "INSERT INTO results(value) VALUES(?)"
        if V==0:
            V = [0] * self.S
            self.cur.executemany(cmd, zip(V))
        else:
            try:
                self.cur.executemany(cmd, zip(V))
            except:
                raise ValueError("V is of unsupported type, use a list or tuple.")
        self.conn.commit()
    
    def __del__(self):
        self.cur.close()
        self.conn.close()
    
    def bellmanOperator(self):
        pass
    
    def calculatePolicy(self):
        """This implements argmax() over the actions of Q."""
        cmd = '''SELECT action
                   FROM Q
                  WHERE state = (SELECT MAX('''
        self.cur.execute(cmd)
    
    def calculateValue(self):
        """This is max() over the actions of Q."""
        pass
    
    def dot(self, a, b):
        pass
    
    def getPolicyValue(self):
        self.cur.execute("SELECT * FROM results")
        r = self.cur.fetchall()
        policy = [x[0] for x in r]
        value = [x[1] for x in r]
        return policy, value
    
    def randomQ(self):
        from numpy.random import random
        for a in range(self.A):
            state = range(self.S)
            action = [a] * self.S
            value = random(self.S).tolist()
            cmd = "INSERT INTO Q VALUES(?, ?, ?)"
            self.cur.executemany(cmd, zip(state, action, value))
        self.conn.commit()
