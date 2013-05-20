# -*- coding: utf-8 -*-

import sqlite3

class MDPSQLite(object):
    """"""
    
    def __init__(self, db, discount, initial_V=0):
        self.discount = discount
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
        self._initResults(initial_V)
    
    def _initQ(self):
        self.cur.executescript('''
            DROP TABLE IF EXISTS Q;
            CREATE TABLE Q (state INTEGER, action INTEGER, value REAL);''')
        for a in range(self.A):
            state = range(self.S)
            action = [a] * self.S
            value = [None] * self.S
            cmd = "INSERT INTO Q VALUES(?, ?, ?)"
            self.cur.executemany(cmd, zip(state, action, value))
        self.conn.commit()
    
    def _initResults(self, initial_V):
        self.cur.executescript('''
            DROP TABLE IF EXISTS policy;
            DROP TABLE IF EXISTS V;
            CREATE TABLE policy (state INTEGER, action INTEGER);
            CREATE TABLE V (state INTEGER, value REAL);''')
        cmd1 = "INSERT INTO V(state, value) VALUES(?, ?)"
        cmd2 = "INSERT INTO policy(state, action) VALUES(?, ?)"
        state = range(self.S)
        action = [None] * self.S
        self.cur.executemany(cmd2, zip(state, action))
        if initial_V==0:
            V = [0] * self.S
            self.cur.executemany(cmd1, zip(state, V))
        else:
            try:
                self.cur.executemany(cmd1, zip(state, V))
            except:
                raise ValueError("V is of unsupported type, use a list or tuple.")
        self.conn.commit()
    
    def __del__(self):
        self.cur.executescript('''
            DROP TABLE IF EXISTS Q;
            DROP TABLE IF EXISTS V;
            DROP TABLE IF EXISTS policy;''')
        self.cur.close()
        self.conn.close()
    
    def bellmanOperator(self):
        g = str(self.discount)
        for a in range(self.A):
            P = "transition%s" % a
            R = "reward%s" % a
            cmd = "" \
"UPDATE Q " \
"   SET value = (" \
"       SELECT value "\
"              FROM (" \
"                   SELECT R.state AS state, (R.val + B.val) AS value " \
"                     FROM "+R+" AS R, (" \
"                          SELECT P.row, "+g+"*SUM(P.prob * V.value) AS val" \
"                            FROM "+P+" AS P, V " \
"                           WHERE V.state = P.col " \
"                           GROUP BY P.row" \
"                          ) AS B " \
"                    WHERE R.state = B.row" \
"                   ) AS C "\
"        WHERE Q.state = C.state) "\
" WHERE action = "+str(a)+";"
            self.cur.execute(cmd)
        self.conn.commit()
        self.calculateValue()
    
    def calculatePolicy(self):
        """This implements argmax() over the actions of Q."""
        cmd = '''SELECT state, action
                   FROM (SELECT state, action, MAX(value)
                           FROM Q
                          GROUP BY state)
                  GROUP BY state;'''
        self.cur.execute(cmd)
        self.conn.commit()
    
    def calculateValue(self):
        """This is max() over the actions of Q."""
        cmd = '''
              UPDATE V
                 SET value = (
                     SELECT MAX(value)
                       FROM Q
                      WHERE V.state = Q.state
                      GROUP BY state);'''
        self.cur.execute(cmd)
        self.conn.commit()
    
    def getPolicyValue(self):
        """Get the policy and value vectors."""
        self.cur.execute("SELECT action FROM policy")
        r = self.cur.fetchall()
        policy = [x[0] for x in r]
        self.cur.execute("SELECT value FROM V")
        r = self.cur.fetchall()
        value = [x[0] for x in r]
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

class ValueIterationSQLite(DatabaseManager):
    pass
