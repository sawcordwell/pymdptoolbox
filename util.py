# -*- coding: utf-8 -*-

import sqlite3

from time import time

class MDPSQLite(object):
    """"""
    
    def __init__(self, db, discount, epsilon, max_iter, initial_V=0):
        self.discount = discount
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.itr = 0
        # The database stuff
        self._conn = sqlite3.connect(db)
        self._cur = self._conn.cursor()
        self._cur.execute("SELECT value FROM info WHERE name='states'")
        try:
            self.S = self._cur.fetchone()[0]
        except TypeError:
            raise ValueError("Cannot determine number of states from database. "
                             "There is no name 'states' in table 'info'.")
        self._cur.execute("SELECT value FROM info WHERE name='actions'")
        try:
            self.A = self._cur.fetchone()[0]
        except TypeError:
            raise ValueError("Cannot determine number of actions from database. "
                             "There is no name 'actions' in table 'info'.")
        self._initQ()
        self._initResults(initial_V)
    
    def _initQ(self):
        self._cur.executescript('''
            DROP TABLE IF EXISTS Q;
            CREATE TABLE Q (state INTEGER, action INTEGER, value REAL);''')
        for a in range(self.A):
            state = range(self.S)
            action = [a] * self.S
            value = [None] * self.S
            cmd = "INSERT INTO Q VALUES(?, ?, ?)"
            self._cur.executemany(cmd, zip(state, action, value))
        self._conn.commit()
    
    def _initResults(self, initial_V):
        self._cur.executescript('''
            DROP TABLE IF EXISTS policy;
            DROP TABLE IF EXISTS V;
            DROP TABLE IF EXISTS Vprev;
            CREATE TABLE policy (state INTEGER, action INTEGER);
            CREATE TABLE V (state INTEGER, value REAL);
            CREATE TABLE Vprev (state INTEGER, value REAL);''')
        cmd1 = "INSERT INTO V(state, value) VALUES(?, ?)"
        cmd2 = "INSERT INTO policy(state, action) VALUES(?, ?)"
        cmd3 = "INSERT INTO Vprev(state, value) VALUES(?, ?)"
        state = range(self.S)
        nones = [None] * self.S
        values = zip(state, nones)
        del nones
        self._cur.executemany(cmd2, values)
        self._cur.executemany(cmd3, values)
        del values
        if initial_V==0:
            V = [0] * self.S
            self._cur.executemany(cmd1, zip(state, V))
        else:
            try:
                self._cur.executemany(cmd1, zip(state, V))
            except:
                raise ValueError("V is of unsupported type, use a list or tuple.")
        self._conn.commit()
    
    def __del__(self):
        #self._cur.executescript('''
        #    DROP TABLE IF EXISTS Q;
        #    DROP TABLE IF EXISTS V;
        #    DROP TABLE IF EXISTS policy;''')
        self._cur.close()
        self._conn.close()
    
    def _bellmanOperator(self):
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
            self._cur.execute(cmd)
        self._conn.commit()
        self._calculateValue()
    
    def _calculatePolicy(self):
        """This implements argmax() over the actions of Q."""
        cmd = '''
              UPDATE policy
                 SET action = (
                     SELECT action
                       FROM (SELECT state, action, MAX(value)
                               FROM Q
                              GROUP BY state) AS A
                       WHERE policy.state = A.state
                       GROUP BY state);'''
        self._cur.execute(cmd)
        self._conn.commit()
    
    def _calculateValue(self):
        """This is max() over the actions of Q."""
        cmd = '''
              UPDATE V
                 SET value = (
                     SELECT MAX(value)
                       FROM Q
                      WHERE V.state = Q.state
                      GROUP BY state);'''
        self._cur.execute(cmd)
        self._conn.commit()
    
    def _getSpan(self):
        cmd = '''
              SELECT (MAX(A.value) - MIN(A.value))
              FROM (
                   SELECT (V.value - Vprev.value) as value
                     FROM V, Vprev
                    WHERE V.state = Vprev.state) AS A;'''
        self._cur.execute(cmd)
        span = self._cur.fetchone()
        if span is not None:
            return span[0]
    
    def getPolicyValue(self):
        """Get the policy and value vectors."""
        self._cur.execute("SELECT action FROM policy")
        r = self._cur.fetchall()
        policy = [x[0] for x in r]
        self._cur.execute("SELECT value FROM V")
        r = self._cur.fetchall()
        value = [x[0] for x in r]
        return policy, value
    
    def _randomQ(self):
        from numpy.random import random
        for a in range(self.A):
            state = range(self.S)
            action = [a] * self.S
            value = random(self.S).tolist()
            cmd = "INSERT INTO Q VALUES(?, ?, ?)"
            self._cur.executemany(cmd, zip(state, action, value))
        self._conn.commit()

class ValueIterationSQLite(MDPSQLite):
    """"""
    
    def __init__(self, db, discount, epsilon=0.01, max_iter=1000,
                 initial_value=0):
        MDPSQLite.__init__(self, db, discount, epsilon, max_iter, initial_value)
        
        if self.discount < 1:
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else:
            self.thresh = epsilon
        
        self._iterate()
    
    def _iterate(self):
        self.time = time()
        done = False
        while not done:
            self.itr += 1
            
            self._copyPreviousValue()
            self._bellmanOperator()
            variation = self._getSpan()
            
            if variation < self.thresh:
                done = True
            elif (self.itr == self.max_iter):
                done = True
        
        self._calculatePolicy()
    
    def _copyPreviousValue(self):
        cmd = '''
              UPDATE Vprev
                 SET value = (
                     SELECT value
                       FROM V
                      WHERE Vprev.state = V.state);'''
        self._cur.execute(cmd)
        self._conn.commit()
