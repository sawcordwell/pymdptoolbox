# -*- coding: utf-8 -*-

import os
import sqlite3

from time import time

from numpy import arange
from numpy.random import permutation, random, randint

def exampleForest(S=3, r1=4, r2=2, p=0.1):
    db = "MDP-forest-%s.db" % S
    if os.path.exists(db):
        os.remove(db)
    conn = sqlite3.connect(db)
    with conn:
        c = conn.cursor()
        cmd = '''
            CREATE TABLE info (name TEXT, value INTEGER);
            INSERT INTO info VALUES('states', %s);
            INSERT INTO info VALUES('actions', 2);''' % S
        c.executescript(cmd)
        cmd = '''
            CREATE TABLE transition1 (row INTEGER, col INTEGER, prob REAL);
            CREATE TABLE reward1 (state INTEGER PRIMARY KEY ASC, val REAL);
            CREATE TABLE transition2 (row INTEGER, col INTEGER, prob REAL);
            CREATE TABLE reward2 (state INTEGER PRIMARY KEY ASC, val REAL);'''
        c.executescript(cmd)
        rows = range(1, S + 1) * 2
        cols = [1] * S + range(2, S + 1) + [S]
        vals = [p] * S + [1-p] * S
        cmd = "INSERT INTO transition1 VALUES(?, ?, ?)"
        c.executemany(cmd, zip(rows, cols, vals))
        rows = range(1, S + 1)
        cols = [1] * S
        vals = [1] * S
        cmd = "INSERT INTO transition2 VALUES(?, ?, ?)"
        c.executemany(cmd, zip(rows, cols, vals))
        cmd = "INSERT INTO reward1(val) VALUES(?)"
        c.executemany(cmd, zip([0] * (S - 1) + [r1]))
        cmd = "INSERT INTO reward2(val) VALUES(?)"
        c.executemany(cmd, zip([0] + [1] * (S - 2) + [r2]))
        cmd = '''
            CREATE INDEX Pidx1 ON transition1 (row, col);
            CREATE INDEX Pidx2 ON transition2 (row, col);'''
        c.executescript(cmd)
    # return the databases name
    return db

def exampleRand(S, A):
    """WARNING: This will delete a database with the same name as 'db'."""
    db = "MDP-%sx%s.db" % (S, A)
    if os.path.exists(db):
        os.remove(db)
    conn = sqlite3.connect(db)
    with conn:
        c = conn.cursor()
        cmd = '''
            CREATE TABLE info (name TEXT, value INTEGER);
            INSERT INTO info VALUES('states', %s);
            INSERT INTO info VALUES('actions', %s);''' % (S, A)
        c.executescript(cmd)
        for a in range(1, A+1):
            cmd = '''
                CREATE TABLE transition%s (row INTEGER, col INTEGER, prob REAL);
                CREATE TABLE reward%s (state INTEGER PRIMARY KEY ASC, val REAL);
                ''' % (a, a)
            c.executescript(cmd)
            cmd = "INSERT INTO reward%s(val) VALUES(?)" % a
            c.executemany(cmd, zip(random(S).tolist()))
            for s in xrange(1, S+1):
                # to be usefully represented as a sparse matrix, the number of
                # nonzero entries should be less than 1/3 of dimesion of the
                # matrix, so S/3
                n = randint(1, S//3)
                # timeit [90894] * 20330
                # ==> 10000 loops, best of 3: 141 us per loop
                # timeit (90894*np.ones(20330, dtype=int)).tolist()
                # ==> 1000 loops, best of 3: 548 us per loop
                col = (permutation(arange(1,S+1))[0:n]).tolist()
                val = random(n)
                val = (val / val.sum()).tolist()
                cmd = "INSERT INTO transition%s VALUES(?, ?, ?)" % a
                c.executemany(cmd, zip([s] * n, col, val))
            cmd = "CREATE UNIQUE INDEX Pidx%s ON transition%s (row, col);" % (a, a)
            c.execute(cmd)
    # return the name of teh database
    return db


class MDP(object):
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
            raise ValueError("Cannot determine number of states from "
                             "database. There is no name 'states' in table "
                             "'info'.")
        self._cur.execute("SELECT value FROM info WHERE name='actions'")
        try:
            self.A = self._cur.fetchone()[0]
        except TypeError:
            raise ValueError("Cannot determine number of actions from "
                             "database. There is no name 'actions' in table "
                             "'info'.")
        self._checkSquareStochastic()
        self._initQ()
        self._initResults(initial_V)
    
    def _checkSquareStochastic(self):
        # check that the columns of the transition matrices sum to one
        for a in range(1, self.A + 1):
            P = "transition%s" % a
            cmd = "SELECT SUM(s) " \
                  "  FROM (" \
                  "       SELECT ABS(SUM(prob)-1)<10e-12 AS s" \
                  "         FROM "+P+"" \
                  "        GROUP BY row);"
            self._cur.execute(cmd)
            try:
                if self._cur.fetchone()[0] != self.S:
                    raise ValueError("The transition matrix for action %s "
                                     "is not stochastic." % a)
            except TypeError:
                raise StandardError("The check stochastic query for a=%s "
                                    "failed." % a)
            cmd = "SELECT MAX(row) FROM " + P
            self._cur.execute(cmd)
            row_max = self._cur.fetchone()[0]
            if int(row_max) != self.S:
                raise ValueError("The transition matrix for action %s is "
                                 "not square: row_max = %s" % (a, row_max))
            cmd = "SELECT MAX(col) FROM " + P
            self._cur.execute(cmd)
            col_max = self._cur.fetchone()[0]
            if int(col_max) > row_max:
                raise ValueError("The transition matrix for action %a id "
                                 "not square: col_max = %s" % (a, col_max))
    
    def _initQ(self):
        self._delQ()
        self._cur.execute("CREATE TABLE Q (state INTEGER, action INTEGER, "
                          "value REAL);")
        for a in range(1, self.A + 1):
            state = xrange(1, self.S + 1)
            action = [a] * self.S
            value = [None] * self.S
            cmd = "INSERT INTO Q VALUES(?, ?, ?)"
            self._cur.executemany(cmd, zip(state, action, value))
        self._cur.execute("CREATE UNIQUE INDEX Qidx ON Q (state, action);")
        self._conn.commit()
    
    def _delQ(self):
        self._cur.executescript('''
            DROP TABLE IF EXISTS Q;
            DROP INDEX IF EXISTS Qidx;''')
    
    def _initResults(self, initial_V):
        self._delResults()
        self._cur.executescript('''
            CREATE TABLE policy (state INTEGER PRIMARY KEY ASC, action INTEGER);
            CREATE TABLE V (state INTEGER PRIMARY KEY ASC, value REAL);
            CREATE TABLE Vprev (state INTEGER PRIMARY KEY ASC, value REAL);''')
        cmd1 = "INSERT INTO V(value) VALUES(?)"
        cmd2 = "INSERT INTO policy(action) VALUES(?)"
        cmd3 = "INSERT INTO Vprev(value) VALUES(?)"
        values = zip([None] * self.S)
        self._cur.executemany(cmd2, values)
        self._cur.executemany(cmd3, values)
        del values
        if initial_V==0:
            self._cur.executemany(cmd1, zip([0] * self.S))
        else:
            try:
                self._cur.executemany(cmd1, zip(initial_V))
            except:
                raise ValueError("V is of unsupported type, use a list or "
                                 "tuple.")
        self._conn.commit()
    
    def _delResults(self):
        self._cur.executescript('''
            DROP TABLE IF EXISTS policy;
            DROP TABLE IF EXISTS V;
            DROP TABLE IF EXISTS Vprev;''')
    
    def __del__(self):
        self._delQ()
        self._cur.executescript('''
            DROP TABLE IF EXISTS Vprev;
            VACUUM;''')
        self._cur.close()
        self._conn.close()
    
    def _bellmanOperator(self):
        g = str(self.discount)
        for a in range(1, self.A + 1):
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
        for a in range(1,self.A+1):
            state = xrange(1,self.S+1)
            action = [a] * self.S
            value = random(self.S).tolist()
            cmd = "INSERT INTO Q VALUES(?, ?, ?)"
            self._cur.executemany(cmd, zip(state, action, value))
        self._conn.commit()

class ValueIteration(MDP):
    """"""
    
    def __init__(self, db, discount, epsilon=0.01, max_iter=1000,
                 initial_value=0):
        MDP.__init__(self, db, discount, epsilon, max_iter, initial_value)
        
        if self.discount < 1:
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else:
            self.thresh = epsilon
        
        self._iterate()
    
    def __del__(self):
        MDP.__del__(self)
    
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
        # get the optimal policy
        self._calculatePolicy()
        # calculate the time taken to finish
        self.time = time() - self.time
    
    def _copyPreviousValue(self):
        cmd = '''
              UPDATE Vprev
                 SET value = (
                     SELECT value
                       FROM V
                      WHERE Vprev.state = V.state);'''
        self._cur.execute(cmd)
        self._conn.commit()
