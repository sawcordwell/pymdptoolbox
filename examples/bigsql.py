# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 07:18:32 2013

@author: steve
"""

import os
import sqlite3

from numpy import arange
from numpy.random import permutation, random, randint
from scipy.sparse import dok_matrix

from mdp import ValueIteration
from mdpsql import ValueIteration as ValueIterationSQL

def setup(S, A):
    P_sparse = [None] * A
    R_sparse = 2*random(S) - 1
    DB_sql = "MDP-big.db" % (S, A)
    if os.path.exists(DB_sql):
        os.remove(DB_sql)
    with sqlite3.connect(DB_sql) as conn:
        c = conn.cursor()
        cmd = '''
            CREATE TABLE info (name TEXT, value INTEGER);
            INSERT INTO info VALUES('states', %s);
            INSERT INTO info VALUES('actions', %s);''' % (S, A)
        c.executescript(cmd)
        for a in range(1, A+1):
            a_sparse = a - 1
            PP_sparse = dok_matrix((S, S))
            cmd = '''
                CREATE TABLE transition%s (row INTEGER, col INTEGER, prob REAL);
                CREATE TABLE reward%s (state INTEGER PRIMARY KEY ASC, val REAL);
                ''' % (a, a)
            c.executescript(cmd)
            cmd = "INSERT INTO reward%s(val) VALUES(?)" % a
            c.executemany(cmd, zip(R_sparse.tolist()))
            for s in xrange(1, S+1):
                s_sparse = s - 1
                n = randint(1, 10)
                col = (permutation(arange(1,S+1))[0:n]).tolist()
                val = random(n)
                val = (val / val.sum()).tolist()
                PP_sparse[s_sparse, col - 1] = val
                cmd = "INSERT INTO transition%s VALUES(?, ?, ?)" % a
                c.executemany(cmd, zip([s] * n, col, val))
            cmd = "CREATE UNIQUE INDEX Pidx%s ON transition%s (row, col);" % (a, a)
            c.execute(cmd)
        P_sparse[a_sparse] = PP_sparse.tocsr()   
    return P_sparse, R_sparse, DB_sql

if __name__ == "__main__":
    P_sparse, R_sparse, DB_sql = setup(100000000, 3)
    try:
        sdp = ValueIteration(P_sparse, R_sparse, 0.9)
    except MemoryError:
        print("Killed. Sparse method ran out of memory.")
    except:
        raise
    try:
        sdp = ValueIterationSQL(DB_sql, 0.9)
    except MemoryError:
        print("Killed. SQL method ran out of memory.")
    except:
        raise
