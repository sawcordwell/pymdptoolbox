# -*- coding: utf-8 -*-

import sqlite3

class DatabaseManager(object):
    """"""
    
    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()
    
    def __del__(self):
        self.conn.close()
    
    def argmax(self, axis=None):
        pass
    
    def dot(self, a, b):
        pass
    
    def max(self, axis=None):
        pass
    