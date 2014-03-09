# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:20:30 2014

@author: Steven A W Cordwell
"""

import mdptoolbox as mdp

def getHabitatSuitability(time):
    """The habitat quality of a patch according to the time since last fire.
    
    The habitat quality is low immediately after a fire, rises rapidly until
    five years after a fire, and declines once the habitat is mature. See
    Figure 2 in Possingham and Tuck (1997) for more details.
    
    Parameters
    ----------
    time : int
        Time in years since last fire.
    
    Returns
    -------
    r : float
        The habitat suitability.
    
    """
    assert time >= 0, "'time' must be a positive number"
    if time <= 5:
        return(0.2 * time)
    elif 5 <= time <= 10:
        return(-0.1 * time + 1.5)
    else:
        return(0.5)

def tests():
    assert getHabitatSuitability(0) == 0
    assert getHabitatSuitability(2) == 0.4
    assert getHabitatSuitability(5) == 1
    assert getHabitatSuitability(7) == 0.8
    assert getHabitatSuitability(10) == 0.5

if __name__ == "__main__":
    tests()
