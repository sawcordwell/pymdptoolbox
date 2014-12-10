#!/usr/bin/env sh

# Is there any difference to using ``python setup.py nosetests``?
nosetests --with-coverage --cover-package=mdptoolbox --with-doctest \
    --doctest-options='+ELLIPSIS,+NORMALIZE_WHITESPACE,+IGNORE_EXCEPTION_DETAIL'
