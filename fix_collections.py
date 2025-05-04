#!/usr/bin/env python
"""
Monkey patch to fix collections imports in Python 3.11+
"""

# Apply patch before importing pymongo
import collections
import collections.abc

# Apply the patches for all common ABC classes
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable
collections.Sequence = collections.abc.Sequence