import functools
from typing import Hashable
from types import FunctionType


class ICached:
    _cache: dict


def cached_func(self: ICached, key: Hashable, func: FunctionType):
    try:
        self._cache
    except AttributeError:
        self._cache = {}
    if key not in self._cache:
        self._cache[key] = func()
    return self._cache[key]


@staticmethod
def cached(func: FunctionType):

    @functools.wraps(func)
    def wrapped(self: ICached):
        return self.cached_func(func.__qualname__, lambda: func(self))

    return wrapped


@staticmethod
def key_cached(func: FunctionType):

    @functools.wraps(func)
    def wrapped(self: ICached, arg: Hashable):
        return self.cached_func(arg, lambda: func(self, arg))

    return wrapped
