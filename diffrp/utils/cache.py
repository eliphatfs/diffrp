import functools
from types import FunctionType
from typing import Hashable, TypeVar


class ICached:
    _cache: dict


FuncType = TypeVar('FuncType')


def cached_func(self: ICached, key: Hashable, func: FunctionType):
    if not hasattr(self, '_cache'):
        self._cache = {}
    if key not in self._cache:
        self._cache[key] = func()
    return self._cache[key]


def cached(func: FuncType) -> FuncType:

    @functools.wraps(func)
    def wrapped(self: ICached):
        return cached_func(self, func.__qualname__, lambda: func(self))

    return wrapped


def key_cached(func: FuncType) -> FuncType:

    @functools.wraps(func)
    def wrapped(self: ICached, arg: Hashable):
        return cached_func(self, arg, lambda: func(self, arg))

    return wrapped


def singleton_cached(func: FuncType) -> FuncType:

    @functools.wraps(func)
    def wrapped():
        return cached_func(func, func.__qualname__, lambda: func())

    return wrapped
