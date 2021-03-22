# -*- coding: utf-8 -*-
from collections import OrderedDict
from functools import wraps
from inspect import iscode


def wrap_decorator(decorator):
    @wraps(decorator)
    def new_decorator(*args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return decorator(args[0])

        def decorator_wrapper(f):
            return decorator(f, *args, **kwargs)

        return decorator_wrapper

    return new_decorator


class CodeObj:
    """Return a (somewhat) flattened, hashable version of func.__code__.

    For a function `func`, use like so:
        code_obj = CodeObj(func.__code__).hashable()

    Note that closure variables are not supported.

    """

    expansion_limit = 1000

    def __init__(self, code, __expansion_count=0):
        assert iscode(code), "Must pass in a code object (function.__code__)."
        self.code = code
        self.__expansion_count = __expansion_count

    def hashable(self):
        if self.__expansion_count > self.expansion_limit:
            raise RuntimeError(
                "Maximum number of code object expansions exceeded ({} > {}).".format(
                    self.__expansion_count, self.expansion_limit
                )
            )

        # Get co_ attributes that describe the code object. Ignore the line number and
        # file name of the function definition here, since we don't want unrelated
        # changes to cause a recalculation of a cached result. Changes in comments are
        # ignored, but changes in the docstring will still causes comparisons to fail
        # (this could be ignored as well, however)!
        self.code_dict = OrderedDict(
            (attr, getattr(self.code, attr))
            for attr in dir(self.code)
            if "co_" in attr
            and "co_firstlineno" not in attr
            and "co_filename" not in attr
        )
        # Replace any nested code object (eg. for list comprehensions) with a reduced
        # version by calling the hashable function recursively.
        new_co_consts = []
        for value in self.code_dict["co_consts"]:
            if iscode(value):
                self.__expansion_count += 1
                value = type(self)(value, self.__expansion_count).hashable()
            new_co_consts.append(value)

        self.code_dict["co_consts"] = tuple(new_co_consts)

        return tuple(self.code_dict.values())
