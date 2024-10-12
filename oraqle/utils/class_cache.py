import functools
from typing import List, Callable

CALL_DICT = {}


def cache(_func: Callable=None, *, default_val=None):
    """
    A caching decorator similar to that of cachetools. The main difference is that caching occurs using only the
    function name and object, and none of the other method parameters, as the key. This is useful for many
    circuit-traversal operations, as you usually only want to call methods once per node.
    
    Some methods may require a default value to be passed after caching, which is possible using the `defualt_val`
    parameter. Note that it must be assigned as a keyword parameter.
    
    Should only be applied to instance methods.
    """
    def fn_inner(fn):
        @functools.wraps(fn)
        def arg_inner(*args, **kwargs):
            global CALL_DICT
            
            method_name = fn.__name__
            instance = args[0]
            
            key = (method_name, instance)

            if key not in CALL_DICT:
                CALL_DICT[key] = fn(*args, **kwargs)
            elif default_val is not None:
                return default_val
            return CALL_DICT[key]
        return arg_inner

    if _func is None:
        return fn_inner
    return fn_inner(_func)


def clear_decorator_cache(methods_to_clear: List[Callable] = None):
    """
    Clear all cached values in the cache decorator.
    
    Can optionally take a list of methods for granular cache clearing.
    """
    if methods_to_clear is None:
        CALL_DICT.clear()
        return 
    for method in methods_to_clear:
        for key_method_name, key_instance in CALL_DICT.copy().keys():
            if method.__name__ == key_method_name:
                CALL_DICT.pop((key_method_name, key_instance))



# Tests

class TestClass:
    def __init__(self):
        self.p_default = 1
        self.p_basic = 1

    @cache(default_val=0)
    def m_with_default(self, inc):
        self.p_default += inc
        return self.p_default

    @cache
    def m_basic(self):
       self.p_basic += 1
       return self.p_basic
   

def test_simple_caching():
    test_class = TestClass()
    assert test_class.m_basic() == 2
    assert test_class.p_basic == 2
    assert test_class.m_basic() == 2
    assert test_class.p_basic == 2


def test_default_value_caching():
    test_class = TestClass()
    assert test_class.m_with_default(2) == 3
    assert test_class.p_default == 3
    assert test_class.m_with_default(3) == 0
    assert test_class.p_default == 3


def test_cache_clearing():
    test_class = TestClass()
    assert test_class.m_with_default(3) == 4
    assert test_class.m_basic() == 2
    assert test_class.p_default == 4
    assert test_class.p_basic == 2
    clear_decorator_cache()
    assert test_class.m_with_default(2) == 6
    assert test_class.m_basic() == 3
    assert test_class.p_default == 6
    assert test_class.p_basic == 3

def test_different_instances():
    test_class1 = TestClass()
    test_class2 = TestClass()

    assert test_class1.m_with_default(1) == 2
    assert test_class2.m_with_default(1) == 2
    assert test_class1.p_default == 2
    assert test_class2.p_default == 2
    assert test_class1.m_with_default(1) == 0
    assert test_class2.m_with_default(1) == 0
    assert test_class1.p_default == 2
    assert test_class2.p_default == 2
    
def test_clear_specific_method():
    test_class1 = TestClass()
    test_class2 = TestClass()

    assert test_class1.m_with_default(1) == 2
    assert test_class2.m_with_default(1) == 2
    assert test_class2.m_basic() == 2
    assert test_class1.p_default == 2
    assert test_class2.p_default == 2
    assert test_class2.p_basic == 2
    
    clear_decorator_cache([TestClass.m_with_default])

    assert test_class1.m_with_default(1) == 3
    assert test_class2.m_with_default(1) == 3
    assert test_class2.m_basic() == 2
    assert test_class1.p_default == 3
    assert test_class2.p_default == 3
    assert test_class2.p_basic == 2
    