# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.10
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_util')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_util')
    _util = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_util', [dirname(__file__)])
        except ImportError:
            import _util
            return _util
        if fp is not None:
            try:
                _mod = imp.load_module('_util', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _util = swig_import_helper()
    del swig_import_helper
else:
    import _util
del _swig_python_version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        object.__setattr__(self, name, value)
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_method(set):
    def set_attr(self, name, value):
        if (name == "thisown"):
            return self.this.own(value)
        if hasattr(self, name) or (name == "this"):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add attributes to %s" % self)
    return set_attr



def disjunction(source, dest):
    """

    `disjunction(const std::vector< bool > &source, std::vector< bool > &dest)`  

    """
    return _util.disjunction(source, dest)

def any_true(b_vector, indices):
    """

    `any_true(const std::vector< bool > &b_vector, const std::vector< unsigned int >
        indices) -> bool`  

    """
    return _util.any_true(b_vector, indices)
class run_stats(object):
    """


    simple class to compute mean and variance sequentially one value at a time  

    C++ includes: util.hpp

    """

    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr

    def push(self, x):
        """

        `push(num_t x)`  

        adds a value to the statistic  

        Parameters
        ----------
        * `x` :  
            the value to add  

        """
        return _util.run_stats_push(self, x)


    def pop(self, x):
        """

        `pop(num_t x)`  

        removes a value from the statistic  

        Consider this the inverse operation to push. Note: you can create a scenario
        where the variance would be negative, so a simple sanity check is implemented
        that raises a RuntimeError if that happens.  

        Parameters
        ----------
        * `x` :  
            the value to remove  

        """
        return _util.run_stats_pop(self, x)


    def divide_sdm_by(self, value):
        """

        `divide_sdm_by(num_t value) const -> num_t`  

        divides the (summed) squared distance from the mean by the argument  

        Parameters
        ----------
        * `value` :  
            to divide by  

        Returns
        -------
        sum([ (x - mean())**2 for x in values])/ value  

        """
        return _util.run_stats_divide_sdm_by(self, value)


    def number_of_points(self):
        """

        `number_of_points() const -> long unsigned int`  

        returns the number of points  

        Returns
        -------
        the current number of points added  

        """
        return _util.run_stats_number_of_points(self)


    def mean(self):
        """

        `mean() const -> num_t`  

        the mean of all values added  

        Returns
        -------
        sum([x for x in values])/number_of_points()  

        """
        return _util.run_stats_mean(self)


    def sum(self):
        """

        `sum() const -> num_t`  

        the sum of all values added  

        Returns
        -------
        the sum of all values (equivalent to number_of_points()* mean())  

        """
        return _util.run_stats_sum(self)


    def sum_of_squares(self):
        """

        `sum_of_squares() const -> num_t`  

        the sum of all values squared  

        Returns
        -------
        sum([x**2 for x in values])  

        """
        return _util.run_stats_sum_of_squares(self)


    def variance_population(self):
        """

        `variance_population() const -> num_t`  

        the variance of all samples assuming it is the total population  

        Returns
        -------
        sum([(x-mean())**2 for x in values])/number_of_points  

        """
        return _util.run_stats_variance_population(self)


    def variance_sample(self):
        """

        `variance_sample() const -> num_t`  

        unbiased variance of all samples assuming it is a sample from a population with
        unknown mean  

        Returns
        -------
        sum([(x-mean())**2 for x in values])/(number_of_points-1)  

        """
        return _util.run_stats_variance_sample(self)


    def variance_MSE(self):
        """

        `variance_MSE() const -> num_t`  

        biased estimate variance of all samples with the smalles MSE  

        Returns
        -------
        sum([(x-mean())**2 for x in values])/(number_of_points+1)  

        """
        return _util.run_stats_variance_MSE(self)


    def std_population(self):
        """

        `std_population() const -> num_t`  

        standard deviation based on variance_population  

        Returns
        -------
        sqrt(variance_population())  

        """
        return _util.run_stats_std_population(self)


    def std_sample(self):
        """

        `std_sample() const -> num_t`  

        (biased) estimate of the standard deviation based on variance_sample  

        Returns
        -------
        sqrt(variance_sample())  

        """
        return _util.run_stats_std_sample(self)


    def std_unbiased_gaussian(self):
        """

        `std_unbiased_gaussian() const -> num_t`  

        unbiased standard deviation for normally distributed values  

        Source: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation  

        Returns
        -------
        std_sample/correction_value  

        """
        return _util.run_stats_std_unbiased_gaussian(self)


    def __iadd__(self, other):
        return _util.run_stats___iadd__(self, other)

    def __mul__(self, a):
        return _util.run_stats___mul__(self, a)

    def __add__(self, *args):
        return _util.run_stats___add__(self, *args)

    def __sub__(self, *args):
        return _util.run_stats___sub__(self, *args)

    def __isub__(self, other):
        return _util.run_stats___isub__(self, other)

    def numerically_equal(self, other, rel_error):
        """

        `numerically_equal(const running_statistics other, num_t rel_error) -> bool`  

        method to check for numerical equivalency  

        Parameters
        ----------
        * `other` :  
            the other running statistic to compare against  
        * `rel_error` :  
            relative tolerance for the mean and variance  

        """
        return _util.run_stats_numerically_equal(self, other, rel_error)

    __swig_destroy__ = _util.delete_run_stats
    __del__ = lambda self: None
run_stats_swigregister = _util.run_stats_swigregister
run_stats_swigregister(run_stats)

def running_statistics(*args):
    """

    `running_statistics(long unsigned int n, num_t a, num_t s)`  

    """
    val = _util.new_running_statistics(*args)
    return val

class weighted_running_stats(object):
    """


    simple class to compute weighted mean and variance sequentially one value at a
    time  

    C++ includes: util.hpp

    """

    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """

        `weighted_running_statistics(num_t m, num_t s, running_statistics< num_t >
            w_stat)`  

        """
        this = _util.new_weighted_running_stats(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push(self, x, weight):
        """

        `push(num_t x, num_t weight)`  

        """
        return _util.weighted_running_stats_push(self, x, weight)


    def pop(self, x, weight):
        """

        `pop(num_t x, num_t weight)`  

        """
        return _util.weighted_running_stats_pop(self, x, weight)


    def number_of_points(self):
        """

        `number_of_points() const -> long unsigned int`  

        returns the number of points  

        Returns
        -------
        the current number of points added  

        """
        return _util.weighted_running_stats_number_of_points(self)


    def squared_deviations_from_the_mean(self):
        """

        `squared_deviations_from_the_mean() const -> num_t`  

        """
        return _util.weighted_running_stats_squared_deviations_from_the_mean(self)


    def divide_sdm_by(self, fraction, min_weight):
        """

        `divide_sdm_by(num_t fraction, num_t min_weight) const -> num_t`  

        """
        return _util.weighted_running_stats_divide_sdm_by(self, fraction, min_weight)


    def mean(self):
        """

        `mean() const -> num_t`  

        """
        return _util.weighted_running_stats_mean(self)


    def sum_of_weights(self):
        """

        `sum_of_weights() const -> num_t`  

        """
        return _util.weighted_running_stats_sum_of_weights(self)


    def sum_of_squares(self):
        """

        `sum_of_squares() const -> num_t`  

        """
        return _util.weighted_running_stats_sum_of_squares(self)


    def variance_population(self):
        """

        `variance_population() const -> num_t`  

        """
        return _util.weighted_running_stats_variance_population(self)


    def variance_unbiased_frequency(self):
        """

        `variance_unbiased_frequency() const -> num_t`  

        """
        return _util.weighted_running_stats_variance_unbiased_frequency(self)


    def variance_unbiased_importance(self):
        """

        `variance_unbiased_importance() const -> num_t`  

        """
        return _util.weighted_running_stats_variance_unbiased_importance(self)


    def __iadd__(self, other):
        return _util.weighted_running_stats___iadd__(self, other)

    def __sub__(self, other):
        return _util.weighted_running_stats___sub__(self, other)

    def __isub__(self, other):
        return _util.weighted_running_stats___isub__(self, other)

    def __mul__(self, a):
        return _util.weighted_running_stats___mul__(self, a)

    def __add__(self, *args):
        return _util.weighted_running_stats___add__(self, *args)

    def multiply_weights_by(self, a):
        """

        `multiply_weights_by(const num_t a) const -> weighted_running_statistics`  

        """
        return _util.weighted_running_stats_multiply_weights_by(self, a)


    def numerically_equal(self, other, rel_error):
        """

        `numerically_equal(weighted_running_statistics other, num_t rel_error) -> bool`  

        """
        return _util.weighted_running_stats_numerically_equal(self, other, rel_error)


    def get_weight_statistics(self):
        """

        `get_weight_statistics() const -> running_statistics< num_t >`  

        """
        return _util.weighted_running_stats_get_weight_statistics(self)

    __swig_destroy__ = _util.delete_weighted_running_stats
    __del__ = lambda self: None
weighted_running_stats_swigregister = _util.weighted_running_stats_swigregister
weighted_running_stats_swigregister(weighted_running_stats)


