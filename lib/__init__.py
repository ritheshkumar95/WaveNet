import ops
import numpy
import theano
import theano.tensor as T
import cPickle as pickle
from collections import OrderedDict

_params = OrderedDict()

import locale

locale.setlocale(locale.LC_ALL, '')

def print_params_info(cost, params):
    """Print information about the parameters in the given param set."""

    params = sorted(params, key=lambda p: p.name)
    values = [p.get_value(borrow=True) for p in params]
    shapes = [p.shape for p in values]
    print "Params for cost:"
    for param, value, shape in zip(params, values, shapes):
        print "\t{0} ({1})".format(
            param.name,
            ",".join([str(x) for x in shape])
        )

    total_param_count = 0
    for shape in shapes:
        param_count = 1
        for dim in shape:
            param_count *= dim
        total_param_count += param_count
    print "Total parameter count: {0}".format(
        locale.format("%d", total_param_count, grouping=True)
    )

def param(name, *args, **kwargs):
    """
    A wrapper for `theano.shared` which enables parameter sharing in models.

    Creates and returns theano shared variables similarly to `theano.shared`,
    except if you try to create a param with the same name as a
    previously-created one, `param(...)` will just return the old one instead of
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it
    creates, so that you can easily search a graph for all params.
    """
    if name not in _params:
        kwargs['name'] = name
        train = not 'train' in kwargs
        if not train:
            del kwargs['train']
        param = theano.shared(*args, **kwargs)
        if train:
            param.param = train
        _params[name] = param
    return _params[name]

def delete_params(name):
    to_delete = [p_name for p_name in _params if name in p_name]
    for p_name in to_delete:
        del _params[p_name]

def search(node, critereon):
    """
    Traverse the Theano graph starting at `node` and return a list of all nodes
    which match the `critereon` function. When optimizing a cost function, you
    can use this to get a list of all of the trainable params in the graph, like
    so:

    `lib.search(cost, lambda x: hasattr(x, "param"))`
    """

    def _search(node, critereon, visited):
        if node in visited:
            return []
        visited.add(node)

        results = []
        if isinstance(node, T.Apply):
            for inp in node.inputs:
                results += _search(inp, critereon, visited)
        else: # Variable node
            if critereon(node):
                results.append(node)
            if node.owner is not None:
                results += _search(node.owner, critereon, visited)
        return results

    return _search(node, critereon, set())

def floatX(x):
    """
    Convert `x` to the numpy type specified in `theano.config.floatX`.
    """
    return numpy.float32(x)

def save_params(path):
    param_vals = {}
    for name, param in _params.iteritems():
        param_vals[name] = param.get_value()

    with open(path, 'wb') as f:
        pickle.dump(param_vals, f)

def load_params(path):
    with open(path, 'rb') as f:
        param_vals = pickle.load(f)

    for name, val in param_vals.iteritems():
        _params[name].set_value(val)

def clear_all_params():
    to_delete = [p_name for p_name in _params]
    for p_name in to_delete:
        del _params[p_name]
