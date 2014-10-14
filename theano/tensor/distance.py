import logging

logger = logging.getLogger(__name__)
import numpy

from theano.gof import Op, Apply

from theano.tensor import as_tensor_variable, dot, DimShuffle, Dot
from theano.tensor.blas import Dot22
from theano import tensor
import theano.tensor
from theano.tensor.opt import (register_stabilize,
        register_specialize, register_canonicalize)
from theano.gof import local_optimizer
from theano.gof.opt import Optimizer
from theano.gradient import DisconnectedType
from theano import printing

try:
    import scipy.linalg
    from scipy.spatial.distance import cdist
    imported_scipy = True
except ImportError:
    # some ops (e.g. Cholesky, Solve, A_Xinv_b) won't work
    imported_scipy = False

MATRIX_STRUCTURES = (
        'general',
        'symmetric',
        'lower_triangular',
        'upper_triangular',
        'hermitian',
        'banded',
        'diagonal',
        'toeplitz',
        )

class SqEuclidean(Op):
    """
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        return (type(self) == type(other) and self.sym_pos == other.sym_pos and
                self.lower == other.lower and
                self.overwrite_a == other.overwrite_a and
                self.overwrite_b == other.overwite_b)

    def __hash__(self):
        return (hash(type(self)))

    def props(self):
        return (self.sym_pos, self.lower, self.overwrite_a, self.overwrite_b)

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def __repr__(self):
        return 'Solve{%s}' % str(self.props())

    def make_node(self, X1, X2):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Solve op")
        #if X1 == X2:
            #raise TypeError("Cannot do CDist on same X - use PDist")
        X1 = tensor.as_tensor_variable(X1)
        X2 = tensor.as_tensor_variable(X2)
        if X1.ndim != X2.ndim:
            raise TypeError('%s: inputs must have same number of dimensions\n' \
                             % self.__class__.__name__)
        if X1.ndim > 2:
            raise TypeError("%s: Inputs must be 1 or 2 dimensional\n" \
                            % self.__class__.__name__)

        out_type = tensor.TensorType(dtype=(X1 * X2).dtype,
                                     broadcastable=X1.type.broadcastable)()
        return Apply(self, [X1, X2], [out_type])

    def infer_shape(self, node, in_shapes):
        N, _ = in_shapes[0]
        M, _ = in_shapes[1]
        return [(N, M)]

    def perform(self, node, inputs, output_storage):
        X1, X2 = inputs
        
        if (len(X1.shape) > 1) and (X1.shape[1] != X2.shape[1]):
            raise TypeError("%s: Inputs must have same trailing dimensions\n" \
                            % self.__class__.__name__)
        output_storage[0][0] = cdist(X1, X2, 'sqeuclidean')

    def grad(self, inputs, cost_grad):
        """
        """

        X1, X2 = inputs
        jac = tensor.as_tensor_variable(cost_grad)
        X1 = tensor.as_tensor_variable(X1)
        X2 = tensor.as_tensor_varialbe(X2)
        N, D = X1.shape
        M, _ = X2.shape
        jacA = jac.reshape([N, -1], ndim=2)[:, None, :].T
        jacB = jac.T.reshape([N, -1], ndim=2)[:, None, :].T
        euclideanA = 2*(X1.T[None, :, :] - X2[:, :, None])
        euclideanB = euclideanA.T
        outgradA = (euclideanA * jacA).sum(0)
        outgradB = (euclideanB * jacB).sum(0)

        return [outgradA.T, outgradB.T]

sqeuclidean = SqEuclidean()


class SqEuclideanSelf(Op):
    """
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        return (type(self) == type(other) and self.sym_pos == other.sym_pos and
                self.lower == other.lower and
                self.overwrite_a == other.overwrite_a and
                self.overwrite_b == other.overwite_b)

    def __hash__(self):
        return (hash(type(self)))

    def props(self):
        return (self.sym_pos, self.lower, self.overwrite_a, self.overwrite_b)

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def __repr__(self):
        return 'Solve{%s}' % str(self.props())

    def make_node(self, X):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Solve op")
        #if X1 == X2:
            #raise TypeError("Cannot do CDist on same X - use PDist")
        X = tensor.as_tensor_variable(X)
        if X.ndim > 2:
            raise TypeError("%s: Inputs must be 1 or 2 dimensional\n" \
                            % self.__class__.__name__)

        out_type = tensor.TensorType(dtype=(X).dtype,
                                     broadcastable=X.type.broadcastable)()
        return Apply(self, [X], [out_type])

    def infer_shape(self, node, in_shapes):
        N, _ = in_shapes[0]
        return [(N, N)]

    def perform(self, node, inputs, output_storage):
        X, = inputs
        
        output_storage[0][0] = cdist(X, X, 'sqeuclidean')

    def grad(self, inputs, cost_grad):
        """
        """

        X,  = inputs
        X = tensor.as_tensor_variable(X)
        jac = tensor.as_tensor_variable(cost_grad)
        N, D = X.shape
        jacA = jac.reshape([N, -1], ndim=2)[:, None, :].T
        jacB = jac.T.reshape([N, -1], ndim=2)[:, None, :].T
        euclideanA = 2*(X.T[None, :, :] - X[:, :, None])
        euclideanB = euclideanA.T
        outgradA = (euclideanA * jacA).sum(0)
        outgradB = (euclideanB * jacB).sum(0)

        return [(outgradA - outgradB).T]

sqeuclidean_self = SqEuclideanSelf()
