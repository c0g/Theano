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
        return (hash(type(self)) ^ hash(self.sym_pos) ^ hash(self.lower) ^
                hash(self.overwrite_a) ^ hash(self.overwrite_b))

    def props(self):
        return (self.sym_pos, self.lower, self.overwrite_a, self.overwrite_b)

    def __str__(self):
        return "%s{%s, %s, %s, %s}" % (self.__class__.__name__,
                "sym_pos=".join(str(self.sym_pos)),
                "lower=".join(str(self.lower)),
                "overwrite_a".join(str(self.overwrite_a)),
                "overwrite_b=".join(str(self.overwrite_b)))

    def __repr__(self):
        return 'Solve{%s}' % str(self.props())

    def make_node(self, x1, x2):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Solve op")
        x1 = tensor.as_tensor_variable(x1)
        x2 = tensor.as_tensor_variable(x2)
        if x1.ndim != x2.ndim:
            raise TypeError('%s: inputs must have same number of dimensions\n' \
                             % self.__class__.__name__)
        if x1.ndim > 2:
            raise TypeError("%s: Inputs must be 1 or 2 dimensional\n" \
                            % self.__class__.__name__)

        out_type = tensor.TensorType(dtype=(x1 * x2).dtype,
                                     broadcastable=b.type.broadcastable)()
        return Apply(self, [x1, x2], [out_type])

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def perform(self, node, inputs, output_storage):
        x1, x2 = inputs

        if x1.shape[1] != x2.shape[2]:
            raise TypeError("%s: Inputs must have same trailing dimensions\n" \
                            % self.__class__.__name__)
        output_storage[0][0] = cdist(x1, x2, 'sqeuclidean')

    def grad(self, inputs, cost_grad):
        """
        inputs:
            X1 is NxD
            X2 is MxD
            cost_grad is (NxM)x1 (?) might be NxM matrix

        outputs:
            outgrad_a is (NxD)? X 1
            outgrad_b is (MxD)? X 1
        """

        x1, x2 = inputs
        dist = cdist(x1, x2, 'sqeuclidean')


        return [outgrad_a, outgrad_b]


def solve(a, b, sym_pos=False, lower=False, overwrite_a=False,
                 overwrite_b=False):
    localop = Solve(sym_pos, lower, overwrite_a, overwrite_b)
    return localop(a, b)
