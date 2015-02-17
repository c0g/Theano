import logging

logger = logging.getLogger(__name__)
import numpy

from theano.gof import Op, Apply
from theano import tensor
from theano.tensor import as_tensor_variable, dot, DimShuffle, Dot
from theano.tensor.blas import Dot22
import theano.tensor
from theano.tensor.opt import (register_stabilize,
        register_specialize, register_canonicalize)
from theano.gof import local_optimizer
from theano.gof.opt import Optimizer
from theano.gradient import DisconnectedType


try:
    import scipy.linalg
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

class CholLogDet(Op):
    """Matrix determinant
    Input should be a square matrix, PSD
    """
    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, (x,), (z, )):
        try:
            z[0] = numpy.asarray(2*numpy.log(numpy.diag(numpy.linalg.cholesky(x))).sum(), dtype=x.dtype)
        except Exception:
            print 'Failed to compute determinant', x
            raise

    def grad(self, inputs, g_outputs):
        X, = inputs
        gz, = g_outputs
        N, _ = X.shape
        return [solve(X.T, theano.tensor.eye(N), sym_pos=True) * gz ]

    def infer_shape(self, node, shapes):
        return [()]

    def __str__(self):
        return "Det"
chollogdet = CholLogDet()

class Cholesky(Op):
    """
    Return a triangular matrix square root of positive semi-definite `x`

    L = cholesky(X, lower=True) implies dot(L, L.T) == X
    """
    #TODO: inplace
    #TODO: for specific dtypes
    #TODO: LAPACK wrapper with in-place behavior, for solve also

    __props__ = ('lower', 'destructive')

    def __init__(self, lower=True):
        self.lower = lower
        self.destructive = False

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def make_node(self, x):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Cholesky op")
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        z[0] = scipy.linalg.cholesky(x, lower=self.lower).astype(x.dtype)

    def grad(self, inputs, gradients):
        return [CholeskyGrad(self.lower)(inputs[0], self(inputs[0]),
                                         gradients[0])]

cholesky = Cholesky()


class CholeskyGrad(Op):
    """
    """

    __props__ = ('lower', 'destructive')

    def __init__(self, lower=True):
        self.lower = lower
        self.destructive = False

    def make_node(self, x, l, dz):
        x = as_tensor_variable(x)
        l = as_tensor_variable(l)
        dz = as_tensor_variable(dz)
        assert x.ndim == 2
        assert l.ndim == 2
        assert dz.ndim == 2
        assert l.owner.op.lower == self.lower, (
            "lower/upper mismatch between Cholesky op and CholeskyGrad op"
        )
        return Apply(self, [x, l, dz], [x.type()])

    def perform(self, node, inputs, outputs):
        """Implements the "reverse-mode" gradient [1]_ for the
        Cholesky factorization of a positive-definite matrix.

        .. [1] S. P. Smith. "Differentiation of the Cholesky Algorithm".
               Journal of Computational and Graphical Statistics,
               Vol. 4, No. 2 (Jun.,1995), pp. 134-147
               http://www.jstor.org/stable/1390762

        """
        x = inputs[0]
        L = inputs[1]
        dz = inputs[2]
        dx = outputs[0]
        N = x.shape[0]
        if self.lower:
            F = numpy.tril(dz)
            for k in xrange(N - 1, -1, -1):
                for j in xrange(k + 1, N):
                    for i in xrange(j, N):
                        F[i, k] -= F[i, j] * L[j, k]
                        F[j, k] -= F[i, j] * L[i, k]
                for j in xrange(k + 1, N):
                    F[j, k] /= L[k, k]
                    F[k, k] -= L[j, k] * F[j, k]
                F[k, k] /= (2 * L[k, k])
        else:
            F = numpy.triu(dz)
            M = N - 1
            for k in xrange(N - 1, -1, -1):
                for j in xrange(k + 1, N):
                    for i in xrange(j, N):
                        F[k, i] -= F[j, i] * L[k, j]
                        F[k, j] -= F[j, i] * L[k, i]
                for j in xrange(k + 1, N):
                    F[k, j] /= L[k, k]
                    F[k, k] -= L[k, j] * F[k, j]
                F[k, k] /= (2 * L[k, k])
        dx[0] = F

    def infer_shape(self, node, shapes):
        return [shapes[0]]


class Solve(Op):
    """
    Solves the matrix equation a x = b for x.

    Parameters:

    a: array, shape (M, M)
    b: array, shape (M,) or (M, N)
    sym_pos: (boolean) Assume a is symmetric and positive definite.
    lower: (boolean) Use only data contained in the lower triangle of a,
        if sym_pos is true. Default is to use upper triangle.
    overwrite_a: (boolean) Allow overwriting data in a (may enhance
    performance).
    overwrite_b: (boolean) Allow overwriting data in b (may enhance
    performance).

    Returns :

    x: array, shape (M,) or (M, N) depending on b
    """

    def __init__(self, sym_pos=False, lower=False, overwrite_a=False,
                 overwrite_b=False):
        self.sym_pos = sym_pos
        self.lower = lower
        self.overwrite_a = overwrite_a
        self.overwrite_b = overwrite_b

    def __eq__(self, other):
        return (type(self) == type(other) and self.sym_pos == other.sym_pos and
                self.lower == other.lower and
                self.overwrite_a == other.overwrite_a and
                self.overwrite_b == other.overwrite_b)

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

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Solve op")
        a = tensor.as_tensor_variable(a)
        b = tensor.as_tensor_variable(b)
        if a.ndim != 2 or  b.ndim > 2 or b.ndim == 0:
            raise TypeError('%s: inputs have improper dimensions:\n'
                    '\'a\' must have two and has %d,'
                    ' \'b\' must have either one or two and has %d' %
                            (self.__class__.__name__, a.ndim, b.ndim))

        out_type = tensor.TensorType(dtype=(a * b).dtype,
                                     broadcastable=b.type.broadcastable)()
        return Apply(self, [a, b], [out_type])

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def perform(self, node, inputs, output_storage):
        a, b = inputs

        if a.shape[0] != a.shape[1] or a.shape[1] != b.shape[0]:
            raise TypeError('%s: inputs have improper lengths' %
                            self.__class__.__name__)
        try:
            output_storage[0][0] = scipy.linalg.solve(a, b, self.sym_pos,
                        self.lower, self.overwrite_a, self.overwrite_b)
        except Exception, e:
            e.args = e.args + ('array \'a\' might be singular',) 
            raise 

    def grad(self, inputs, cost_grad):
        """
        See The Matrix Reference Manual,
        Copyright 1998-2011 Mike Brookes, Imperial College, London, UK

        Note: In contrast with the usual mathematical presentation, in order
        to apply theano's 'reshape' function wich implements row-order
        (i.e. C order), the differential expressions below have been derived
        around the row-vectorizations of inputs 'a' and 'b'.
        """

        A, b = inputs
        b = b
        ingrad = cost_grad[0]
        ingrad = tensor.as_tensor_variable(ingrad)
        N, _ = A.shape
        outgrad_a = -solve(A, b).dot(solve(A, ingrad).T).T
        outgrad_b = ingrad.T.dot(solve(A, tensor.eye(N))).T
        return [outgrad_a, outgrad_b]


def solve(a, b, sym_pos=False, lower=False, overwrite_a=False,
                 overwrite_b=False):
    localop = Solve(sym_pos, lower, overwrite_a, overwrite_b)
    return localop(a, b)

#TODO: Optimizations to replace multiplication by matrix inverse
#      with Ops solve() or solve_triangular()

class SolveTriangular(Op):
    """
    An instance of this class solves the matrix equation a x = b for x where
    'a' is triangular.

    Parameters:

    a: array, shape (M, M)
    b: array, shape (M,) or (M, N)
    lower: (boolean) Use only data contained in the lower triangle of a,
        if sym_pos is true. Default is to use upper triangle.
    unit_diagonal : (boolean) If True, diagonal elements of A are assumed to be
        1 and will not be referenced.
    overwrite_b: (boolean) Allow overwriting data in b (may enhance
        performance).

    Returns :

    x: array, shape (M,) or (M, N) depending on b
    """

    def __init__(self, lower=False, unit_diagonal=False, overwrite_b=False):

        self.lower = lower
        self.unit_diagonal = unit_diagonal
        self.overwrite_b = overwrite_b

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.lower == other.lower and
                self.unit_diagonal == other.unit_diagonal and
                self.overwrite_b == other.overwrite_b)

    def __hash__(self):
        return (hash(type(self)) ^ hash(self.lower) ^
                hash(self.unit_diagonal) ^ hash(self.overwrite_b))

    def props(self):
        return (self.lower, self.unit_diagonal, self.overwrite_b)

    def __str__(self):
        return "%s{%s, %s, %s}" % (self.__class__.__name__,
                "lower=".join(str(self.lower)),
                "unit_diagonal".join(str(self.unit_diagonal)),
                "overwrite_b=".join(str(self.overwrite_b)))

    def __repr__(self):
        return 'SolveTriangular{%s}' % str(self.props())

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the SolveTriangular op")
        a = tensor.as_tensor_variable(a)
        b = tensor.as_tensor_variable(b)
        if a.ndim != 2 or  b.ndim > 2 or b.ndim == 0:
            raise TypeError('%s: inputs have improper dimensions:\n'
                    '\'a\' must have two,'
                    ' \'b\' must have either one or two' %
                            self.__class__.__name__)

        out_type = tensor.TensorType(dtype=(a * b).dtype,
                                     broadcastable=b.type.broadcastable)()
        return Apply(self, [a, b], [out_type])

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def perform(self, node, inputs, output_storage):
        a, b = inputs
        if a.shape[0] != a.shape[1] or a.shape[1] != b.shape[0]:
            raise TypeError('%s: inputs have improper lengths' %
                            self.__class__.__name__)
        try:
            output_storage[0][0] = scipy.linalg.solve_triangular(a, b,
                            trans=0, lower=self.lower,
                       unit_diagonal=self.unit_diagonal,
                             overwrite_b=self.overwrite_b, debug=False)

        except Exception, e:
            e.args = e.args + ('array \'a\' might be singular',) 
            raise 



        #except:
        #    raise  Exception('%s: array \'a\' is singular'
        #                     % self.__class__.__name__)

    def grad(self, inputs, cost_grad):
        """
        Notes:
        1. The gradient is computed under the assumption that perturbations
        of the input array respect triangularity, i.e. partial derivatives wrt
        triangular region are zero.
        2. In contrast with the usual mathematical presentation, in order to
        apply theano's 'reshape' function wich implements row-order (i.e. C
        order), the differential expressions below have been derived based on
        the row-vectorizations of inputs 'a' and 'b'.

        See The Matrix Reference Manual,
        Copyright 1998-2011 Mike Brookes, Imperial College, London, UK
        """

        a, b = inputs
        ingrad = cost_grad
        ingrad = tensor.as_tensor_variable(ingrad)
        shp_a = (tensor.shape(inputs[0])[1],
                               tensor.shape(inputs[0])[1])
        I_M = tensor.eye(*shp_a)
        if self.lower:
            inv_a = solve_triangular(a, I_M, lower=True)
            tri_M = tril(tensor.ones(shp_a))
        else:
            inv_a = solve_triangular(a, I_M, lower=False)
            tri_M = triu(tensor.ones(shp_a))
        if b.ndim == 1:
            prod_a_b = tensor.tensordot(-b.T, inv_a.T, axes=1)
            prod_a_b = tensor.shape_padleft(prod_a_b)
            jac_veca = kron(inv_a, prod_a_b)
            jac_b = inv_a
            outgrad_veca = tensor.tensordot(ingrad, jac_veca, axes=1)
            outgrad_a = tensor.reshape(outgrad_veca,
                        (inputs[0].shape[0], inputs[0].shape[0])) * tri_M
            outgrad_b = tensor.tensordot(ingrad, jac_b, axes=1).flatten(ndim=1)
        else:
            ingrad_vec = ingrad.flatten(ndim=1)
            prod_a_b = tensor.tensordot(-b.T, inv_a.T, axes=1)
            jac_veca = kron(inv_a, prod_a_b)
            I_N = tensor.eye(tensor.shape(inputs[1])[1],
                               tensor.shape(inputs[1])[1])
            jac_vecb = kron(inv_a, I_N)
            outgrad_veca = tensor.tensordot(ingrad_vec, jac_veca, axes=1)
            outgrad_a = tensor.reshape(outgrad_veca,
                        (inputs[0].shape[0], inputs[0].shape[0])) * tri_M
            outgrad_vecb = tensor.tensordot(ingrad_vec, jac_vecb, axes=1)
            outgrad_b = tensor.reshape(outgrad_vecb,
                        (inputs[1].shape[0], inputs[1].shape[1]))
        return [outgrad_a, outgrad_b]


def solve_triangular(a, b, lower=False, unit_diagonal=False,
                             overwrite_b=False):
    return SolveTriangular(lower=lower, unit_diagonal=unit_diagonal,
                           overwrite_b=overwrite_b)(a, b)

class SolveTriangular(Op):
    """
    An instance of this class solves the matrix equation a x = b for x where
    'a' is triangular.

    Parameters:

    a: array, shape (M, M)
    b: array, shape (M,) or (M, N)
    lower: (boolean) Use only data contained in the lower triangle of a,
        if sym_pos is true. Default is to use upper triangle.
    unit_diagonal : (boolean) If True, diagonal elements of A are assumed to be
        1 and will not be referenced.
    overwrite_b: (boolean) Allow overwriting data in b (may enhance
        performance).

    Returns :

    x: array, shape (M,) or (M, N) depending on b
    """

    def __init__(self, lower=False, unit_diagonal=False, overwrite_b=False):

        self.lower = lower
        self.unit_diagonal = unit_diagonal
        self.overwrite_b = overwrite_b

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.lower == other.lower and
                self.unit_diagonal == other.unit_diagonal and
                self.overwrite_b == other.overwrite_b)

    def __hash__(self):
        return (hash(type(self)) ^ hash(self.lower) ^
                hash(self.unit_diagonal) ^ hash(self.overwrite_b))

    def props(self):
        return (self.lower, self.unit_diagonal, self.overwrite_b)

    def __str__(self):
        return "%s{%s, %s, %s}" % (self.__class__.__name__,
                "lower=".join(str(self.lower)),
                "unit_diagonal".join(str(self.unit_diagonal)),
                "overwrite_b=".join(str(self.overwrite_b)))

    def __repr__(self):
        return 'SolveTriangular{%s}' % str(self.props())

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the SolveTriangular op")
        a = tensor.as_tensor_variable(a)
        b = tensor.as_tensor_variable(b)
        if a.ndim != 2 or  b.ndim > 2 or b.ndim == 0:
            raise TypeError('%s: inputs have improper dimensions:\n'
                    '\'a\' must have two,'
                    ' \'b\' must have either one or two' %
                            self.__class__.__name__)

        out_type = tensor.TensorType(dtype=(a * b).dtype,
                                     broadcastable=b.type.broadcastable)()
        return Apply(self, [a, b], [out_type])

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def perform(self, node, inputs, output_storage):
        a, b = inputs
        if a.shape[0] != a.shape[1] or a.shape[1] != b.shape[0]:
            raise TypeError('%s: inputs have improper lengths' %
                            self.__class__.__name__)
        try:
            output_storage[0][0] = scipy.linalg.solve_triangular(a, b,
                            trans=0, lower=self.lower,
                       unit_diagonal=self.unit_diagonal,
                             overwrite_b=self.overwrite_b, debug=False)

        except Exception, e:
            e.args = e.args + ('array \'a\' might be singular',) 
            raise 



        #except:
        #    raise  Exception('%s: array \'a\' is singular'
        #                     % self.__class__.__name__)

    def grad(self, inputs, cost_grad):
        """
        Notes:
        1. The gradient is computed under the assumption that perturbations
        of the input array respect triangularity, i.e. partial derivatives wrt
        triangular region are zero.
        2. In contrast with the usual mathematical presentation, in order to
        apply theano's 'reshape' function wich implements row-order (i.e. C
        order), the differential expressions below have been derived based on
        the row-vectorizations of inputs 'a' and 'b'.

        See The Matrix Reference Manual,
        Copyright 1998-2011 Mike Brookes, Imperial College, London, UK
        """

        a, b = inputs
        ingrad = cost_grad
        ingrad = tensor.as_tensor_variable(ingrad)
        shp_a = (tensor.shape(inputs[0])[1],
                               tensor.shape(inputs[0])[1])
        I_M = tensor.eye(*shp_a)
        if self.lower:
            inv_a = solve_triangular(a, I_M, lower=True)
            tri_M = tril(tensor.ones(shp_a))
        else:
            inv_a = solve_triangular(a, I_M, lower=False)
            tri_M = triu(tensor.ones(shp_a))
        if b.ndim == 1:
            prod_a_b = tensor.tensordot(-b.T, inv_a.T, axes=1)
            prod_a_b = tensor.shape_padleft(prod_a_b)
            jac_veca = kron(inv_a, prod_a_b)
            jac_b = inv_a
            outgrad_veca = tensor.tensordot(ingrad, jac_veca, axes=1)
            outgrad_a = tensor.reshape(outgrad_veca,
                        (inputs[0].shape[0], inputs[0].shape[0])) * tri_M
            outgrad_b = tensor.tensordot(ingrad, jac_b, axes=1).flatten(ndim=1)
        else:
            ingrad_vec = ingrad.flatten(ndim=1)
            prod_a_b = tensor.tensordot(-b.T, inv_a.T, axes=1)
            jac_veca = kron(inv_a, prod_a_b)
            I_N = tensor.eye(tensor.shape(inputs[1])[1],
                               tensor.shape(inputs[1])[1])
            jac_vecb = kron(inv_a, I_N)
            outgrad_veca = tensor.tensordot(ingrad_vec, jac_veca, axes=1)
            outgrad_a = tensor.reshape(outgrad_veca,
                        (inputs[0].shape[0], inputs[0].shape[0])) * tri_M
            outgrad_vecb = tensor.tensordot(ingrad_vec, jac_vecb, axes=1)
            outgrad_b = tensor.reshape(outgrad_vecb,
                        (inputs[1].shape[0], inputs[1].shape[1]))
        return [outgrad_a, outgrad_b]


def solve_triangular(a, b, lower=False, unit_diagonal=False,
                             overwrite_b=False):
    return SolveTriangular(lower=lower, unit_diagonal=unit_diagonal,
                           overwrite_b=overwrite_b)(a, b)


class Eigvalsh(Op):
    """Generalized eigenvalues of a Hermetian positive definite eigensystem
    """

    __props__ = ('lower',)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not  available. Scipy is needed for the Eigvalsh op")

        if b == theano.tensor.NoneConst:
            a = as_tensor_variable(a)  
            assert a.ndim == 2

            out_dtype = theano.scalar.upcast(a.dtype)
            w = theano.tensor.vector(dtype=out_dtype)
            return Apply(self, [a], [w])
        else:
            a = as_tensor_variable(a)
            b = as_tensor_variable(b)
            assert a.ndim == 2
            assert b.ndim == 2

            out_dtype = theano.scalar.upcast(a.dtype, b.dtype)
            w = theano.tensor.vector(dtype=out_dtype)
            return Apply(self, [a, b], [w])

    def perform(self, node, inputs, (w,)):
        if len(inputs) == 2:
            w[0] = scipy.linalg.eigvalsh(a=inputs[0], b=inputs[1], lower=self.lower)
        else:
            w[0] = scipy.linalg.eigvalsh(a=inputs[0], b=None, lower=self.lower)

    def grad(self, inputs, g_outputs):
        a, b = inputs
        gw, = g_outputs
        return EigvalshGrad(self.lower)(a, b, gw)

    def infer_shape(self, node, shapes):
        n = shapes[0][0]
        return [(n,)]


class EigvalshGrad(Op):
    """Gradient of generalized eigenvalues of a Hermetian positive definite
    eigensystem
    """

    # Note: This Op (EigvalshGrad), should be removed and replaced with a graph
    # of theano ops that is constructed directly in Eigvalsh.grad.
    # But this can only be done once scipy.linalg.eigh is available as an Op
    # (currently the Eigh uses numpy.linalg.eigh, which doesn't let you
    # pass the right-hand-side matrix for a generalized eigenproblem.) See the
    # discussion on github at
    # https://github.com/Theano/Theano/pull/1846#discussion-diff-12486764

    __props__ = ('lower',)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower
        if lower:
            self.tri0 = numpy.tril
            self.tri1 = lambda a: numpy.triu(a, 1)
        else:
            self.tri0 = numpy.triu
            self.tri1 = lambda a: numpy.tril(a, -1)

    def make_node(self, a, b, gw):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the GEigvalsh op")
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        gw = as_tensor_variable(gw)  
        assert a.ndim == 2
        assert b.ndim == 2
        assert gw.ndim == 1

        out_dtype = theano.scalar.upcast(a.dtype, b.dtype, gw.dtype)
        out1 = theano.tensor.matrix(dtype=out_dtype)
        out2 = theano.tensor.matrix(dtype=out_dtype)
        return Apply(self, [a, b, gw], [out1, out2])

    def perform(self, node, (a, b, gw), outputs):
        w, v = scipy.linalg.eigh(a, b, lower=self.lower)
        gA = v.dot(numpy.diag(gw).dot(v.T))
        gB = - v.dot(numpy.diag(gw*w).dot(v.T))

        # See EighGrad comments for an explanation of these lines
        out1 = self.tri0(gA) + self.tri1(gA).T
        out2 = self.tri0(gB) + self.tri1(gB).T
        outputs[0][0] = numpy.asarray(out1, dtype=node.outputs[0].dtype)
        outputs[1][0] = numpy.asarray(out2, dtype=node.outputs[1].dtype)

    def infer_shape(self, node, shapes):
        return [shapes[0], shapes[1]]


def eigvalsh(a, b, lower=True):
    return Eigvalsh(lower)(a, b)


def kron(a, b):
    """ Kronecker product

    Same as scipy.linalg.kron(a, b).

    :note: numpy.kron(a, b) != scipy.linalg.kron(a, b)!
        They don't have the same shape and order when
        a.ndim != b.ndim != 2.

    :param a: array_like
    :param b: array_like
    :return: array_like with a.ndim + b.ndim - 2 dimensions.

    """
    a = tensor.as_tensor_variable(a)
    b = tensor.as_tensor_variable(b)
    if (a.ndim + b.ndim <= 2):
        raise TypeError('kron: inputs dimensions must sum to 3 or more. '
                        'You passed %d and %d.' % (a.ndim, b.ndim))
    o = tensor.outer(a, b)
    o = o.reshape(tensor.concatenate((a.shape, b.shape)),
                  a.ndim + b.ndim)
    shf = o.dimshuffle(0, 2, 1, * range(3, o.ndim))
    if shf.ndim == 3:
        shf = o.dimshuffle(1, 0, 2)
        o = shf.flatten()
    else:
        o = shf.reshape((o.shape[0] * o.shape[2],
                         o.shape[1] * o.shape[3]) +
                        tuple([o.shape[i] for i in range(4, o.ndim)]))
    return o
