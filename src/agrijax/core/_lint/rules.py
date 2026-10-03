"""The rule table of :mod:`agrijax.core.lint`, the AJ007 whitelist, the name sets the rules
match against, and the :class:`Finding` record."""

from __future__ import annotations

from dataclasses import dataclass

RULES: dict[str, tuple[str, str]] = {
    "AJ001": ("error", "Python branch on a value derived from the process arguments"),
    "AJ002": ("error", "subscript assignment inside a for loop"),
    "AJ003": ("warning", "unguarded log/sqrt/division inside a where/select branch"),
    "AJ004": ("warning", "return value not built with eqx.tree_at / replace"),
    "AJ005": ("warning", "missing docstring or no 'Source:' line"),
    "AJ006": ("error", "Python loop over a shape-derived range or an array (unrolled layer loop)"),
    "AJ007": ("warning", "bare numeric literal in process or kernel code"),
    "AJ008": (
        "warning",
        "import of another slot's package from code under processes/ (or of processes from core/, iface/)",
    ),
    "AJ009": ("warning", "module-level state mutated from a function (no hidden state in slots)"),
    "AJ010": ("warning", "@process reads/writes names a global path instead of a bound port"),
    "AJ011": ("warning", "@process reads from the forcing a quantity that a port record delivers"),
    "AJ012": (
        "warning",
        "import across the io / processes layers (io imports no processes, processes no io; "
        "only agrijax.sites imports both)",
    ),
    "AJ020": ("error", "NumPy call on a value derived from the arguments (NumPy does not trace)"),
    "AJ021": ("error", "in-place mutation of an argument (state / params / forcing or an alias of one)"),
}
#: warnings that ``--strict`` does not turn into failures (``--enforce RULE`` makes each fail):
#: AJ009 until the numerical-settings registry of the soil-water slot has moved to core, and
#: AJ011 until the soil-water day no longer reads the root water uptake from the forcing
NOT_STRICT_RULES: frozenset[str] = frozenset({"AJ009", "AJ011"})
#: file-level rules (run once per file, not per function)
FILE_RULES: frozenset[str] = frozenset({"AJ008", "AJ009", "AJ010", "AJ011", "AJ012"})

# ---- AJ007 whitelist (the one place that says which bare numbers are allowed) ----------------
#: values allowed as a bare literal anywhere (int or float, either sign where listed)
AJ007_TRIVIAL: frozenset[float] = frozenset({0.0, 1.0, -1.0})
#: largest ``|n|`` of an integer literal used as an index, slice bound, axis, shape entry or count
AJ007_MAX_INDEX: int = 16
#: largest ``|n|`` of an integer-valued exponent (``x**2``, ``x**-2``, ``jnp.power(x, 3)``)
AJ007_MAX_EXPONENT: int = 4
#: calls whose integer arguments are structure (shapes, axes, counts), not model numbers
AJ007_STRUCTURAL_CALLS: frozenset[str] = frozenset(
    {
        "range", "arange", "reshape", "zeros", "ones", "empty", "full", "eye", "identity",
        "expand_dims", "squeeze", "moveaxis", "swapaxes", "transpose", "broadcast_to",
        "concatenate", "stack", "hstack", "vstack", "split", "take", "take_along_axis", "roll",
        "flip", "tile", "repeat", "pad", "linspace", "index_in_dim", "slice_in_dim",
        "dynamic_slice", "dynamic_slice_in_dim", "dynamic_update_slice", "tril", "triu", "diag",
        "diagonal", "enumerate", "ndim", "shape",
    }
)  # fmt: skip
#: keyword arguments whose integer values are structure
AJ007_STRUCTURAL_KEYWORDS: frozenset[str] = frozenset(
    {"axis", "axes", "ndim", "shape", "n", "k", "num", "offset", "keepdims", "unroll", "length",
     "size", "start", "stop", "step", "indices_or_sections", "static_argnums", "in_axes",
     "out_axes", "decimals", "ord"}
)  # fmt: skip

_RISKY_CALLS = {"log", "log2", "log10", "sqrt", "rsqrt", "power", "pow", "arccos", "arcsin", "arctanh"}
_GUARD_CALLS = {"maximum", "clip", "clamp", "minimum", "abs", "exp", "where", "select", "square", "softplus"}
_UPDATE_CALLS = {"tree_at", "replace", "set", "set_path"}
_WHERE_CALLS = {"where", "select"}
_PROCESS_DECORATOR = "process"
#: rules applied to non-``@process`` functions of ``processes/`` modules (numerical kernels)
KERNEL_RULES: frozenset[str] = frozenset({"AJ001", "AJ002", "AJ003", "AJ006", "AJ007", "AJ020", "AJ021"})
ALL_RULES: frozenset[str] = frozenset(RULES)
#: directories whose non-process functions are numerical kernels (KERNEL_RULES)
_KERNEL_DIRS: frozenset[str] = frozenset({"processes"})
#: rules applied to non-``@process`` functions of ``forcing/`` modules: the forcing preprocessing
#: is host-side NumPy that runs before the day (not traced; the coupling contract keeps it outside
#: the day), so the tracing rules do not apply, but its coefficients are labelled like a kernel's
HOST_RULES: frozenset[str] = frozenset({"AJ007"})
#: directories whose non-process functions are host-side preprocessing (HOST_RULES)
_HOST_DIRS: frozenset[str] = frozenset({"forcing"})
_KERNEL_DIR = "processes"
_STATIC_ANNOTATIONS = {"bool", "int", "str", "float", "None", "Literal", "type"}
_STATIC_ATTRS = {"shape", "ndim", "dtype", "size"}
_STATIC_CALLS = {"ndim", "shape", "len", "isinstance", "hasattr", "callable", "type", "issubclass"}
_RANGE_CALLS = {"range", "arange"}
#: attributes that hold the size of a grid axis (AJ006)
_GRID_SIZE_ATTRS = {"n_node", "n_layer", "n_slice", "n_horizon", "n_lyr", "nlayr", "n_cell", "n_depth"}

# ---- AJ020 / AJ021 ---------------------------------------------------------------------------
#: NumPy functions that only query the structure or the dtype of their argument (they read
#: ``.shape``/``.ndim``/``.dtype`` and work on a tracer): never reported by AJ020
AJ020_STATIC_CALLS: frozenset[str] = frozenset(
    {"ndim", "shape", "size", "dtype", "result_type", "issubdtype", "finfo", "iinfo",
     "promote_types", "can_cast", "isscalar", "iterable", "broadcast_shapes"}
)  # fmt: skip
#: the structured marker of host-side code: the first line of the docstring of a function (or
#: of a module, for every function in it) starts with it, e.g. ``"""Host-side: build the grid
#: from the file records."""``; a mention anywhere else in a docstring does not count
AJ020_HOST_MARKER = "Host-side:"
#: NumPy names that are not functions (dtypes, scalar and array types, constants): passing one as
#: a value (``dtype=np.float32``, ``x.astype(np.int64)``, ``isinstance(x, np.ndarray)``) is fine
_NP_NON_FUNCTIONS: frozenset[str] = frozenset(
    {"bool_", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "intp",
     "uintp", "int_", "uint", "intc", "uintc", "short", "ushort", "byte", "ubyte", "longlong",
     "ulonglong", "float16", "float32", "float64", "float128", "longdouble", "half", "single",
     "double", "float_", "complex64", "complex128", "complex_", "csingle", "cdouble",
     "clongdouble", "object_", "str_", "bytes_", "void", "datetime64", "timedelta64", "generic",
     "number", "integer", "signedinteger", "unsignedinteger", "inexact", "floating",
     "complexfloating", "character", "flexible", "ndarray", "dtype", "pi", "e", "inf", "nan",
     "newaxis", "euler_gamma", "NINF", "PINF", "NAN", "NaN", "Inf", "Infinity", "NZERO", "PZERO"}
)  # fmt: skip
#: NumPy functions whose result is a NumPy callable (``np.vectorize(f)(x)`` runs NumPy on ``x``)
_NP_FACTORIES: frozenset[str] = frozenset({"vectorize", "frompyfunc"})
#: calls that run their function argument on the host with NumPy arrays (``jax.pure_callback``;
#: ``jax.debug.callback`` is recognised as an attribute or when imported from ``jax.debug``)
_CALLBACK_CALLS: frozenset[str] = frozenset({"pure_callback", "io_callback"})
#: methods that change their receiver in place wherever they are called (NumPy ``ndarray`` and
#: containers; ``pop`` / ``setdefault`` return a value, so a used result does not make them pure)
_INPLACE_ANYWHERE: frozenset[str] = frozenset(
    {"fill", "put", "itemset", "setfield", "setflags", "pop", "popitem", "setdefault"}
)
#: methods that change their receiver in place and return ``None``: reported when the result is
#: discarded (a statement, a lambda body, the element of a discarded comprehension); a functional
#: method of the same name (``jax.Array.sort``, an optimizer's ``update``) returns what is used
_INPLACE_IF_DISCARDED: frozenset[str] = frozenset(
    {"append", "extend", "insert", "remove", "clear", "update", "add", "discard", "sort",
     "partition", "resize", "reverse"}
)  # fmt: skip
#: array methods among the above (the fix hint names the functional jnp form)
_ARRAY_METHODS: frozenset[str] = frozenset({"fill", "put", "itemset", "setfield", "setflags", "sort",
                                            "partition", "resize", "reverse"})  # fmt: skip
#: dunder methods that change their receiver (``x.__setitem__(k, v)``, ``x.__iadd__(y)``)
_DUNDER_MUTATORS: frozenset[str] = frozenset(
    {"__setattr__", "__delattr__", "__setitem__", "__delitem__", "__iadd__", "__isub__", "__imul__",
     "__imatmul__", "__itruediv__", "__ifloordiv__", "__imod__", "__ipow__", "__ilshift__",
     "__irshift__", "__iand__", "__ixor__", "__ior__"}
)  # fmt: skip
#: receivers that make ``R.method(obj, ...)`` an unbound call on ``obj`` (``dict.update(state, d)``,
#: ``object.__setattr__(x, "a", v)``; ``np.ndarray`` and ``type(x)`` count too)
_TYPE_RECEIVERS: frozenset[str] = frozenset({"object", "dict", "list", "set", "bytearray", "type"})
#: functions of :mod:`operator` that change their first argument
_OPERATOR_MUTATORS: frozenset[str] = frozenset(
    {"setitem", "delitem", "iadd", "iand", "iconcat", "ifloordiv", "ilshift", "imatmul", "imod",
     "imul", "ior", "ipow", "irshift", "isub", "itruediv", "ixor", "__setitem__", "__delitem__",
     "__iadd__", "__iand__", "__iconcat__", "__ifloordiv__", "__ilshift__", "__imatmul__",
     "__imod__", "__imul__", "__ior__", "__ipow__", "__irshift__", "__isub__", "__itruediv__",
     "__ixor__"}
)  # fmt: skip
#: NumPy functions that write into their first argument
_NP_INPLACE_FUNCS: frozenset[str] = frozenset(
    {"copyto", "put", "place", "putmask", "fill_diagonal", "put_along_axis", "random.shuffle"}
)
#: methods whose result is the receiver or a view of it, not a copy (``a.view()``, ``a.reshape``)
_ALIAS_METHODS: frozenset[str] = frozenset({"view", "reshape", "ravel", "squeeze", "transpose", "swapaxes"})
#: NumPy functions that return their argument itself or a view of it
_NP_ALIAS_FUNCS: frozenset[str] = frozenset(
    {"asarray", "asanyarray", "ascontiguousarray", "asfortranarray", "ravel", "reshape",
     "transpose", "squeeze", "swapaxes", "moveaxis", "expand_dims", "atleast_1d", "atleast_2d",
     "atleast_3d"}
)  # fmt: skip
#: calls whose result is a new container holding the elements of their arguments (shallow)
_SHALLOW_CALLS: frozenset[str] = frozenset(
    {"dict", "list", "tuple", "set", "frozenset", "sorted", "iter", "reversed", "enumerate", "zip", "copy"}
)
#: methods that give a new container of the receiver's elements (``d.copy()``, ``d.items()``)
_SHALLOW_METHODS: frozenset[str] = frozenset({"copy", "items", "values"})
#: base classes that make a class a pytree record (its ``self`` is traced and owned by the caller)
_PYTREE_BASES: frozenset[str] = frozenset({"Module", "State", "Params", "Coefficients"})
#: methods in which ``self`` is the object being built, not the caller's (AJ021)
_CONSTRUCTOR_METHODS: frozenset[str] = frozenset(
    {"__init__", "__post_init__", "__check_init__", "__new__", "__setattr__", "__delattr__", "__setstate__"}
)
#: the local names ``numpy`` / ``operator`` are bound to when a file's imports say nothing else
_NUMPY_DEFAULT: dict[str, str] = {"np": "", "numpy": ""}
_OPERATOR_DEFAULT: dict[str, str] = {"operator": ""}


@dataclass(frozen=True)
class Finding:
    rule: str
    level: str
    path: str
    line: int
    col: int
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.rule} [{self.level}] {self.message}"
