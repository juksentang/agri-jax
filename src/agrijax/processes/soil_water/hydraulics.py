r"""Brooks-Corey soil hydraulic functions in the modified form used by RZWQM2.

Pure ``jnp`` functions of a matric potential ``h`` [cm] (negative when
unsaturated) and a :class:`SoilHydraulicParams` pytree, vectorised over
nodes. Every branch is written with :func:`jnp.where` on a *clamped* argument
so that both branches are finite for every input in range and ``jax.grad``
never sees ``inf * 0``.

Equations (Ahuja et al., 2000, RZWQM manual chapter 3; ``h`` in cm, ``|h| = -h``)
-------------------------------------------------------------------------------

Water retention, with the near-saturation linear segment of the *modified*
Brooks-Corey curve (``a1`` is the "constant for the theta(h) curve"):

.. math::

    \theta(h) = \begin{cases}
        \theta_s                                  & h \ge 0 \\
        \theta_s + a_1 h                          & -h_b < h < 0 \\
        \theta_r + B\,|h|^{-\lambda}              & h \le -h_b
    \end{cases}
    \qquad B = (\theta_s - \theta_r - a_1 h_b)\, h_b^{\lambda}

so that :math:`\theta(-h_b) = \theta_s - a_1 h_b` from both sides.  The
specific moisture capacity :math:`C(h) = d\theta/dh` is ``0``, ``a1`` and
:math:`B\lambda |h|^{-\lambda-1}` on the three segments.  The inverse is

.. math::

    h(\theta) = \begin{cases}
        0                                          & \theta \ge \theta_s \\
        -(\theta_s - \theta)/a_1                   & \theta_s - a_1 h_b \le \theta < \theta_s \\
        -h_b\, S_e^{-1/\lambda},\;
            S_e = \frac{\theta - \theta_r}{\theta_s - \theta_r - a_1 h_b}
                                                   & \theta < \theta_s - a_1 h_b
    \end{cases}

Hydraulic conductivity, two power-law segments joined at the K-curve bubbling
pressure ``hb_k`` (RZWQM keeps it separately from the retention ``hb``; both
are equal in the CA-TPA file):

.. math::

    K(h) = \begin{cases}
        K_s                       & h \ge 0 \\
        K_s\,|h|^{-N_1}           & -h_{bk} \le h < 0 \\
        C_2\,|h|^{-\varepsilon}   & h < -h_{bk}
    \end{cases}
    \qquad C_2 = K_s\, h_{bk}^{\,\varepsilon - N_1}

The value of :math:`C_2` makes :math:`K` continuous at :math:`-h_{bk}`.
With the default :math:`N_1 = 0` the first segment is flat and
:math:`K = K_s (h_{bk}/|h|)^{\varepsilon}` below the air-entry value, i.e.
the classical Brooks-Corey conductivity with :math:`\varepsilon = 2 + 3\lambda`
(RZWQM instead takes :math:`\varepsilon` per texture class, 2.966 for CA-TPA).

How RZWQM2 actually uses the 13 ``SOILHP`` values (reference-model source, not redistributed)
---------------------------------------------------------------------------------------------

The rzwqm.dat "SOIL HORIZON HYDRAULIC PROPERTIES" block holds, per horizon,
rec1 ``hb lambda eps ksat wr ws`` and rec2 ``fc13 fc110 wp hb_k c2 n1 a1``.
The Fortran functions in ``RZWQM/Rzrich.for`` that evaluate the curves are

* ``WC`` (theta from h)                lines 1971-2046: the three-segment
  retention above, plus a fourth "original curve" segment for ``h < -10 hb``
  that only differs after tillage has changed the horizon (``TRHYDP`` saved
  at ``ISTAT = -1``);
* ``SPMOIS`` (dtheta/dh)               lines 1892-1969, same segmentation;
* ``WCH`` (h from theta)               lines 2048-2143: same inverse, with a
  ``1e-10`` tolerance on the ``theta >= ws - a1*hb`` test, and it *keeps the
  previous h* on the flat plateau (``a1 = 0``, ``theta >= ws``), where the
  inverse does not exist; this module returns ``h = 0`` there;
* ``POINTK`` (K from h)                lines 809-880: the two-segment K(h),
  with the ``C22``/``SN22`` copies that only matter after tillage;
* ``BCBETA``                           lines 1860-1889: ``B`` as above;
* ``HYDPAR``                           lines 622-731: per-node loop that
  clamps ``theta`` at the porosity, calls ``WCH`` then ``WC`` (the round trip
  this module's tests check), and clamps ``h >= HMIN`` (``-15000`` cm by default,
  :data:`H_CLAMP_RZWQM`).

The parameters are *not* taken from the file as they stand.  ``SOILPR``
(``RZWQM/RZTEST.for`` lines 4380-4866, called once per horizon at
initialisation from ``Rzmain.for`` line 5969 with ``ITYPE = SOILPP(1,J)``,
the first token of the soil-physical record, ``0`` for every CA-TPA horizon)
takes, for ``ITYPE = 0`` and unchanged bulk density, the ``ELSE`` branch at
lines 4587-4637:

* ``hb lambda eps ksat wr ws hb_k n1 a1`` are used as read (lines 4589-4597);
* **fc13 and fc110 are recomputed from the curve** at ``-333`` and ``-100`` cm
  (lines 4609-4625, ``FC33 = B/333**lambda + WR`` when ``hb < 333``, the
  linear segment when ``hb >= 333``), overwriting the file values;
* ``C2 = ksat * hb_k**(eps - n1)`` (line 4632), overwriting the file value;
* on exit (label 10, lines 4849-4862) **wp is recomputed** as
  ``B/|HWP|**lambda + WR`` with ``HWP = -15000`` cm (line 4858;
  ``Rzmain.for`` line 4589 sets ``HWP``), overwriting the file value.

So in RZWQM the curve is the only source of truth: the rec2 values
``fc13 fc110 wp c2`` are *derived diagnostics* that the model overwrites at
start-up, and they never override a curve segment.  Checked two ways
(``tests/integration/test_hydraulics_reference.py``):

* *File semantics, all 15 RZWQM_sw_batch scenarios (105 horizons, every one
  ``ITYPE = 0`` with ``hb < 333`` cm):* ``theta_of_h(-333)``, ``theta_of_h(-100)``
  and ``theta_of_h(-15000)`` reproduce the rec2 ``fc13 fc110 wp`` written with six
  decimals, largest difference 6.95e-7 over the 315 values; a 1 % change of the
  head would move theta by at least 3.7e-5 on every horizon, so the heads are
  resolved.  The file ``c2`` differs from ``ksat * hb_k**(eps - n1)`` by more
  than 1e-4 (relative) on 26 of the 105 horizons (CA-TPA's ``c2 = 7440.01``
  equals ``2.59 * hb_k**eps``, horizon 5's Ksat, on all five) and is silently
  replaced.
* *Binary experiment, CA-TPA 2015 (RZWQM2 ``main_ryzen5_avx512``):* ``fc13``,
  ``fc110`` and ``wp`` of horizon 1 at -30 % give a ``.ana`` byte-identical to the
  base run (all 138 columns) and the same yield, 9916 kg/ha.  Positive controls
  on the same horizon: Ksat at -50 % changes 61 of 138 columns and the yield to
  9923 kg/ha; theta_r at -30 % (a curve parameter) changes 57 columns and the
  yield to 9937 kg/ha.  The calibration ranges in ``all_parameters.csv`` for
  ``FC 1/3 WC``, ``FC 1/10 WC`` and ``WP WC`` therefore perturb nothing in RZWQM
  (for ``ITYPE = 0``): sampling them is a no-op.

The derived values are still carried in the pytree because other RZWQM
processes read them directly (``SOILHP(7)`` field capacity and ``SOILHP(9)``
wilting point in the PET and crop-water routines): use :func:`derive_rzwqm`
to refresh them from the primary parameters exactly as ``SOILPR`` does.

For ``ITYPE > 0`` (texture-class estimation, lines 4622-4835) RZWQM scales a
reference curve so that it passes through the given ``fc13`` (or ``fc110``);
that path is not implemented here since every scenario in the batch data
uses ``ITYPE = 0``.  Tillage (``MATILL``, ``Rzman.for`` lines 2704-3121) and
bulk-density change (``CHBD``) re-derive ``SOILHP`` through ``SOILPR``; that
re-derivation is not implemented here (the new parameters are an input).

Post-tillage two-segment curve (:class:`TilledSoilHydraulicParams`)
-------------------------------------------------------------------

After tillage has changed ``SOILHP`` the curves are not single-segment any
more: tillage loosens the soil at the wet end, while the dry end, set by the
texture, keeps the pre-tillage curve (Ahuja et al. 1998, changes of the water
retention curve by tillage and reconsolidation; RZWQM manual, Ahuja et al.
2000 ch. 3). RZWQM2 keeps a copy of the start-up parameters per horizon
(``TRHYDP`` in ``WC``/``SPMOIS``/``WCH``, ``C22``/``SN22`` in ``POINTK``, saved by the
``ISTAT = -1`` calls of ``TILADJ``, ``Rzmain.for`` lines 6436-6441, and the
first ``SPMOIS``/``POINTK`` call per horizon) and evaluates, with ``c`` the
current and ``o`` the original parameters and ``f = 10``:

.. math::

    \theta(h) = \begin{cases}
        \theta_c(h)                                   & h \ge -f\,h_{b,c} \\
        \theta_{r,o} + B_o\,|h|^{-\lambda_o}          & h < -f\,h_{b,c}
    \end{cases}
    \qquad
    K(h) = \begin{cases}
        K_c(h)                                        & h > -f\,h_{bk,c} \\
        C_{2,o}\,|h|^{-\varepsilon_o}                 & h \le -f\,h_{bk,c}
    \end{cases}

``C(h)`` follows ``theta`` segment by segment (``SPMOIS``); the inverse ``h(theta)``
(``WCH``) takes the original branch for ``theta <= theta_o(-f h_{b,o})`` below the
linear segment of the current curve. Note the two thresholds: ``WC`` switches at
the *current* ``-f hb``, ``WCH`` at the water content of the *original* curve at
the *original* ``-f hb`` (``WC10S2``); both are reproduced as they are. The curve
jumps at ``-f hb`` (by up to 4.8e-3 on the CA-TPA 2015 calls). With
``original == current`` every function returns the single-segment values.
Source lines: ``WC`` ``Rzrich.for`` 2033-2037, ``SPMOIS`` 1954-1958, ``WCH``
2105 and 2123-2128, ``POINTK`` 860-864 and 875-876.

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.),
2000. Root Zone Water Quality Model. Water Resources Publications, ch. 3;
Brooks, R.H., Corey, A.T., 1964. Hydraulic properties of porous media.
Hydrology Paper 3, Colorado State University.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.coefficients import Coefficients, Provenance, coef, numerical_guard
from agrijax.core.state import Params, field

__all__ = [
    "A1_MIN",
    "H_CLAMP_RZWQM",
    "H_FC13",
    "H_FC110",
    "H_MIN",
    "H_WP",
    "RZWQM_HYDRAULICS",
    "AnyHydraulicParams",
    "HydraulicsCoefficients",
    "SoilHydraulicParams",
    "TilledSoilHydraulicParams",
    "c2_of_params",
    "c_of_h",
    "c_of_h_tilled",
    "derive_rzwqm",
    "h_of_theta",
    "h_of_theta_tilled",
    "k_of_h",
    "k_of_h_tilled",
    "theta_of_h",
    "theta_of_h_tilled",
]

_REF = "rzwqm2-4.6"
_BOOK = "Ahuja et al. (2000)"


class HydraulicsCoefficients(Coefficients):
    """The reference heads of the RZWQM2 soil-water characteristic (``SOILPR``, ``INPUT``, ``HYDPAR``).

    The Brooks-Corey parameters themselves are read from ``rzwqm.dat`` (:class:`SoilHydraulicParams`);
    the coefficients here are the heads at which RZWQM2 evaluates the curve for its derived
    diagnostics (:func:`derive_rzwqm`) and the dry-end clamp of the state. They are conventions of the
    reference model (1/3 bar written as 333 cm, 15 bar as 15000 cm), declared with ``calibrate=False``;
    the reference source writes the magnitudes (``333.0D0``, ``-15000.D0``). The two
    ``tillage_split_*`` factors place the joint of the post-tillage two-segment curves
    (:class:`TilledSoilHydraulicParams`), written ``10D0`` / ``10.0D0`` in the source.
    """

    h_fc13: float = coef(
        -333.0,
        "cm",
        "matric potential of the 1/3-bar field capacity SOILPR derives (fc13 = theta(h))",
        Provenance(
            _REF,
            file="RZWQM/RZTEST.for",
            line=4610,
            routine="SOILPR",
            paper=_BOOK,
            note="written as 333.0D0 (|h|); also the linear-segment case on line 4615",
        ),
        calibrate=False,
    )
    h_fc110: float = coef(
        -100.0,
        "cm",
        "matric potential of the 1/10-bar field capacity SOILPR derives (fc110 = theta(h))",
        Provenance(
            _REF,
            file="RZWQM/RZTEST.for",
            line=4620,
            routine="SOILPR",
            paper=_BOOK,
            note="written as 100.0D0 (|h|); also the linear-segment case on line 4625",
        ),
        calibrate=False,
    )
    h_wp: float = coef(
        -15000.0,
        "cm",
        "matric potential of the 15-bar wilting point SOILPR derives (wp = theta(HWP))",
        Provenance(
            _REF,
            file="RZWQM/Rzmain.for",
            line=4589,
            routine="INPUT",
            paper=_BOOK,
            note="HWP of the 16-item soil-physics control record; used by SOILPR (RZTEST.for:4858)",
        ),
        calibrate=False,
        fortran_name="HWP",
    )
    h_clamp: float = coef(
        -15000.0,
        "cm",
        "dry-end clamp of the matric potential of the state, H = max(H, Hmin)",
        Provenance(
            _REF,
            file="RZWQM/Rzmain.for",
            line=4587,
            routine="INPUT",
            paper=_BOOK,
            note="Hmin of the 16-item record (item 17 of a 17/19-item record; CA-TPA: -15000); "
            "applied in HYDPAR (Rzrich.for:706) and CNHEAD (Rzrich.for:387, 426)",
        ),
        calibrate=False,
        fortran_name="HMIN",
    )
    tillage_split_retention: float = coef(
        10.0,
        "-",
        "multiple of the current bubbling pressure hb below which (h < -f hb) the post-tillage "
        "theta(h) and C(h) follow the pre-tillage curve; WCH switches at theta_orig(-f hb_orig)",
        Provenance(
            _REF,
            file="RZWQM/Rzrich.for",
            line=2033,
            routine="WC",
            paper="Ahuja et al. (1998), Soil Sci. Soc. Am. J. 62:1228-1233",
            note="also SPMOIS (Rzrich.for:1954) and the WC10S2 threshold of WCH (Rzrich.for:2105); "
            "a segment boundary where the curve jumps: zero derivative almost everywhere",
        ),
        calibrate=False,
    )
    tillage_split_conductivity: float = coef(
        10.0,
        "-",
        "multiple of the current K-curve bubbling pressure hb_k at and below which (h <= -f hb_k) "
        "the post-tillage K(h) follows the pre-tillage C2 |h|**(-eps)",
        Provenance(
            _REF,
            file="RZWQM/Rzrich.for",
            line=875,
            routine="POINTK",
            paper="Ahuja et al. (1998), Soil Sci. Soc. Am. J. 62:1228-1233",
            note="pre-tillage C2 and eps are the C22/SN22 copies of the first call per horizon "
            "(Rzrich.for:860-864); a segment boundary: zero derivative almost everywhere",
        ),
        calibrate=False,
    )


#: the RZWQM2 reference heads (the defaults of the hydraulic functions)
RZWQM_HYDRAULICS = HydraulicsCoefficients()

#: matric potential of "1/3 bar" field capacity used by RZWQM [cm] (alias of ``RZWQM_HYDRAULICS.h_fc13``).
H_FC13: float = RZWQM_HYDRAULICS.h_fc13
#: matric potential of "1/10 bar" field capacity used by RZWQM [cm] (alias of ``RZWQM_HYDRAULICS.h_fc110``).
H_FC110: float = RZWQM_HYDRAULICS.h_fc110
#: matric potential of the 15 bar wilting point used by RZWQM [cm] (``HWP``),
#: alias of ``RZWQM_HYDRAULICS.h_wp``.
H_WP: float = RZWQM_HYDRAULICS.h_wp
#: lowest matric potential returned by :func:`h_of_theta` [cm]: an overflow guard for ``theta -> theta_r``,
#: not a physical limit. The dry-end clamp of the *state* belongs to the Richards process, not to the
#: curve: see :data:`H_CLAMP_RZWQM`.
H_MIN: float = numerical_guard(
    "hydraulics.h_min",
    -1.0e30,
    "lowest matric potential [cm] of h_of_theta: overflow guard as theta -> theta_r",
)
#: dry-end clamp of the matric potential in RZWQM2 [cm] (``Hmin``: ``H = MAX(H, HMIN)`` in ``HYDPAR``,
#: ``Rzrich.for`` line 706, and in the Richards solver ``CNHEAD``, lines 387 and 426). The active default is
#: ``Hmin = -15000`` cm set with the 16-item soil-physics control record (``Rzmain.for`` line 4587,
#: equal to ``HWP``); a 17- or 19-item record reads it as item 17 instead (``Rzmain.for`` lines
#: 4590-4599), and CA-TPA's 19-item record gives ``-15000`` too. The ``HMIN = -35000`` PARAMETER seen
#: in older comments (``Rzrich.for`` lines 46, 240; ``Rzmain.for`` line 6104) is commented out.
#: Alias of ``RZWQM_HYDRAULICS.h_clamp``.
H_CLAMP_RZWQM: float = RZWQM_HYDRAULICS.h_clamp
#: ``a1`` at or below this is treated as 0 by :func:`h_of_theta` (no linear segment) [cm3 cm-3 cm-1];
#: RZWQM's reference value is 0.002, so anything this small is a rounding artefact, not a segment.
A1_MIN: float = numerical_guard(
    "hydraulics.a1_min", 1.0e-12, "a1 [cm3 cm-3 cm-1] at or below which h_of_theta has no linear segment"
)

_LOG_H_MAX: float = float(np.log(-H_MIN))
#: value floor of the effective saturation inside the log; 1 / _SE_FLOOR must stay finite in float32.
_SE_FLOOR: float = numerical_guard(
    "hydraulics.se_floor",
    1.0e-30,
    "floor of the effective saturation inside the log (1/floor finite in float32)",
)
#: floor of |h| [cm] of the wet K(h) segment, finite |h|**(-n1) for n1 > 0 as h -> 0-
_ABSH_WET_FLOOR: float = numerical_guard(
    "hydraulics.absh_wet_floor",
    1.0e-6,
    "floor of |h| [cm] in the wet K(h) segment (finite for n1 > 0 at h -> 0-)",
)


def _as_horizon_index(x: Any) -> tuple[int, ...] | None:
    if x is None:
        return None
    return tuple(int(i) for i in np.asarray(x).ravel())


class SoilHydraulicParams(Params):
    """Modified Brooks-Corey parameters per soil horizon (RZWQM ``SOILHP(1:13, horizon)``).

    Every array field has shape ``[n_horizon]`` (or is a scalar in single-horizon
    use).  ``node_horizon`` is the static node -> horizon index map (``NDXN2H`` in
    RZWQM); when it is set, the hydraulic functions gather the horizon
    parameters onto the ``[n_node]`` axis of ``h``.  When it is ``None`` the
    parameter arrays broadcast directly against ``h``.

    The four fields ``fc13 fc110 wp c2`` are *derived* in RZWQM (see module
    docstring); the curve functions never read them.
    """

    # ---- rec1: primary retention and conductivity parameters -----------------
    hb: Array = field(
        dims=("n_horizon?",),
        unit="cm",
        description="bubbling (air-entry) pressure of the theta(h) curve, > 0",
        fortran_name="SOILHP(1)",
    )
    lambda_: Array = field(
        dims=("n_horizon?",),
        unit="-",
        description="Brooks-Corey pore size distribution index",
        fortran_name="SOILHP(2)",
    )
    eps: Array = field(
        dims=("n_horizon?",),
        unit="-",
        description="exponent of the K(h) curve below hb_k (N2 in SOILPR)",
        fortran_name="SOILHP(3)",
    )
    ksat: Array = field(
        dims=("n_horizon?",),
        unit="cm hr-1",
        description="saturated hydraulic conductivity",
        fortran_name="SOILHP(4)",
    )
    theta_r: Array = field(
        dims=("n_horizon?",), unit="cm3 cm-3", description="residual water content", fortran_name="SOILHP(5)"
    )
    theta_s: Array = field(
        dims=("n_horizon?",), unit="cm3 cm-3", description="saturated water content", fortran_name="SOILHP(6)"
    )
    # ---- rec2 ----------------------------------------------------------------
    fc13: Array = field(
        dims=("n_horizon?",),
        unit="cm3 cm-3",
        description="water content at 1/3 bar (-333 cm); derived from the curve by RZWQM at start-up",
        fortran_name="SOILHP(7)",
    )
    fc110: Array = field(
        dims=("n_horizon?",),
        unit="cm3 cm-3",
        description="water content at 1/10 bar (-100 cm); derived from the curve by RZWQM at start-up",
        fortran_name="SOILHP(8)",
    )
    wp: Array = field(
        dims=("n_horizon?",),
        unit="cm3 cm-3",
        description="water content at 15 bar (-15000 cm); derived from the curve by RZWQM at start-up",
        fortran_name="SOILHP(9)",
    )
    hb_k: Array = field(
        dims=("n_horizon?",),
        unit="cm",
        description="bubbling pressure of the K(h) curve, > 0 (S1 in SOILPR)",
        fortran_name="SOILHP(10)",
    )
    c2: Array = field(
        dims=("n_horizon?",),
        unit="cm hr-1 cm^eps",
        description="second intercept of the K(h) curve; RZWQM overwrites it with ksat * hb_k**(eps - n1)",
        fortran_name="SOILHP(11)",
    )
    n1: Array = field(
        dims=("n_horizon?",),
        unit="-",
        description="exponent of the K(h) curve between -hb_k and 0 (0 in every RZWQM reference class)",
        fortran_name="SOILHP(12)",
    )
    a1: Array = field(
        dims=("n_horizon?",),
        unit="cm3 cm-3 cm-1",
        description="slope of the linear theta(h) segment between -hb and 0 (0 = classical Brooks-Corey)",
        fortran_name="SOILHP(13)",
    )
    # ---- geometry ------------------------------------------------------------
    node_horizon: tuple[int, ...] | None = field(
        description="static node -> horizon index (0-based), NDXN2H in RZWQM; None = already per node",
        fortran_name="NDXN2H",
        static=True,
        converter=_as_horizon_index,
        default=None,
    )

    @property
    def n_horizon(self) -> int:
        """Number of horizons (size of the leading axis of ``hb``; 1 for scalars)."""
        return int(np.prod(np.shape(self.hb))) if np.ndim(self.hb) else 1

    def at_nodes(self) -> SoilHydraulicParams:
        """Gather every horizon parameter onto the node axis; ``node_horizon`` becomes ``None``.

        Idempotent: a pytree without a node map is returned unchanged.
        """
        if self.node_horizon is None:
            return self
        idx = jnp.asarray(self.node_horizon, dtype=jnp.int32)
        gathered = {name: jnp.take(getattr(self, name), idx, axis=0) for name in _ARRAY_FIELDS}
        return SoilHydraulicParams(**gathered, node_horizon=None)

    @classmethod
    def from_rzwqm_records(
        cls, rec1: Any, rec2: Any, *, node_horizon: Any = None, derive: bool = True
    ) -> SoilHydraulicParams:
        """Build from the rzwqm.dat horizon records ``rec1[:, 1:7]`` (sequence number stripped) and ``rec2``.

        ``rec1`` rows are ``(hb, lambda, eps, ksat, wr, ws)`` (the leading horizon
        number must already be stripped), ``rec2`` rows are
        ``(fc13, fc110, wp, hb_k, c2, n1, a1)``.  With ``derive=True`` the four
        derived values are recomputed as ``SOILPR`` does at start-up.
        """
        r1 = jnp.asarray(np.asarray(rec1, dtype=float))
        r2 = jnp.asarray(np.asarray(rec2, dtype=float))
        if r1.shape[-1] != 6 or r2.shape[-1] != 7:
            raise ValueError(f"expected rec1[..., 6] and rec2[..., 7], got {r1.shape} and {r2.shape}")
        p = cls(
            hb=r1[..., 0],
            lambda_=r1[..., 1],
            eps=r1[..., 2],
            ksat=r1[..., 3],
            theta_r=r1[..., 4],
            theta_s=r1[..., 5],
            fc13=r2[..., 0],
            fc110=r2[..., 1],
            wp=r2[..., 2],
            hb_k=r2[..., 3],
            c2=r2[..., 4],
            n1=r2[..., 5],
            a1=r2[..., 6],
            node_horizon=node_horizon,
        )
        return derive_rzwqm(p) if derive else p

    @classmethod
    def from_rzwqm_dat(
        cls, hydraulics: Mapping[str, Any], *, node_horizon: Any = None, derive: bool = True
    ) -> SoilHydraulicParams:
        """Build from the ``hydraulics`` mapping of :class:`agrijax.io.rzwqm.dat.RzwqmDat`.

        Keys: ``hb lam eps ksat theta_r theta_s theta_fc33 theta_fc10 theta_wp hb_k c2 n1 a1``
        (``ksat_lat`` is ignored; it is the tile-drain lateral conductivity).
        """
        rec1 = np.stack([np.asarray(hydraulics[k], dtype=float) for k in _DAT_REC1], axis=-1)
        rec2 = np.stack([np.asarray(hydraulics[k], dtype=float) for k in _DAT_REC2], axis=-1)
        return cls.from_rzwqm_records(rec1, rec2, node_horizon=node_horizon, derive=derive)


_DAT_REC1: tuple[str, ...] = ("hb", "lam", "eps", "ksat", "theta_r", "theta_s")
_DAT_REC2: tuple[str, ...] = ("theta_fc33", "theta_fc10", "theta_wp", "hb_k", "c2", "n1", "a1")

_ARRAY_FIELDS: tuple[str, ...] = (
    "hb",
    "lambda_",
    "eps",
    "ksat",
    "theta_r",
    "theta_s",
    "fc13",
    "fc110",
    "wp",
    "hb_k",
    "c2",
    "n1",
    "a1",
)


class TilledSoilHydraulicParams(Params):
    """Post-tillage hydraulics: the current parameters and the pre-tillage (start-up) ones.

    RZWQM2 evaluates the current curve ``current`` (``SOILHP`` after tillage and
    reconsolidation) near saturation and the start-up curve ``original`` (``TRHYDP``,
    ``C22``/``SN22``) at the dry end, joined at ``-10 hb`` (see the module docstring and
    :attr:`HydraulicsCoefficients.tillage_split_retention`). Both are full
    :class:`SoilHydraulicParams` pytrees of the same shape, so every parameter of both
    curves is a calibratable leaf. The hydraulic functions of this module accept this
    class wherever they accept :class:`SoilHydraulicParams`; ``original == current`` gives
    the single-segment values.
    """

    current: SoilHydraulicParams = field(
        description="parameters of the curve now (after tillage / reconsolidation)", fortran_name="SOILHP"
    )
    original: SoilHydraulicParams = field(
        description="start-up (pre-tillage) parameters, used below -10 hb",
        fortran_name="TRHYDP",
    )

    def __check_init__(self) -> None:
        if not isinstance(self.current, SoilHydraulicParams) or not isinstance(
            self.original, SoilHydraulicParams
        ):
            raise TypeError("current and original must be SoilHydraulicParams")
        if self.current.node_horizon != self.original.node_horizon:
            raise ValueError("current and original must share the node -> horizon map")

    @classmethod
    def untilled(cls, params: SoilHydraulicParams) -> TilledSoilHydraulicParams:
        """A soil before any tillage: the original curve is the current one."""
        return cls(current=params, original=params)

    @property
    def hb(self) -> Array:
        """Bubbling pressure of the current curve (the air-entry kink the solver works around)."""
        return self.current.hb

    @property
    def n_horizon(self) -> int:
        """Number of horizons."""
        return self.current.n_horizon

    def at_nodes(self) -> TilledSoilHydraulicParams:
        """Gather both curves onto the node axis (see :meth:`SoilHydraulicParams.at_nodes`)."""
        return TilledSoilHydraulicParams(current=self.current.at_nodes(), original=self.original.at_nodes())


#: either parameter set accepted by the hydraulic functions
AnyHydraulicParams = SoilHydraulicParams | TilledSoilHydraulicParams


# ---------------------------------------------------------------------------
# safe primitives
# ---------------------------------------------------------------------------


def _clamp_min(x: Array, lo: Array | float) -> Array:
    """``max(x, lo)`` written with ``where`` so that the gradient at a tie goes to ``x`` only.

    ``jnp.maximum`` splits the cotangent 50/50 at ``x == lo``, which would halve
    ``d theta/dh`` exactly at ``h = -hb``; ``where`` keeps the derivative of the
    selected branch.
    """
    return jnp.where(x >= lo, x, lo)


def _clamp(x: Array, lo: Array | float, hi: Array | float) -> Array:
    return jnp.where(x >= lo, jnp.where(x <= hi, x, hi), lo)


def _beta(p: SoilHydraulicParams) -> Array:
    """``B = (theta_s - theta_r - a1 hb) hb**lambda`` (``BCBETA``)."""
    return (p.theta_s - p.theta_r - p.a1 * p.hb) * p.hb**p.lambda_


def _prepare(h: Any, p: SoilHydraulicParams) -> tuple[Array, SoilHydraulicParams]:
    return jnp.asarray(h, dtype=jnp.result_type(float)), p.at_nodes()


# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------


def theta_of_h(h: Any, params: AnyHydraulicParams) -> Array:
    """Volumetric water content ``theta(h)`` [cm3 cm-3] (RZWQM ``WC``).

    Vectorised over the shape of ``h``; ``params`` is gathered per node when it
    carries ``node_horizon``.  Both ``where`` branches are finite for every
    finite ``h``.
    """
    if isinstance(params, TilledSoilHydraulicParams):
        return theta_of_h_tilled(h, params)
    h, p = _prepare(h, params)
    absh = _clamp_min(-h, p.hb)  # BC branch argument: |h| >= hb, so the power is bounded by hb**-lambda
    theta_bc = p.theta_r + _beta(p) * absh ** (-p.lambda_)
    theta_lin = p.theta_s + p.a1 * _clamp(h, -p.hb, 0.0)
    theta = jnp.where(h > -p.hb, theta_lin, theta_bc)
    return jnp.where(h >= 0.0, p.theta_s, theta)


def c_of_h(h: Any, params: AnyHydraulicParams) -> Array:
    """Specific moisture capacity ``C(h) = d theta / dh`` [cm-1] (RZWQM ``SPMOIS``).

    Piecewise: ``0`` for ``h >= 0``, ``a1`` on the linear segment, and
    ``B lambda |h|**(-lambda-1)`` for ``h <= -hb``; it is the exact derivative
    of :func:`theta_of_h` on every segment, including the branch chosen at the
    two joints (``h = -hb`` belongs to the Brooks-Corey segment, ``h = 0`` to
    the saturated one).
    """
    if isinstance(params, TilledSoilHydraulicParams):
        return c_of_h_tilled(h, params)
    h, p = _prepare(h, params)
    absh = _clamp_min(-h, p.hb)
    c_bc = _beta(p) * p.lambda_ * absh ** (-p.lambda_ - 1.0)
    c = jnp.where(h > -p.hb, p.a1, c_bc)
    return jnp.where(h >= 0.0, 0.0, c)


def h_of_theta(theta: Any, params: AnyHydraulicParams) -> Array:
    """Matric potential ``h(theta)`` [cm] (RZWQM ``WCH``), the inverse of :func:`theta_of_h`.

    * ``theta >= theta_s``: ``0`` (RZWQM keeps the old head there; saturation
      is the only value both agree on).
    * ``theta_s - a1 hb <= theta < theta_s``: the linear segment (empty when ``a1 <= A1_MIN``).
    * below: ``-hb Se**(-1/lambda)``, with ``Se`` floored so that ``h >= H_MIN`` (``-1e30`` cm).

    ``theta`` below ``theta_r`` is treated as ``theta_r`` (returns ``H_MIN``).
    The floor is reached before ``theta_r`` when lambda is small: on 6 of the
    105 RZWQM_sw_batch horizons (lambda 0.101-0.127) the exact inverse of
    ``theta_r + 1e-4`` is below ``-1e30`` cm (``-6.6e37`` cm for lambda = 0.101)
    and ``H_MIN`` is returned for theta up to ``theta_r + 6.2e-4``.  The band is
    far outside the model range: ``theta(H_CLAMP_RZWQM)`` exceeds it by at least
    0.0117 on every horizon.
    """
    if isinstance(params, TilledSoilHydraulicParams):
        return h_of_theta_tilled(theta, params)
    theta, p = _prepare(theta, params)
    span = p.theta_s - p.theta_r - p.a1 * p.hb  # theta range of the BC segment, > 0 for valid parameters
    se = (theta - p.theta_r) / _clamp_min(span, _SE_FLOOR)
    # Brooks-Corey inverse in log space, |h| = hb * Se**(-1/lambda), floored at H_MIN on the log scale:
    # every intermediate derivative is then bounded by |H_MIN| / (lambda * Se), whereas the power form
    # builds (|H_MIN| / hb)**(1 + 1/lambda) in its chain rule, which overflows float32 (inf gradient).
    log_abs_h = jnp.log(p.hb) - jnp.log(_clamp(se, _SE_FLOOR, 1.0)) / p.lambda_
    h_bc = -jnp.exp(jnp.where(log_abs_h <= _LOG_H_MAX, log_abs_h, _LOG_H_MAX))
    # a1 below A1_MIN is treated as 0 (no linear segment) so that the unselected branch stays finite:
    # with a1 ~ 1e-200 the division would overflow and poison the where gradient with 0 * inf.
    has_lin = p.a1 > A1_MIN
    a1_safe = jnp.where(has_lin, p.a1, 1.0)
    h_lin = -(p.theta_s - theta) / a1_safe
    h = jnp.where(theta >= p.theta_s - jnp.where(has_lin, p.a1, 0.0) * p.hb, h_lin, h_bc)
    return jnp.where(theta >= p.theta_s, 0.0, h)


def c2_of_params(params: SoilHydraulicParams) -> Array:
    """``C2 = ksat * hb_k**(eps - n1)``: the second K(h) intercept RZWQM derives at start-up (SOILPR l. 4632).

    This is the value :func:`k_of_h` uses; it makes ``K`` continuous at ``-hb_k``
    whatever ``ksat`` is, which the stored ``c2`` field would not.
    """
    return params.ksat * params.hb_k ** (params.eps - params.n1)


def k_of_h(h: Any, params: AnyHydraulicParams) -> Array:
    """Hydraulic conductivity ``K(h)`` [cm hr-1] (RZWQM ``POINTK``).

    ``ksat`` for ``h >= 0``; ``ksat |h|**(-n1)`` on ``[-hb_k, 0)``; and
    ``C2 |h|**(-eps)`` below ``-hb_k`` with ``C2`` from :func:`c2_of_params`.
    The first segment's argument is floored at ``1e-6`` cm so that it stays
    finite for ``n1 > 0`` as ``h -> 0-`` (RZWQM does not guard this; its
    reference classes all have ``n1 = 0`` so the segment is flat).
    """
    if isinstance(params, TilledSoilHydraulicParams):
        return k_of_h_tilled(h, params)
    h, p = _prepare(h, params)
    absh_dry = _clamp_min(-h, p.hb_k)
    k_dry = c2_of_params(p) * absh_dry ** (-p.eps)
    absh_wet = _clamp(-h, _ABSH_WET_FLOOR, p.hb_k)
    k_wet = p.ksat * absh_wet ** (-p.n1)
    k = jnp.where(h >= -p.hb_k, k_wet, k_dry)
    return jnp.where(h >= 0.0, p.ksat, k)


# ---------------------------------------------------------------------------
# post-tillage two-segment curves (RZWQM2 WC / SPMOIS / WCH / POINTK with TRHYDP)
# ---------------------------------------------------------------------------


def _prepare_tilled(
    x: Any, params: TilledSoilHydraulicParams
) -> tuple[Array, SoilHydraulicParams, SoilHydraulicParams]:
    p = params.at_nodes()
    return jnp.asarray(x, dtype=jnp.result_type(float)), p.current, p.original


def theta_of_h_tilled(
    h: Any, params: TilledSoilHydraulicParams, coefficients: HydraulicsCoefficients = RZWQM_HYDRAULICS
) -> Array:
    """Post-tillage ``theta(h)`` (RZWQM ``WC`` with ``TRHYDP``, ``Rzrich.for`` 2033-2037).

    The current curve (:func:`theta_of_h` of ``params.current``, bit for bit) for
    ``h >= -f hb_current``; the original Brooks-Corey segment below, ``f`` =
    ``coefficients.tillage_split_retention``. The original branch's argument is clamped to
    ``|h| >= f hb_current`` so that it is finite wherever it is not selected.
    """
    h, c, o = _prepare_tilled(h, params)
    split = coefficients.tillage_split_retention * c.hb
    absh = _clamp_min(-h, split)
    theta_orig = o.theta_r + _beta(o) * absh ** (-o.lambda_)
    return jnp.where(h < -split, theta_orig, theta_of_h(h, c))


def c_of_h_tilled(
    h: Any, params: TilledSoilHydraulicParams, coefficients: HydraulicsCoefficients = RZWQM_HYDRAULICS
) -> Array:
    """Post-tillage ``C(h)`` (RZWQM ``SPMOIS`` with ``TRHYDP``, ``Rzrich.for`` 1954-1958).

    The derivative of :func:`theta_of_h_tilled` on each segment (the jump at ``-f hb`` has
    no derivative; RZWQM ignores it the same way).
    """
    h, c, o = _prepare_tilled(h, params)
    split = coefficients.tillage_split_retention * c.hb
    absh = _clamp_min(-h, split)
    c_orig = _beta(o) * o.lambda_ * absh ** (-o.lambda_ - 1.0)
    return jnp.where(h < -split, c_orig, c_of_h(h, c))


def h_of_theta_tilled(
    theta: Any, params: TilledSoilHydraulicParams, coefficients: HydraulicsCoefficients = RZWQM_HYDRAULICS
) -> Array:
    """Post-tillage ``h(theta)`` (RZWQM ``WCH`` with ``TRHYDP``/``WC10S2``, ``Rzrich.for`` 2105, 2123-2128).

    The original inverse where ``theta`` is at or below ``theta_o(-f hb_o)`` (the water
    content of the *original* curve at the *original* split head, ``WC10S2``) and below the
    linear segment of the current curve; :func:`h_of_theta` of ``params.current`` elsewhere.
    """
    theta, c, o = _prepare_tilled(theta, params)
    theta_split = o.theta_r + _beta(o) * (coefficients.tillage_split_retention * o.hb) ** (-o.lambda_)
    below_lin = theta < c.theta_s - c.a1 * c.hb
    return jnp.where((theta <= theta_split) & below_lin, h_of_theta(theta, o), h_of_theta(theta, c))


def k_of_h_tilled(
    h: Any, params: TilledSoilHydraulicParams, coefficients: HydraulicsCoefficients = RZWQM_HYDRAULICS
) -> Array:
    """Post-tillage ``K(h)`` (RZWQM ``POINTK`` with ``C22``/``SN22``, ``Rzrich.for`` 860-864, 875-876).

    ``C2_o |h|**(-eps_o)`` at and below ``-f hb_k,current`` (``C2_o`` from
    :func:`c2_of_params` of the original parameters, the value ``SOILPR`` stored at
    start-up), :func:`k_of_h` of ``params.current`` above; ``f`` =
    ``coefficients.tillage_split_conductivity``.
    """
    h, c, o = _prepare_tilled(h, params)
    split = coefficients.tillage_split_conductivity * c.hb_k
    absh = _clamp_min(-h, split)
    k_orig = c2_of_params(o) * absh ** (-o.eps)
    return jnp.where(h <= -split, k_orig, k_of_h(h, c))


def derive_rzwqm(
    params: SoilHydraulicParams, coefficients: HydraulicsCoefficients = RZWQM_HYDRAULICS
) -> SoilHydraulicParams:
    """Recompute ``fc13 fc110 wp c2`` from the primary parameters as ``SOILPR`` does at start-up.

    ``fc13 = theta(-333)``, ``fc110 = theta(-100)``, ``wp = theta(-15000)``
    (RZTEST.for lines 4609-4625 and 4858; the ``hb >= 333`` case falls on the
    linear segment there as here) and ``c2 = ksat hb_k**(eps - n1)`` (line 4632).
    The heads are :class:`HydraulicsCoefficients` (default :data:`RZWQM_HYDRAULICS`).
    The node map is untouched.
    """
    c = coefficients
    hp = params.replace(node_horizon=None)
    shape = jnp.shape(hp.hb)
    return params.replace(
        fc13=theta_of_h(jnp.full(shape, c.h_fc13), hp),
        fc110=theta_of_h(jnp.full(shape, c.h_fc110), hp),
        wp=theta_of_h(jnp.full(shape, c.h_wp), hp),
        c2=c2_of_params(hp),
    )
