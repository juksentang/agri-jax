"""The config of the adaptive Richards integrator: :class:`AdaptiveStepping`, its registry key and presets.

Part of :mod:`~agrijax.processes.soil_water.richards_adaptive` (split out as a pure move); the step
control, the tolerances and the budgets are described in that module's docstring.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agrijax.core.coefficients import Provenance

from .coefficients import numerical_setting, rzwqm2, setting_field
from .integrator import SteppingConfig

#: key of the adaptive integrator (RZWQM2's time integration)
KEY = "soil_water/richards_time@rzwqm2-4.6:faithful"
#: largest adaptive sub-step of the fast tier [h] (the exact tier is AdaptiveStepping.dt_max = 0.1 h)
DT_MAX_FAST: float = numerical_setting(
    "richards.dt_max_fast",
    0.25,
    "h",
    "largest adaptive sub-step of the fast tier (AdaptiveStepping.fast())",
    origin="agrijax",
    basis="measured against the converged reference (Crank-Nicolson with the alpha = 1 fallback, 960 "
    "sub-steps a day): at dt_max 0.25 h the worst numerical error is 7.9e-3 cm over the "
    "58 site-years with a clean reference, within 1e-3 cm on 44 of the 50 outside CA-TPA "
    "(the RZWQM2 4.6 replays, data tier outside this repository)",
)


class AdaptiveStepping(SteppingConfig):
    """Config of :class:`~agrijax.processes.soil_water.richards_adaptive.AdaptiveCN` (static settings;
    hashable, part of the compiled program).

    ``time_scheme`` ``"rzwqm"`` (``alpha = 1`` on a segment's first step, then 1/2, the default) or
    ``"implicit"``; the damping of the Newton iteration (``h_upper``, ``dv_max``, ``c_floor``,
    ``chop``) as :class:`~agrijax.processes.soil_water.fixed_cn.FixedStepping`; the step control,
    the tolerances and the budgets of the
    :mod:`~agrijax.processes.soil_water.richards_adaptive` docstring. Newton always uses the exact Jacobian
    and every solve has an implicit-function VJP. Presets: :meth:`exact` (``dt_max`` 0.1 h, RZWQM2's
    ``PDTMAX``; the defaults) and :meth:`fast` (:data:`DT_MAX_FAST`, 0.25 h; worst numerical error 7.9e-3 cm
    over the 58 site-years on which the converged reference has no unconverged step).
    """

    KEY: ClassVar[str] = KEY

    time_scheme: str = setting_field(
        "richards.adaptive_time_scheme",
        "rzwqm",
        "-",
        "time weights of the adaptive steps: 'rzwqm' (alpha = 1 on a segment's first step and on a "
        "retry, 1/2 after: richards.alpha_cn) or 'implicit' (alpha = 1 throughout)",
        origin="agrijax",
        basis="measured against the converged reference: Crank-Nicolson with the alpha = 1 fallback is "
        "one to two orders of magnitude more accurate than fully implicit at the same cost "
        "(convergence order about 2.2 against 1)",
    )
    h_upper: float = setting_field(
        "richards.h_upper",
        10.0,
        "cm",
        "upper clamp of the head iterate above hydrostatic (node i clamped at h_upper + z_i); a "
        "divergence guard that should never activate (n_clamp)",
        origin="agrijax",
        basis="+10 cm above hydrostatic is a divergence guard of this implementation; RZWQM2 caps the "
        "surface head at HMAX = 0",
    )
    dv_max: float = setting_field(
        "richards.dv_max",
        1.0,
        "-",
        "largest Newton update of the transformed variable v per node (a factor e in |h|)",
        origin="agrijax",
        basis="richards.py module docstring (damping); tests/unit/test_richards.py",
    )
    c_floor: float = setting_field(
        "richards.c_floor",
        1.0e-7,
        "cm-1",
        "storage floor added to the Jacobian diagonal only (never the residual): keeps it "
        "non-singular on a saturated profile",
        origin="agrijax",
        basis="added to the Jacobian only because C(h) must stay the exact derivative of theta(h): the "
        "floor changes the iteration path, not the residual or the root",
    )
    chop: bool = setting_field(
        "richards.chop",
        True,
        "-",
        "stop an update that leaves the saturated side across the air-entry kink h = -hb on the kink",
        origin="agrijax",
        provenance=Provenance("none", paper="Wang & Tchelepi (2013), J. Comput. Phys. 253, 114-137"),
        basis="tests/unit/test_richards.py pond-emptying sub-step (measured with and without)",
    )
    dt_max: float = setting_field(
        "richards.dt_max",
        0.1,
        "h",
        "largest adaptive sub-step (the exact tier; the fast tier is richards.dt_max_fast); the last "
        "step before a breakpoint may be stretched to breakpoint_stretch times it",
        origin="rzwqm2-4.6",
        provenance=rzwqm2("RZWQM/Rzday.for:52", "ADJDT", note="PDTMAX; the step is limited at Rzday.for:119"),
    )
    dt_min: float = setting_field(
        "richards.dt_min",
        1.0e-4,
        "h",
        "smallest adaptive sub-step; a step that fails at dt_min with alpha = 1 is accepted unconverged "
        "and counted (n_unconverged)",
        origin="rzwqm2-4.6",
        provenance=rzwqm2("RZWQM/Rzday.for:52", "ADJDT", note="PDTMIN; the step is limited at Rzday.for:120"),
    )
    dt_reset: float = setting_field(
        "richards.dt_reset",
        1.0e-4,
        "h",
        "sub-step the adaptive stepping restarts from at the onset of surface supply and after an "
        "infiltration event",
        origin="rzwqm2-4.6",
        provenance=rzwqm2(
            "RZWQM/Rzday.for:52",
            "ADJDT",
            note="PDTMIN; ADJDT restarts from it on the first call and after an event (Rzday.for:57-66, "
            "FIRST5 set at Rzday.for:1807)",
        ),
    )
    dt_grow: float = setting_field(
        "richards.dt_grow",
        1.5,
        "-",
        "growth factor of the sub-step after a step that converged in at most iter_grow_max updates",
        origin="rzwqm2-4.6",
        provenance=rzwqm2("RZWQM/Rzday.for:75", "ADJDT", note="growth limited to 1.5 times the last step"),
    )
    dt_shrink_fail: float = setting_field(
        "richards.dt_shrink_fail",
        0.5,
        "-",
        "factor of the sub-step after a failed Newton solve at alpha = 1 (the step is redone from its "
        "initial state)",
        origin="rzwqm2-4.6",
        provenance=rzwqm2("RZWQM/Rzrich.for:542", "CNHEAD", note="the step is halved and restarted"),
    )
    dt_shrink_slow: float = setting_field(
        "richards.dt_shrink_slow",
        0.7,
        "-",
        "factor of the sub-step after a step that needed at least iter_shrink_min Newton updates",
        origin="agrijax",
        provenance=Provenance("none", paper="Simunek et al., HYDRUS-1D manual (dMul2)"),
        basis="step-size rule after HYDRUS-1D, used in the adaptive runs compared with the converged "
        "reference (the RZWQM2 4.6 replays, data tier outside this repository)",
    )
    iter_grow_max: int = setting_field(
        "richards.iter_grow_max",
        3,
        "-",
        "a converged step with at most this many Newton updates lets the next step grow (dt_grow)",
        origin="agrijax",
        provenance=Provenance("none", paper="Simunek et al., HYDRUS-1D manual (ItMin)"),
        basis="step-size rule after HYDRUS-1D, used in the adaptive runs compared with the converged "
        "reference (the RZWQM2 4.6 replays, data tier outside this repository)",
    )
    iter_shrink_min: int = setting_field(
        "richards.iter_shrink_min",
        7,
        "-",
        "a converged step with at least this many Newton updates shrinks the next step (dt_shrink_slow)",
        origin="agrijax",
        provenance=Provenance("none", paper="Simunek et al., HYDRUS-1D manual (ItMax)"),
        basis="step-size rule after HYDRUS-1D, used in the adaptive runs compared with the converged "
        "reference (the RZWQM2 4.6 replays, data tier outside this repository)",
    )
    cn_fallback: bool = setting_field(
        "richards.cn_fallback",
        True,
        "-",
        "a Crank-Nicolson step whose Newton solve fails is retried at the same dt with alpha = 1 "
        "before the step is halved",
        origin="agrijax",
        basis="measured: plain Crank-Nicolson fails (Newton does not converge) at ponding switches and "
        "at very dry nodes; the retry with alpha = 1 at the same dt saves the halving",
    )
    newton_max_iter: int = setting_field(
        "richards.newton_max_iter",
        10,
        "-",
        "Newton evaluations per adaptive sub-step (the polish update included)",
        origin="agrijax",
        basis="a choice of this implementation: the 8 iterations of the fixed 96 x 8 configuration left "
        "one day unconverged in each of two CA-TPA years (2016, 2020); tests/unit/test_richards_adaptive.py",
    )
    newton_tol_theta: float = setting_field(
        "richards.newton_tol_theta",
        1.0e-10,
        "cm3 cm-3",
        "convergence: max_i |R_i| dt / tl_i (node water-content residual), float64",
        origin="agrijax",
        basis="four orders above the float64 rounding floor (about 1e-14), so that a residual that stops "
        "at the floor is not taken for non-convergence; Newton is quadratic, so a solve that passes is "
        "usually far below it",
    )
    newton_tol_theta_f32: float = setting_field(
        "richards.newton_tol_theta_f32",
        2.0e-6,
        "cm3 cm-3",
        "convergence: node water-content residual, float32 (about 20 rounding units of theta ~ 0.3)",
        origin="agrijax",
        basis="float32 noise floor: the residual contains two subtractions of theta ~ 0.3 (rounding "
        "unit about 3e-8), noise about 1e-7; the tolerance is about 20 times that",
    )
    newton_tol_balance: float = setting_field(
        "richards.newton_tol_balance",
        1.0e-10,
        "cm",
        "convergence: |dt sum_i R_i| (the sub-step water balance), float64",
        origin="agrijax",
        basis="four orders above the float64 rounding floor (about 1e-14), so that a residual that stops "
        "at the floor is not taken for non-convergence; Newton is quadratic, so a solve that passes is "
        "usually far below it",
    )
    newton_tol_balance_f32: float = setting_field(
        "richards.newton_tol_balance_f32",
        3.0e-5,
        "cm",
        "convergence: sub-step water balance, float32 (about 4 times the rounding of a 150 cm profile)",
        origin="agrijax",
        basis="float32 noise floor: the rounding noise of the sum over a 150 cm profile is about "
        "7.5e-6 cm; the tolerance is about 4 times that",
    )
    newton_polish: bool = setting_field(
        "richards.newton_polish",
        True,
        "-",
        "apply one more Newton update after the convergence test passes (balance to rounding)",
        origin="agrijax",
        basis="one more update costs one more evaluation a step and brings the sub-step balance to "
        "rounding level (the RZWQM2 4.6 replays, data tier outside this repository)",
    )
    bc_switch: bool = setting_field(
        "richards.bc_switch",
        True,
        "-",
        "adaptive mode: a try whose Newton solve with the clipped surface flux fails is solved again "
        "with the surface condition fixed, ponded (ghost head 0), then flux (the requested flux); "
        "the first that converges to a solution of the clipped problem (the same surface flux) is "
        "taken before the step is retried or halved",
        origin="agrijax",
        provenance=rzwqm2(
            "RZWQM/Rzrich.for:59",
            "CHKBC",
            note="Rzrich.for:59-86: RZWQM2 chooses the flux or head condition outside the iteration and "
            "restarts on a switch; here the condition is fixed only on a retry and taken only when "
            "consistent",
        ),
        basis="measured on the US_OPE storms: Newton 2-cycles between the flux branch and the floored "
        "ponded branch of the clipped surface flux at the ponding switches; the switch stays inside the "
        "residual and a fixed condition is tried only on a failed solve",
    )
    newton_div_after: int = setting_field(
        "richards.newton_div_after",
        3,
        "-",
        "from this Newton evaluation on, a residual that rose newton_div_patience times in a row fails "
        "the solve early",
        origin="agrijax",
        basis="divergence test of the adaptive solve: a solve that diverges is ended early and its "
        "remaining evaluations go to the halved step",
    )
    newton_div_patience: int = setting_field(
        "richards.newton_div_patience",
        2,
        "-",
        "consecutive rises of the node residual that count as divergence",
        origin="agrijax",
        basis="divergence test of the adaptive solve (see newton_div_after)",
    )
    newton_max_iter_last: int = setting_field(
        "richards.newton_max_iter_last",
        30,
        "-",
        "adaptive mode: Newton evaluations of the last-resort solve (dt_min, alpha = 1), whose "
        "failure is accepted unconverged",
        origin="agrijax",
        basis="measured on the US_OPE storms: the ponding-switch steps that stayed unconverged at dt_min "
        "with newton_max_iter = 10 converge in 11-12 evaluations",
    )
    dry_guard: float = setting_field(
        "richards.dry_guard",
        10.0,
        "-",
        "adaptive mode: the head iterate is bounded below at dry_guard * h_min (a divergence guard "
        "that should never activate, n_active), not at h_min: a node at h_min that still loses water "
        "(the gravity flux K(h_min) to the node below, the explicit half of Crank-Nicolson) drains "
        "below h_min, so every accepted step conserves water; RZWQM2 clamps at Hmin",
        origin="agrijax",
        basis="measured: the nodes held at h_min as an active set left the gravity outflow K(h_min) dt "
        "(up to 3e-8 cm a step) as balance error",
    )
    line_search: tuple[float, ...] = setting_field(
        "richards.line_search",
        (1.0, 0.5, 0.25, 0.125),
        "-",
        "backtracking factors of the Newton update: the first that lowers sum((R dt / tl)^2) is taken, "
        "else the last (adaptive mode, fix N2)",
        origin="agrijax",
        basis="fix N2 of the richards_adaptive.py module docstring: breaks the 2-cycles measured at "
        "saturated and dry fronts",
    )
    breakpoint_stretch: float = setting_field(
        "richards.breakpoint_stretch",
        1.25,
        "-",
        "a remaining time to the next breakpoint of at most this many dt is taken in one step",
        origin="agrijax",
        basis="breakpoint rule of the step control (richards_adaptive.py module docstring): it avoids a "
        "very short remainder step before a breakpoint",
    )
    breakpoint_split: float = setting_field(
        "richards.breakpoint_split",
        2.0,
        "-",
        "a remaining time of at most this many dt is split into this many equal steps",
        origin="agrijax",
        basis="breakpoint rule of the step control (richards_adaptive.py module docstring): it avoids a "
        "very short remainder step before a breakpoint",
    )
    max_steps: int = setting_field(
        "richards.max_steps",
        1000,
        "-",
        "accepted adaptive sub-steps per segment (the length of the step table); the rest of a "
        "segment that exhausts the budget is one unconverged step (budget_exhausted)",
        origin="agrijax",
        basis="a day at dt_max = 0.1 h needs at least 240 accepted steps; the budget bounds the step table",
    )
    max_trips: int = setting_field(
        "richards.max_trips",
        4000,
        "-",
        "Newton solves (accepted and rejected) per segment in the step search",
        origin="agrijax",
        basis="a hard bound on the trips of the step-search loop; rejected solves count as trips, so it "
        "exceeds max_steps",
    )

    @classmethod
    def exact(cls, **overrides: Any) -> AdaptiveStepping:
        """The exact tier: ``dt_max`` 0.1 h (the defaults)."""
        return cls(**overrides)

    @classmethod
    def fast(cls, **overrides: Any) -> AdaptiveStepping:
        """The fast tier: ``dt_max`` = :data:`DT_MAX_FAST` (0.25 h)."""
        kw: dict[str, Any] = {"dt_max": DT_MAX_FAST}
        return cls(**(kw | overrides))

    @classmethod
    def tier(cls, tier: str = "exact", **overrides: Any) -> AdaptiveStepping:
        """:meth:`exact` or :meth:`fast` by name."""
        if tier not in ("exact", "fast"):
            raise ValueError(f"tier must be 'exact' or 'fast', got {tier!r}")
        return cls.fast(**overrides) if tier == "fast" else cls.exact(**overrides)

    def __check_init__(self) -> None:
        if self.time_scheme not in ("rzwqm", "implicit"):
            raise ValueError(f"time_scheme must be 'rzwqm' or 'implicit', got {self.time_scheme!r}")
        if self.dv_max <= 0.0 or self.c_floor < 0.0:
            raise ValueError("dv_max > 0 and c_floor >= 0 are required")
        if not 0.0 < self.dt_min <= self.dt_reset <= self.dt_max:
            raise ValueError("adaptive stepping needs 0 < dt_min <= dt_reset <= dt_max")
        if self.newton_max_iter_last < self.newton_max_iter:
            raise ValueError("newton_max_iter_last must be >= newton_max_iter")
        if self.newton_max_iter <= 1 or self.max_steps < 1 or self.max_trips < 1:
            raise ValueError("newton_max_iter > 1, max_steps >= 1 and max_trips >= 1 are required")
        if self.dry_guard < 1.0:
            raise ValueError("dry_guard must be >= 1 (the guard is at dry_guard * h_min)")
        if not self.line_search or not 1.0 <= self.breakpoint_stretch <= self.breakpoint_split:
            raise ValueError("line_search must be non-empty and 1 <= breakpoint_stretch <= breakpoint_split")
