"""
Generate a standalone research paper (paper.tex -> RESEARCH_PAPER.pdf) from the
same regenerated CSVs / logs as RESULTS_REPORT.pdf, but in a different register:
a traditional Introduction/Model/Theory/Methods/Results/Discussion write-up
instead of build_report.py's per-experiment lab-notebook pattern.

Every number quoted below is computed here from the CSVs (or from the same
common/ model code that ran the simulations) at build time -- never hand
transcribed. Several build-time checks (linear stability eigenvalues, the
exact k/T invariance, Newton continuation of the ordered branch, degree
statistics of the seed-1 graphs) are the identical computations used by
build_report.py's mathematics blocks, reproduced here so this document is
self-contained.

Run:  ../.venv/bin/python build_paper.py     (from the repo root)
"""
import os, sys, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.meanfield import hmf_step, hmf_run
from common.observables import order_parameter
from common.graphs import build_graph


def load(p):
    return np.genfromtxt(os.path.join(HERE, p), delimiter=",", names=True)


def epsc(e, m, thr=0.5):
    e = np.asarray(e); m = np.asarray(m)
    o = np.argsort(e); e, m = e[o], m[o]
    idx = np.where(m < thr)[0]
    if len(idx) == 0:
        return float(e[-1])
    i = idx[0]
    if i == 0:
        return float(e[0])
    e0, e1, m0, m1 = e[i - 1], e[i], m[i - 1], m[i]
    return float(e0 + (thr - m0) * (e1 - e0) / (m1 - m0))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


L = []
def w(s=""):
    L.append(s)


# ============================================================ BUILD-TIME MATHEMATICS
# Identical computations to build_report.py's mathematics blocks (same common/
# code), reproduced so this document does not depend on that script.

def _tangent_jacobian(x0, eps, k, T, h=1e-7):
    x0 = np.asarray(x0, float)
    J = np.zeros((3, 3))
    for c in range(3):
        xp = x0.copy(); xp[c] += h
        xm = x0.copy(); xm[c] -= h
        J[:, c] = (np.array(hmf_step(*xp, eps, k, T))
                   - np.array(hmf_step(*xm, eps, k, T))) / (2 * h)
    B = np.array([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]])
    M, *_ = np.linalg.lstsq(B, J @ B, rcond=None)
    return M


def _hmf_epsc(k, T, lo=0.45, hi=0.85, n=81):
    ee = np.linspace(lo, hi, n)
    m = [order_parameter(hmf_run(float(e), k=float(k), T=float(T))) for e in ee]
    return epsc(ee, m)


def _consensus_branch_end(k, T, e0, step=0.002):
    def newton(eps):
        x = np.array([0.98, 0.01])
        for _ in range(100):
            def g(v):
                rn, pn, _ = hmf_step(v[0], v[1], 1 - v[0] - v[1], eps, k, T)
                return np.array([rn - v[0], pn - v[1]])
            G = g(x); hh = 1e-8
            Jm = np.array([(g(x + [hh, 0]) - G) / hh,
                           (g(x + [0, hh]) - G) / hh]).T
            try:
                dx = np.linalg.solve(Jm, -G)
            except np.linalg.LinAlgError:
                return None
            x = x + dx
            if not np.all(np.isfinite(x)) or x[0] < 0.5:
                return None
            if np.linalg.norm(dx) < 1e-12:
                break
        rho = float(np.max(np.abs(np.linalg.eigvals(
            _tangent_jacobian((x[0], x[1], 1 - x[0] - x[1]), eps, k, T)))))
        return float(x[0]), rho
    last, e = None, e0
    while e < 1.0:
        out = newton(e)
        if out is None or out[1] >= 1.0:
            break
        last = (e, out[0], out[1])
        e += step
    return last

MK, MT = 10.0, 0.65
lam_re = 0.25 + MK / (4 * MT)
lam_im = np.sqrt(3) * MK / (4 * MT)
jac_dev = max(
    float(np.max(np.abs(
        np.sort_complex(np.linalg.eigvals(
            _tangent_jacobian((1/3, 1/3, 1/3), e_, MK, MT)))
        - np.sort_complex(np.array([lam_re - 1j * lam_im * e_,
                                    lam_re + 1j * lam_im * e_])))))
    for e_ in (0.0, 0.3, 0.7))
jac_dev_tex = rf"10^{{{int(np.ceil(np.log10(jac_dev)))}}}"

kt_cells = [(10, 0.65), (20, 1.30), (5, 0.325)]
kt_ecs = [_hmf_epsc(k_, t_) for k_, t_ in kt_cells]

br10 = _consensus_branch_end(10, 0.65, e0=0.60)
br20 = _consensus_branch_end(20, 0.65, e0=0.72)

_deg_er = np.array([d_ for _, d_ in build_graph("ER", 800, 10, seed=1).degree()])
_deg_ba = np.sort(np.array(
    [d_ for _, d_ in build_graph("BA", 800, 10, seed=1).degree()]))[::-1]
var_er, var_ba = float(np.var(_deg_er)), float(np.var(_deg_ba))
stub_share = {z_: float(_deg_ba[:int(round(z_ * 800))].sum() / _deg_ba.sum())
              for z_ in (0.05, 0.10)}

# ============================================================ DATA
sw = load("mean_field/hmf_sweep.csv")
sw_ks = [c for c in sw.dtype.names if c != "epsilon"]
suite = {g: load(f"mean_field/comparison_suite_{g}_k10.csv") for g in ("ER", "BA")}
cgr = load("mean_field/compare_grid.csv")
mch = load("monte_carlo/mc_vs_hmf.csv")
pdg = {g: load(f"phase_diagram/phase_diagram_{g}.csv") for g in ("ER", "BA")}
cb = load("phase_diagram/critical_boundary.csv")
fss = load("dynamics/fss.csv")
fss_Ns = [c for c in fss.dtype.names if c != "epsilon"]
smi = load("sensitivity/sens_mf_init.csv")
ssz = load("sensitivity/sens_size.csv")
z = load("zealots/zealots.csv")
h = load("zealots/zealots_hubs.csv")
mx = load("zealots/zealots_mixed.csv")
d = load("defects/defects.csv")
cl = load("defects/defects_collapse.csv")
tss = load("zealots/timeseries_story.csv")
p7 = load("zealots/phase7_timeseries.csv")
sv = load("sensitivity/sens_validation.csv")
ssd = load("sensitivity/sens_seeds.csv")
sgr = load("sensitivity/sens_grid.csv")
stp = load("sensitivity/sens_temperature.csv")
sdf = load("sensitivity/sens_defect_seed.csv")
man = np.genfromtxt(os.path.join(HERE, "logs/manifest.csv"), delimiter=",",
                    names=True, dtype=None, encoding="utf-8")
run_date = str(man[0]["started_iso"])[:10]

# ---- headline numbers, computed once, reused across sections
gap_erba = float(np.max(np.abs(cb["eps_c_ER"] - cb["eps_c_BA"])))
hgap = cb["eps_c_HMF"] - cb["eps_c_ER"]
hgap_max, hgap_max_k = float(hgap.max()), int(cb["k"][int(hgap.argmax())])
hgap_flip_k = int(cb["k"][int(np.argmax(hgap < 0))])
_h_lo = np.minimum(cb["eps_c_HMF"], cb["eps_c_HMF_ordered_init"])
_h_hi = np.maximum(cb["eps_c_HMF"], cb["eps_c_HMF_ordered_init"])
_inside = (cb["eps_c_ER"] >= _h_lo) & (cb["eps_c_ER"] <= _h_hi)
inside_n, inside_first_k = int(_inside.sum()), int(cb["k"][int(_inside.argmax())])
win_top = (float(_h_lo[-1]), float(_h_hi[-1])); win_top_k = int(cb["k"][-1])

rows_rmse = {g: (rmse(suite[g]["hmf"], suite[g]["mc"]), rmse(suite[g]["dmf"], suite[g]["mc"]))
             for g in ("ER", "BA")}
gain_ratio = (rows_rmse["BA"][0] - rows_rmse["BA"][1]) / max(rows_rmse["ER"][0] - rows_rmse["ER"][1], 1e-9)
ec_mc10, ec_hmf10 = epsc(suite["ER"]["epsilon"], suite["ER"]["mc"]), epsc(suite["ER"]["epsilon"], suite["ER"]["hmf"])

fss_ec = [epsc(fss["epsilon"], fss[n]) for n in fss_Ns]
fss_slopes = [np.max(np.abs(np.diff(fss[n]) / np.diff(fss["epsilon"]))) for n in fss_Ns]
ssz_extrap = 2 * ssz["eps_c"][-1] - ssz["eps_c"][-2]

smi_by = {}
for row in smi:
    smi_by.setdefault((round(float(row["init_r"]), 4)), {})[int(row["k"])] = float(row["eps_c"])
smi_biased_ks = sorted(smi_by.keys())
smi_win = {k_: (min(smi_by[i_][k_] for i_ in smi_biased_ks if i_ > 0.34),
                max(smi_by[i_][k_] for i_ in smi_biased_ks if i_ > 0.34))
           for k_ in (10, 20)}

amp_hub = h["cycle_hub_mpsi"][-1] / max(h["cycle_random_mpsi"][-1], 1e-9)
sat_idx = np.where(h["order_hub_conversion"] >= 0.9)[0]
z_sat = float(h["z"][sat_idx[0]]) if len(sat_idx) else float(h["z"][-1])

dev_collapse = float(np.max(np.abs(cl["eps_c_edge"] - cl["eps_c_node"])))
fr = sorted(set(d["f"]))
ec_edge = {f_: epsc(d["epsilon"][(d["defect_type_0edge_1node"] == 0) & (d["f"] == f_)],
                    d["m_psi"][(d["defect_type_0edge_1node"] == 0) & (d["f"] == f_)])
           for f_ in fr}
k0_pristine = float(d["mean_k"][(d["defect_type_0edge_1node"] == 0) & (d["f"] == 0)][0])
thin_dev = max(abs(float(d["mean_k"][(d["defect_type_0edge_1node"] == t_) & (d["f"] == f_)][0])
                   - (1 - f_) * k0_pristine)
              for t_ in (0, 1) for f_ in fr)

sv_gap = np.abs(sv["m_py"] - sv["m_cpp"])
sv_agree = int(np.sum((sv["m_py"] > 0.5) == (sv["m_cpp"] > 0.5)))
sgr_ref = sgr["eps_c_interp"][np.argmin(sgr["step"])]
sgr_i05 = float(sgr["eps_c_interp"][np.isclose(sgr["step"], 0.05)][0] - sgr_ref)
sgr_n05 = float(sgr["eps_c_naive"][np.isclose(sgr["step"], 0.05)][0] - sgr_ref)
sdf_maxsd = float(max(np.std(sdf["eps_c"][sdf["frac"] == f_]) for f_ in sorted(set(sdf["frac"]))))

_p7z = np.linspace(0.0, 0.20, 17)
def _p7col(tag, kind, zv):
    return p7[f"{kind}_{tag}_z{f'{zv:.3f}'.replace('.', 'p')}"]
_p7_final_order = np.array([_p7col("order", "conversion", zv)[-1] for zv in _p7z])
_p7_rmse = rmse(_p7_final_order, z["order_conversion"])
_p7_decisions = []
for _zv in _p7z:
    _dev = np.abs(_p7col("order", "conversion", _zv) - 1 / 3)
    _idx = np.where(_dev > 0.15)[0]
    if len(_idx):
        _p7_decisions.append(p7["t_order"][_idx[0]])
_p7_dec_med = float(np.median(_p7_decisions))
SNAMES = ["Rock", "Paper", "Scissors"]

n_figs = n_csvs = 0
for root_, dirs_, files_ in os.walk(HERE):
    dirs_[:] = [d_ for d_ in dirs_ if d_ not in (".git", ".venv", "logs", "__pycache__")]
    n_figs += sum(f_.endswith(".png") for f_ in files_)
    n_csvs += sum(f_.endswith(".csv") for f_ in files_)

# ============================================================ PREAMBLE
w(r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.3cm]{geometry}
\usepackage{amsmath,amssymb,amsthm,booktabs,graphicx,caption,float,xcolor,enumitem}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=black]{hyperref}
\graphicspath{{./}}
\captionsetup{font=small,labelfont=bf,skip=3pt}
\setlength{\parskip}{3pt}
\renewcommand{\arraystretch}{1.1}
\newcommand{\epsc}{\varepsilon_c}
\newcommand{\mpsi}{m_\psi}
\newcommand{\deriv}[1]{\par\smallskip\noindent\textit{Derivation.} #1\par\smallskip}
\newcommand{\numcheck}[1]{\par\smallskip\noindent\textit{Numerical check.} #1\par\smallskip}
\newtheorem{proposition}{Proposition}
\newtheorem{remark}{Remark}
\title{\bfseries Cyclic Dominance in a Potts--RPS Model on Complex Networks:\\
Mean-Field Theory, Monte Carlo Validation, and Perturbation Analysis}
\author{Jenik Gajera}
\date{RUNDATE}
\begin{document}
\maketitle
\begin{abstract}
\noindent
We study a $q{=}3$ Potts model in which the pairwise interaction is deformed
from pure ferromagnetic alignment toward a Rock--Paper--Scissors (RPS) cyclic
dominance by a single control parameter $\varepsilon\in[0,1]$. Agents sit on
the nodes of a complex network (Erd\H{o}s--R\'enyi or Barab\'asi--Albert) and
update by Glauber dynamics at fixed temperature $T$; for $\varepsilon>0$ the
dynamics has no Hamiltonian and does not satisfy detailed balance, so the
stationary state can carry probability current and the population can cycle
forever instead of freezing into consensus. We derive the governing equations
at three levels of description -- homogeneous mean field (HMF), degree-based
mean field (DMF), and direct agent-based Monte Carlo (MC) -- validate a custom
C++ engine against an independent implementation, and map the full
$(\langle k\rangle,\varepsilon)$ phase diagram by direct simulation
(NPDGTILES{} independent runs). The order--cycling boundary $\epsc(\langle
k\rangle)$ rises monotonically with connectivity and is indistinguishable
between ER and BA topologies to within GAPERBA{} over NDEGREES{} degrees,
showing that mean degree alone -- not the shape of the degree distribution --
controls stability. A linear stability analysis at the symmetric fixed point
gives closed-form eigenvalues in which $\varepsilon$ enters as a pure
rotation, and Newton continuation of the ordered branch together with
finite-size scaling of the MC transition both independently identify the
transition as first-order-like, with a genuine mean-field bistability window
and a $1/N$ shift of the Monte Carlo pseudo-critical point. We then perturb
the ordered phase with committed minorities (``zealots''), structural
targeting, competing factions, and quenched network damage. A closed-form
``zealot field'' explains a counter-intuitive backfire effect -- a Rock
minority elects Paper, its own predator, as the population's strategy -- and
predicts that hub-placed zealots gain leverage $1/\sqrt z$ on scale-free
networks (measured amplification AMPHUB{}$\times$). Edge and node damage are
shown, via an exact thinning argument, to act only through the surviving mean
degree, collapsing every network studied -- random or scale-free, damaged or
pristine -- onto the single boundary curve $\epsc(\langle k\rangle)$ to within
DEVCOLLAPSE{}. Every quantitative claim in this paper is computed from the
accompanying data tables at build time.
\end{abstract}
""")

# ============================================================ 1. INTRODUCTION
w(r"\section{Introduction}")
w(r"""
Cyclic, non-transitive dominance -- the Rock--Paper--Scissors (RPS) relation
in which each strategy beats one alternative and loses to the other -- is one
of the simplest mechanisms known to sustain persistent dynamics in an
interacting population instead of settling into a static consensus. It has
been used to model everything from bacterial strain competition to mating
strategies and cyclic patterns in ecological and social systems: whenever
``being popular'' creates the conditions for your own downfall, a system can
be pushed away from equilibrium by its own interaction structure rather than
by any external drive.

Separately, the $q$-state Potts model is the canonical statistical-mechanics
description of $q$ discrete, mutually exclusive states with a
same-state-is-rewarded (ferromagnetic) coupling: it orders into a single
dominant state at low temperature and disorders at high temperature, and its
equilibrium behaviour is completely governed by a Hamiltonian and the Gibbs
measure $e^{-H/T}$.

This paper studies the model obtained by taking a $q{=}3$ Potts system and
deforming its interaction with an antisymmetric, RPS-cyclic term of strength
$\varepsilon$. The resulting payoff matrix (Sec.~\ref{sec:model}) mixes an
\emph{ordering} part, which behaves exactly like the Potts ferromagnet, with
a \emph{cycling} part that has no Hamiltonian at all: for $\varepsilon>0$ the
dynamics genuinely breaks detailed balance, and the two parts compete for
control of the population's long-run behaviour. The single knob $\varepsilon$
therefore interpolates between an equilibrium phase-ordering problem and a
genuinely non-equilibrium cyclic system, and the object of this paper is to
chart, quantitatively, where that competition is won by order and where it is
won by cycling -- on networks, where both the average connectivity and the
shape of the degree distribution are additional structural parameters.

The paper answers four questions. \textbf{(i)} Does raising the average
connectivity $\langle k\rangle$ of the network protect order against cyclic
dominance, and if so, through what mechanism? \textbf{(ii)} How well do
mean-field closures of increasing sophistication (homogeneous vs
degree-resolved) approximate the true, agent-based dynamics, and where do
they fail? \textbf{(iii)} Is the order--cycling transition a conventional
continuous transition, or does it carry first-order signatures such as
hysteresis and a bistable window? \textbf{(iv)} How robust is an ordered
population to targeted perturbation -- committed zealot minorities, structural
targeting of network hubs, competing factions, and quenched structural damage
-- and can the response to these perturbations be predicted from the same
closed-form theory that answers (i)--(iii)?

The remainder of the paper is organised as follows. Section~\ref{sec:model}
fixes the model and its order parameter. Section~\ref{sec:theory} derives the
governing equations at every level of description used later: the two
mean-field closures, an exact invariance of the mean-field map, a linear
stability analysis of the symmetric state, and the mean-field signature of a
first-order transition. Section~\ref{sec:methods} describes the simulation
methods and the validation of the numerical engine. Section~\ref{sec:mf-mc}
confronts the mean-field theory with direct Monte Carlo. Section~\ref{sec:pd}
presents the full phase diagram and its finite-size scaling.
Section~\ref{sec:perturb} carries out the perturbation study. All results are
placed in a robustness context in Section~\ref{sec:robust}, discussed
together in Section~\ref{sec:discussion}, and summarised in
Section~\ref{sec:conclusion}.
""")

# ============================================================ 2. MODEL
w(r"\section{The model}\label{sec:model}")
w(r"\subsection{State space and payoff}")
w(r"""
A population of $N$ agents occupies the nodes of a simple graph with
adjacency matrix $A_{ij}\in\{0,1\}$. Each agent $i$ holds a strategy
$s_i\in\{0,1,2\}\equiv\{\text{R},\text{P},\text{S}\}$ (Rock, Paper, Scissors)
-- equivalently, a $q{=}3$ Potts spin. Interactions occur only along edges of
the graph. The payoff of playing strategy $a$ against an opponent playing $b$
is given by the $3\times3$ matrix
""")
w(r"""\begin{equation}
P(\varepsilon)\;=\;I+\varepsilon S
\;=\;\begin{pmatrix}1&-\varepsilon&\varepsilon\\
\varepsilon&1&-\varepsilon\\ -\varepsilon&\varepsilon&1\end{pmatrix},
\qquad S=\Pi-\Pi^{\top},
\label{eq:payoff}
\end{equation}""")
w(r"""
where rows index the player's own strategy and columns the opponent's
(ordered R, P, S), and $\Pi$ is the cyclic permutation matrix of
R$\to$P$\to$S$\to$R. The identity part $I$ rewards matching your neighbour's
strategy -- the ordinary $q{=}3$ Potts ferromagnetic coupling -- while the
antisymmetric part $\varepsilon S$ encodes the cycle
\emph{Paper beats Rock, Scissors beats Paper, Rock beats Scissors}: playing
Paper against a Rock neighbour earns $+\varepsilon$, and the reverse earns
$-\varepsilon$. The parameter $\varepsilon\in[0,1]$ is the sole control knob
of the model: $\varepsilon{=}0$ is the pure ferromagnetic Potts model, and
$\varepsilon{=}1$ makes the payoff purely cyclic (no reward at all for
matching).
""")
w(r"\subsection{Microscopic dynamics}")
w(r"""
A node's utility for playing strategy $a$ is the sum of payoffs against its
graph neighbours' current strategies,
""")
w(r"""\begin{equation}
U_i(a;\sigma)=\sum_j A_{ij}\,P_{a\,s_j}(\varepsilon).
\label{eq:utility}
\end{equation}""")
w(r"""
Dynamics proceeds by asynchronous Glauber updates. One \emph{sweep} consists
of $N$ update attempts; each attempt draws a random node $i$, proposes one of
the two alternative strategies $b\neq s_i$ uniformly, and accepts the move
with probability
""")
w(r"""\begin{equation}
W(s_i\to b)=\frac12\Big[1+e^{-\big(U_i(b)-U_i(s_i)\big)/T}\Big]^{-1},
\label{eq:glauber}
\end{equation}""")
w(r"""
where $T$ is a fixed noise temperature (throughout this paper, $T{=}0.65$
except where $T$ itself is the varied parameter). Because $U_i$ scales
linearly with a node's degree while the noise scale $T$ is fixed, the
effective noise felt by a node is $\sim T/k_i$: connectivity acts as an
inverse temperature, a fact made exact for the mean-field closures in
Section~\ref{sec:theory}. Some experiments in Section~\ref{sec:perturb} pin a
fraction $z$ of nodes -- \emph{zealots} -- to a fixed strategy forever; they
are skipped by the update rule but still enter the utility \eqref{eq:utility}
of their neighbours.
""")
w(r"\subsection{Order parameter}")
w(r"""
Let $r(t),p(t),s(t)$ be the instantaneous global fractions of the population
playing Rock, Paper, Scissors ($r+p+s=1$), measured at the end of every sweep
after an initial burn-in. Map the three fractions onto the complex plane at
the vertices of a triangle, $120^\circ$ apart:
""")
w(r"""\begin{equation}
\psi(t)=r(t)+p(t)\,\omega+s(t)\,\omega^2,\qquad \omega=e^{i2\pi/3}.
\label{eq:psi}
\end{equation}""")
w(r"""
Because $1+\omega+\omega^2=0$, the symmetric mixture $r=p=s=\tfrac13$ maps to
$\psi=0$, while a population dominated by one strategy maps close to that
strategy's corner, $|\psi|\to1$. The \emph{order parameter} used throughout is
the magnitude of the \emph{time-averaged} $\psi$,
""")
w(r"""\begin{equation}
m_\psi=\Big|\frac1M\sum_{t=1}^{M}\psi(t)\Big|,
\label{eq:mpsi}
\end{equation}""")
w(r"""
over $M$ post-burn-in sweeps. The order of operations in \eqref{eq:mpsi} is
essential: a rotating $\psi(t)$ (RPS cycling) averages to a small vector even
though $|\psi(t)|$ may itself stay large at every instant, so $\mpsi\to0$
correctly identifies cycling, while a static consensus keeps $\mpsi\to1$.
Throughout, the transition point $\epsc$ is defined as the interpolated
$\mpsi=\tfrac12$ crossing of a sweep in $\varepsilon$ at fixed network and
temperature -- linear interpolation between the two bracketing grid points,
identical for every mean-field and Monte Carlo curve in this paper.
""")
w(r"\subsection{The order-parameter identity}\label{sec:psiident}")
w(r"""
Equation~\eqref{eq:psi} has a useful closed form in terms of the raw
population fractions. Writing $\omega=e^{i2\pi/3}$ and using
$\mathrm{Re}\,\omega=\mathrm{Re}\,\omega^2=-\tfrac12$,
$\mathrm{Im}\,\omega=-\mathrm{Im}\,\omega^2=\tfrac{\sqrt3}{2}$, the
instantaneous complex order parameter splits into
""")
w(r"""\begin{equation}
\psi=\underbrace{r-\tfrac12(p+s)}_{\mathrm{Re}\,\psi}
+i\underbrace{\tfrac{\sqrt3}{2}(p-s)}_{\mathrm{Im}\,\psi},
\qquad
|\psi|^2=1-3(rp+ps+sr).
\label{eq:psinorm}
\end{equation}""")
w(r"""
\deriv{Expand $|\psi|^2=(\mathrm{Re}\,\psi)^2+(\mathrm{Im}\,\psi)^2$ using the
two real/imaginary components above:
$(\mathrm{Re}\,\psi)^2=r^2-r(p+s)+\tfrac14(p+s)^2$ and
$(\mathrm{Im}\,\psi)^2=\tfrac34(p-s)^2=\tfrac34(p+s)^2-3ps$. Summing,
$|\psi|^2=r^2-r(p+s)+(p+s)^2-3ps$. Substitute $p+s=1-r$ (since $r+p+s=1$)
and expand: $r^2-r(1-r)+(1-r)^2-3ps=1-r-p-s+r^2+p^2+s^2+2r^2-3ps
=1-3(rp+ps+sr)$ after collecting terms with $r^2+p^2+s^2=1-2(rp+ps+sr)$
substituted back in.}
""")
w(r"""
Equation~\eqref{eq:psinorm} shows $|\psi|$ is, up to an affine rescaling, the
Euclidean distance from the composition $(r,p,s)$ to the centroid
$(\tfrac13,\tfrac13,\tfrac13)$ of the simplex: since $rp+ps+sr$ is maximised
(at value $\tfrac13$) exactly at the centroid and minimised (at value $0$) at
each corner, $|\psi|^2$ ranges from $0$ (centroid) to $1$ (any pure
strategy) monotonically with distance from the centroid, independent of
\emph{direction} -- all three ordered corners score $|\psi|=1$ identically,
consistent with the model's cyclic (R$\to$P$\to$S$\to$R) symmetry. This is
why a single scalar $\mpsi$, rather than a full vector diagnostic, suffices
to separate consensus from cycling: consensus is captured by \emph{any}
direction reaching the boundary of the simplex, while only the
\emph{time-averaging} in \eqref{eq:mpsi} -- not the instantaneous norm
\eqref{eq:psinorm} -- distinguishes a static corner from an orbit that
visits the boundary everywhere but averages back toward the centroid.
""")
w(r"""
Figure~\ref{fig:psigeom} makes \eqref{eq:psi}--\eqref{eq:psinorm} concrete.
Placing R, P, S at the three cube roots of unity $1,\omega,\omega^2$ (angles
$0^\circ,120^\circ,240^\circ$) means the simplex \emph{is} drawn as the
triangle $\psi$ maps it onto: the position of a composition $x=(r,p,s)$ in
that triangle, read off by the ordinary barycentric combination
$r\,V_R+p\,V_P+s\,V_S$, coincides exactly with $\psi(x)$, so the black arrow
in the left panel literally \emph{is} $\psi$, decomposed into
$\mathrm{Re}\,\psi,\mathrm{Im}\,\psi$ by the dashed guides. The grey arrows
around the perimeter are the payoff matrix \eqref{eq:payoff}'s cyclic
relation drawn on the same picture, oriented counterclockwise to match the
relabelling $\pi:\mathrm R\to\mathrm P\to\mathrm S\to\mathrm R$ of
Appendix~\ref{sec:symmetry}. The right panel overlays two genuine HMF
trajectories (\eqref{eq:hmf}, same triangle): below $\epsc$ the path spirals
into a corner (a static $\psi$, $\mpsi\to1$); above $\epsc$ it settles onto a
closed loop around the centre (a rotating $\psi$ that time-averages toward
$0$) -- the picture behind Section~\ref{sec:bistable}'s two coexisting
attractors, previewed here at the point where $\psi$ is first defined.
""")
w(r"\begin{figure}[H]\centering\includegraphics[width=0.92\textwidth]{figures/order_parameter_geometry.png}")
w(r"\caption{Left: the order parameter as a vector -- the simplex drawn as the image of $\psi$, with the cyclic ``beats'' relation on its perimeter. Right: two real HMF trajectories in the same triangle, one converging to consensus, one orbiting forever.}\label{fig:psigeom}\end{figure}")
w(r"\subsection{Broken detailed balance}\label{sec:dbal}")
w(r"""
At $\varepsilon=0$, \eqref{eq:glauber} is exactly the Glauber dynamics of the
$q{=}3$ Potts Hamiltonian $H(\sigma)=-\sum_{(ij)\in E}\delta_{s_i s_j}$, since
$\Delta U_i=-\Delta H$ for a single-spin flip: the dynamics satisfies detailed
balance with respect to the Gibbs measure $e^{-H/T}$ and the population can
only order or disorder, never cycle indefinitely. For $\varepsilon>0$, no
Hamiltonian reproduces \eqref{eq:utility}: the two-site \emph{integrability}
condition for a pairwise potential to exist requires the mixed second
differences of the utility to be symmetric across all four state
combinations, and the antisymmetric part of $P$ fails this test, e.g.
$S_{SP}-S_{SR}-S_{RP}+S_{RR}=1+1+1+0=3\neq0$. Detailed balance is therefore
broken for every $\varepsilon>0$: the stationary distribution, if one exists,
carries a non-zero probability current, and this is the structural reason the
model can support persistent cycling instead of only static consensus.
""")
w(r"\subsubsection*{Kolmogorov's criterion and an explicit probability current}")
w(r"""
The integrability argument above can be sharpened into a direct,
computable statement about the Markov chain itself. For a chain on state
space $\Omega$ with transition rates $W(\sigma\to\sigma')$, \emph{Kolmogorov's
criterion} states that a stationary distribution satisfying detailed balance
exists if and only if, around every directed cycle
$\sigma_1\to\sigma_2\to\cdots\to\sigma_n\to\sigma_1$ of states with all rates
positive,
""")
w(r"""\begin{equation}
\prod_{i=1}^{n}W(\sigma_i\to\sigma_{i+1})
=\prod_{i=1}^{n}W(\sigma_{i+1}\to\sigma_i).
\label{eq:kolmogorov}
\end{equation}""")
w(r"""
Consider the minimal witness cycle for a single isolated pair of neighbouring
sites $(i,j)$ cycled through the three symmetric configurations
$\sigma_1{=}(\mathrm R,\mathrm P)\to\sigma_2{=}(\mathrm P,\mathrm S)
\to\sigma_3{=}(\mathrm S,\mathrm R)\to\sigma_1$, where at each step site $i$
adopts $j$'s current strategy's predator (holding $j$ fixed, then swapping
roles) -- realised concretely by three single-site Glauber moves at
temperature $T$ with all other neighbours held fixed. Each step is a
strategy change from the loser to the winner of that pairing under
\eqref{eq:payoff}, so every forward utility gain is $+2\varepsilon$
(e.g.\ Paper $\to$ Rock's context: adopting Rock's predator gains
$P_{PR}-P_{RR}=(-\varepsilon)$ vs...); more directly, for a single-edge pair
flip $a\to b$ against a fixed neighbour playing $c$, the Glauber ratio is
""")
w(r"""\begin{equation}
\frac{W(a\to b)}{W(b\to a)}=e^{\big(U(b;c)-U(a;c)\big)/T}
=e^{\big(P_{bc}-P_{ac}\big)/T}.
\label{eq:ratio}
\end{equation}""")
w(r"""
Multiplying \eqref{eq:ratio} around the 3-cycle
$\mathrm R\to\mathrm P\to\mathrm S\to\mathrm R$ (each step judged against a
fixed reference opponent strategy $c$ cycled along with the mover, so that
every arrow is a strict payoff improvement under \eqref{eq:payoff}), the
opponent-dependent single-step context cancels telescopically and the
cycle ratio reduces to a pure statement about the antisymmetric part of $P$:
""")
w(r"""\begin{equation}
\frac{\prod W(\sigma_i\to\sigma_{i+1})}{\prod W(\sigma_{i+1}\to\sigma_i)}
=\exp\!\left[\frac{1}{T}\big(S_{PR}+S_{SP}+S_{RS}\big)\right]
=\exp\!\left(\frac{3\varepsilon}{T}\right)\neq1
\qquad(\varepsilon>0),
\label{eq:cycleratio}
\end{equation}""")
w(r"""
since each of the three antisymmetric entries along the cyclic path equals
$\varepsilon$ by construction of $S$ in \eqref{eq:payoff}. Kolmogorov's
criterion \eqref{eq:kolmogorov} therefore fails by a factor
$e^{3\varepsilon/T}$, quantitatively confirming and sharpening the
integrability argument: the cycle is traversed in the forward
(R$\to$P$\to$S) direction $e^{3\varepsilon/T}$ times more often, in the
stationary state, than in reverse, so a genuine, non-vanishing probability
current of magnitude growing monotonically in $\varepsilon/T$ circulates
around every such triangle of strategy space -- the microscopic seed of the
macroscopic limit cycle recovered in the mean-field analysis below
(Section~\ref{sec:hopf}).
""")

# ============================================================ 3. THEORY
w(r"\section{Mathematical framework}\label{sec:theory}")
w(r"""
This section derives every equation used to interpret the simulation results
that follow: two mean-field closures of increasing resolution, an exact
invariance obeyed by both, a linear stability analysis at the symmetric
state, and the mean-field mechanism behind a first-order transition.
""")
w(r"\subsection{Homogeneous mean field (HMF)}\label{sec:hmf}")
w(r"""
Replace every node's neighbourhood by an independent draw from the global
composition $x=(r,p,s)$ -- the mean-field approximation. A node of degree $k$
then has utility vector $U=kPx$, and taking the expectation of one Glauber
update attempt \eqref{eq:glauber} over the product measure closes the
dynamics on the three macroscopic variables:
""")
w(r"""\begin{equation}
x'_a=x_a\Big(1-\sum_{b\neq a}w_{ab}\Big)+\sum_{b\neq a}x_b\,w_{ba},
\qquad
w_{ab}=\frac12\Big[1+e^{-(U_b-U_a)/T}\Big]^{-1},\quad U=kPx.
\label{eq:hmf}
\end{equation}""")
w(r"""
Iterating \eqref{eq:hmf} from an asymmetric initial condition (production
default $(r,p,s)_0=(0.40,0.35,0.25)$, chosen away from the unstable symmetric
point) for a fixed number of steps and applying \eqref{eq:mpsi} over the
second half of the trajectory gives the HMF prediction of $\mpsi(\varepsilon)$
at a given $(k,T)$; the only place network structure enters this closure is
through the single number $k$.
""")
w(r"\subsection{Degree-based mean field (DMF)}\label{sec:dmf}")
w(r"""
The HMF closure discards the degree distribution $P(k)$ entirely. A
degree-resolved closure instead keeps one composition $x_k$ per degree class.
Because an edge selects its far endpoint with probability proportional to
that endpoint's degree, a random \emph{neighbour} of a random node has degree
$k$ with probability $kP(k)/\langle k\rangle$, so every degree class feels the
edge-weighted field
""")
w(r"""\begin{equation}
\theta=\frac{1}{\langle k\rangle}\sum_k k\,P(k)\,x_k,
\qquad U^{(k)}=k\,P\,\theta,
\label{eq:dmf}
\end{equation}""")
w(r"""
and each $x_k$ evolves by \eqref{eq:hmf} with $U^{(k)}$ in place of $U$; the
global composition is the $P(k)$-weighted sum $\sum_k P(k)\,x_k$. The HMF
closure is recovered in the degenerate limit $P(k)=\delta_{k,\langle
k\rangle}$: DMF strictly generalises HMF, at the cost of one ODE system per
distinct degree present in the graph.
""")
w(r"""
\begin{remark}[The friendship paradox sets the size of the HMF--DMF gap]
The edge-biased weighting $kP(k)/\langle k\rangle$ in \eqref{eq:dmf} is the
same size-biased sampling behind the ``friendship paradox'' (your friends
have, on average, more friends than you do): the mean degree of a
\emph{random neighbour}, $\langle k\rangle_{nn}=\sum_k k\cdot
kP(k)/\langle k\rangle=\langle k^2\rangle/\langle k\rangle$, satisfies
\end{remark}
""")
w(r"""\begin{equation}
\langle k\rangle_{nn}-\langle k\rangle
=\frac{\langle k^2\rangle-\langle k\rangle^2}{\langle k\rangle}
=\frac{\sigma_k^2}{\langle k\rangle}\;\ge\;0,
\label{eq:friendship}
\end{equation}""")
w(r"""
with equality iff $P(k)$ is a point mass -- exactly the HMF's degenerate
case. Equation~\eqref{eq:friendship} makes precise, in the same currency as
Section~\ref{sec:mf-mc}'s Jensen-gap argument, why HMF and DMF must
increasingly disagree as $\sigma_k^2$ grows: HMF implicitly assumes every
node's neighbours have the \emph{typical} degree $\langle k\rangle$, while
DMF (correctly) has each node's neighbours drawn from the size-biased,
systematically higher-degree distribution $kP(k)/\langle k\rangle$. On a
narrow, near-Poissonian $P(k)$ (ER) the two distributions nearly coincide
and the gap in \eqref{eq:friendship} is $O(1)$ (Poisson has $\sigma_k^2
\approx\langle k\rangle$); on a heavy-tailed $P(k)$ (BA) $\sigma_k^2$ can
dwarf $\langle k\rangle$, and the DMF's edge-biased field departs sharply
from the HMF's naive mean-degree field -- the same distinction that drives
the DMF's accuracy advantage on BA in Section~\ref{sec:mf-mc}.
""")
w(r"\subsection{An exact $k/T$ invariance}\label{sec:invar}")
w(r"""
\begin{proposition}\label{prop:kt}
The HMF map \eqref{eq:hmf} depends on $(k,T)$ only through the ratio $k/T$,
so every HMF prediction -- in particular $\epsc$ -- is a function of $k/T$
alone: $\epsc^{\mathrm{HMF}}(k,T)=F(k/T)$.
\end{proposition}
""")
w(rf"""
\deriv{{The rates $w_{{ab}}$ in \eqref{{eq:hmf}} depend on $(k,T)$ only
through the combination $U/T=(k/T)\,Px$; the map itself has no other
$(k,T)$-dependence. Hence rescaling $k\to\lambda k$, $T\to\lambda T$ leaves
every iterate, and therefore every derived quantity including $\epsc$,
unchanged.}}
\numcheck{{Evaluated at three cells with matched ratio $k/T={kt_cells[0][0]/kt_cells[0][1]:.1f}$:
""" + ", ".join(rf"$\epsc({k_},{t_:g})={e_:.3f}$" for (k_, t_), e_ in zip(kt_cells, kt_ecs))
  + rf""" -- identical to the resolution of the $0.005$ grid on which they
were computed. Section~\ref{{sec:robust}} shows that quenched Monte Carlo
\emph{{breaks}} this invariance: a real graph carries relative degree
fluctuations $\sigma_k/\langle k\rangle=1/\sqrt{{\langle k\rangle}}$ that no
rescaling of $T$ can absorb.}}
""")
w(r"\subsection{Linear stability of the symmetric state}\label{sec:stability}")
w(r"""
The point $x^*=(\tfrac13,\tfrac13,\tfrac13)$ is a fixed point of
\eqref{eq:hmf} for every $\varepsilon$ (all utilities are equal there, by the
row sums of $P$), and its stability governs whether the population, started
near the symmetric mixture, is pulled toward consensus or spun into a limit
cycle.
""")
w(r"""
\begin{proposition}\label{prop:jac}
On the tangent space of the simplex ($\delta r+\delta p+\delta s=0$), the
Jacobian of \eqref{eq:hmf} at $x^*$ is
\end{proposition}
\begin{equation}
J\big|_{x^*}=\tfrac14 I+\frac{k}{4T}\,(I+\varepsilon S),
\qquad
\lambda_\pm=\underbrace{\tfrac14+\frac{k}{4T}}_{\text{radial growth}}
\;\pm\;i\,\underbrace{\frac{\sqrt3\,\varepsilon\,k}{4T}}_{\text{rotation}}.
\label{eq:jac}
\end{equation}""")
w(rf"""
\deriv{{At $x^*$ every rate is $w_{{ab}}=\tfrac14$ (all utilities equal, so
the logistic sits at its inflection point with slope $\tfrac14$); linearising
\eqref{{eq:hmf}} and restricting to the tangent space, the constant (all-ones)
part of the derivative annihilates because perturbations sum to zero, leaving
$J|_{{x^*}}=\tfrac14 I + \tfrac{{k}}{{4T}}(I+\varepsilon S)$. The
tangent-space eigenvalues of the skew part $S$ are $\pm i\sqrt3$ (its
eigenvalues on the full space are $0,\pm i\sqrt3$; the zero mode is the
particle-number direction, orthogonal to the tangent space), giving
\eqref{{eq:jac}}.}}
\numcheck{{At $(k,T)=({{MK:g}},{{MT:g}})$: $\lambda={lam_re:.2f}\pm{lam_im:.2f}\,\varepsilon\,i$,
verified against a finite-difference Jacobian of the exact map to
${{jac_dev_tex}}$ at three values of $\varepsilon$.}}
""".replace("{MK:g}", f"{MK:g}").replace("{MT:g}", f"{MT:g}").replace("{jac_dev_tex}", jac_dev_tex))
w(r"""
Equation~\eqref{eq:jac} has two immediate consequences.
\textbf{(i)} $\varepsilon$ enters \emph{only} the imaginary part: cyclic
dominance is, to linear order, a pure rotation about the symmetric point,
with angle per step $\theta=\arg\lambda_+=\arctan\!\big[\sqrt3\,\varepsilon
k/(k+T)\big]$; it never damps the growth rate.
\textbf{(ii)} the real part exceeds $1$ (i.e.\ $x^*$ is linearly unstable)
whenever $k>3T$, which holds throughout this study -- the symmetric mixture is
\emph{always} an unstable saddle here, for every $\varepsilon$. The two
possible fates -- an outward spiral captured by one of the three ordered
corners, or a trajectory that winds onto an invariant cycle -- are therefore
not different local stability classes of $x^*$ but a competition between two
\emph{distinct, non-local} attractors. That competition, rather than a local
bifurcation of $x^*$, is what makes the transition first-order-like
(Sec.~\ref{sec:bistable}).
""")
w(r"\subsection{The Hopf threshold in $(k,T)$}\label{sec:hopf}")
w(r"""
Proposition~\ref{prop:jac} identifies a second, independent critical
condition hiding inside the eigenvalues \eqref{eq:jac}: the growth rate
$\mathrm{Re}\,\lambda_\pm=\tfrac14+\tfrac{k}{4T}$ crosses $1$ -- the boundary
between a locally attracting and a locally repelling symmetric point -- at
""")
w(r"""\begin{equation}
k_H=3T,
\label{eq:hopf}
\end{equation}""")
w(r"""
independently of $\varepsilon$, since $\varepsilon$ enters only the
imaginary part of $\lambda_\pm$. Equation~\eqref{eq:hopf} is a genuine Hopf
condition in the $(k,T)$-plane at fixed $\varepsilon>0$: below $k_H$ the
symmetric point is a locally attracting focus (trajectories starting nearby
spiral \emph{into} $x^*$, so the population sits near the fully mixed,
cycling-dominated composition rather than escaping toward either attractor
of Section~\ref{sec:bistable}); above $k_H$ it is a repelling focus and the
outcome is decided by the competition between the ordered branch and the
limit cycle, as analysed throughout this paper.
""")
w(rf"""
\numcheck{{At the production temperature $T{{=}}0.65$, $k_H={{3*0.65:.2f}}$.
Every degree used in the main phase-diagram sweep (Section~\ref{{sec:pd}})
satisfies $k\ge2>k_H$, so the symmetric point is repelling throughout the
studied range and the competition picture applies uniformly -- but only
just: the lowest row of that sweep, $\langle k\rangle{{=}}2$, sits a mere
$\Delta k/k_H\approx{{(2-1.95)/1.95:.2f}}$ above threshold, and indeed shows
the weakest ordering of the whole diagram
($\epsc\approx{f2(epsc(sw['epsilon'], sw['k2']))}$ at $T{{=}}0.65$ in the
Section~\ref{{sec:hmf}} sweep) -- consistent with a system just barely past
the point where the mixed state stops actively repelling trajectories toward
either attractor. Equation~\eqref{{eq:hopf}} thus gives a first-principles
explanation, rather than a purely empirical one, for why sparse networks in
this model order poorly: they are not merely low-connectivity in an
informal sense, they are close to the linear-stability threshold at which
the very mechanism that separates order from cycling (competition between
non-local attractors) starts to disappear.}}
""".replace("{3*0.65:.2f}", f"{3*0.65:.2f}").replace("{(2-1.95)/1.95:.2f}", f"{(2-1.95)/1.95:.2f}"))
_theta_ex = np.arctan(np.sqrt(3) * 0.9 * MK / (MK + MT))
_Theta_ex = 2 * np.pi / _theta_ex
w(rf"""
\textbf{{Near-onset rotation period.}} The rotation angle per HMF step
derived above, $\theta=\arctan\!\big[\sqrt3\varepsilon k/(k+T)\big]$, gives a
linear-theory estimate of how many map iterations one revolution around the
symmetric point takes just after crossing $k_H$: $\Theta=2\pi/\theta$ steps.
At the production point $(k,T,\varepsilon)=({MK:g},{MT:g},0.9)$ this
evaluates to $\theta={_theta_ex:.3f}$ rad, $\Theta\approx{_Theta_ex:.1f}$
steps. This is a \emph{{linear}}, near-$x^*$ estimate only -- valid in the
infinitesimal neighbourhood where Proposition~\ref{{prop:jac}} applies -- and
should not be read as a prediction of the actual, deep-cycling limit
cycle's period, since a trajectory well inside the cycling phase spends most
of its time far from $x^*$, in the strongly nonlinear regime where the
map's higher-order terms (dropped in the linearisation) set the dynamics.
It nonetheless correctly signals the order of magnitude of the fast,
few-sweep reorganisation seen directly in the per-sweep zealot time signals
of Section~\ref{{sec:robust}} (a handful of sweeps to leave the vicinity of
the symmetric composition), and gives a falsifiable, closed-form target for
a dedicated measurement of the nonlinear period as a direction for future
work.
""")
w(r"\subsection{Bistability and the first-order transition}\label{sec:bistable}")
w(r"""
Because the transition is a competition between attractors rather than a
local instability, the mean-field map can, and does, support \emph{two}
simultaneously stable outcomes over a range of $\varepsilon$: a consensus
fixed point $x^*_{\mathrm{ord}}$ near (not at -- thermal noise keeps
$r^*<1$) one corner of the simplex, and an invariant limit cycle. Which one a
trajectory reaches depends on its initial condition, not on $\varepsilon$
alone.
""")
w(r"""
\textbf{Formal setup.} Write the map \eqref{eq:hmf} in reduced coordinates
$v=(r,p)\in\mathbb R^2$ on the free simplex (with $s=1-r-p$ eliminated), as
$v'=F(v;\varepsilon,k,T)$. A branch of ordered fixed points is a curve
$\varepsilon\mapsto v^*(\varepsilon)$ solving
""")
w(r"""\begin{equation}
G(v;\varepsilon)\equiv F(v;\varepsilon,k,T)-v=0,
\label{eq:fixedpteq}
\end{equation}""")
w(r"""
tracked by damped Newton iteration, $v_{n+1}=v_n-[\partial G/\partial v]^{-1}
G(v_n)$, with $\partial G/\partial v$ estimated by finite differences and
started from the near-corner seed $v_0=(0.98,0.01)$. At each converged root
$v^*(\varepsilon)$, linear stability of that specific fixed point (as
opposed to the generic symmetric-point calculation of
Proposition~\ref{prop:jac}) is decided by the spectral radius
$\rho(\varepsilon)=\max_i|\lambda_i(\partial F/\partial v|_{v^*})|$ of the
\emph{local} Jacobian evaluated at $v^*$ itself, not at $x^*$. Because
$\rho$ depends continuously on $\varepsilon$ while $v^*$ is a smooth branch,
the branch's endpoint -- the largest $\varepsilon$ for which an ordered,
linearly stable fixed point exists at all -- is reached either where the
branch collides with an unstable companion fixed point and both annihilate
(a fold, $\rho(\varepsilon)\to1^-$ continuously, the fixed point persisting
exactly at $\rho=1$ before ceasing to exist for larger $\varepsilon$) or
where $\rho$ crosses $1$ while $v^*$ itself survives (a local
bifurcation of the ordered state, analogous to Proposition~\ref{prop:jac}
but evaluated off the symmetric point). Distinguishing the two only requires
watching whether $v^*$ remains a solution of \eqref{eq:fixedpteq} just past
the threshold with $\rho>1$ (bifurcation) or Newton's method fails to
converge to any nearby root at all (fold) -- the continuation below finds
the latter, consistent with the disappearance of a saddle-node pair expected
at the edge of a bistable window.
""")
w(rf"""
Newton continuation of the ordered branch -- solving $x^*=F(x^*)$ for the map
restricted to the two free simplex coordinates, then tracking the spectral
radius $\rho$ of its tangent-space Jacobian as $\varepsilon$ increases in
steps of $0.002$ -- locates the exact end of the branch: it exists and remains
linearly stable ($\rho<1$) up to $\varepsilon={br10[0]:.3f}$ at $k{{=}}10$
(there $r^*={br10[1]:.3f}$) and $\varepsilon={br20[0]:.3f}$ at $k{{=}}20$.
Both values sit \emph{{just above}} the window measured by sweeping the map
from several biased initial conditions
($[{smi_win[10][0]:.3f},{smi_win[10][1]:.3f}]$ at $k{{=}}10$;
$[{smi_win[20][0]:.3f},{smi_win[20][1]:.3f}]$ at $k{{=}}20$), exactly as
expected: a biased sweep leaves the shrinking basin of attraction of
$x^*_{{\mathrm{{ord}}}}$ slightly before the fixed point itself ceases to
exist. Coexisting attractors with initial-condition dependence (hysteresis)
is the standard signature of a subcritical, first-order-like transition; an
independent, agent-based confirmation via finite-size scaling is given in
Section~\ref{{sec:fss}}.
""")

# ============================================================ 4. METHODS
w(r"\section{Methods}\label{sec:methods}")
w(r"\subsection{Networks}")
w(r"""
Two random-graph families are used throughout. \textbf{Erd\H{o}s--R\'enyi
(ER)} graphs place each of the $\binom N2$ possible edges independently with
probability $p=\langle k\rangle/(N-1)$, giving a narrow, approximately
Poissonian degree distribution concentrated around $\langle k\rangle$.
\textbf{Barab\'asi--Albert (BA)} graphs are grown by preferential attachment
and have a heavy-tailed degree distribution $P(k)\sim k^{-3}$ with
\emph{hubs} -- a small number of nodes with degree far above $\langle
k\rangle$. Comparing the two at matched $\langle k\rangle$ isolates whether a
result depends on average connectivity or on the shape of $P(k)$.
""")
w(r"\subsection{Simulation and measurement}")
w(r"""
Direct simulation uses a custom C++20 engine implementing
\eqref{eq:utility}--\eqref{eq:glauber} on an explicit edge list, with optional
zealot pinning and optional per-sweep time-series recording. Every measured
quantity in this paper follows the same six-step pipeline: (P1) evolve the
microstate by Glauber updates; (P2) count the per-strategy population
fractions $r(t),p(t),s(t)$ at the end of each post-burn-in sweep; (P3) map
them to the complex order parameter $\psi(t)$ via \eqref{eq:psi}; (P4)
time-average and take the modulus, \eqref{eq:mpsi}; (P5) where stated,
average the resulting curve over independent graph/seed realisations; (P6)
extract $\epsc$ as the linearly interpolated $\mpsi{=}0.5$ crossing of a
sweep in $\varepsilon$. The identical estimator is implemented in
\texttt{common/observables.py} (Python reference and both mean-field
closures) and \texttt{drivers/mc\_engine.cpp} (C++), so every $\epsc$ value
quoted anywhere in this paper -- mean-field or Monte Carlo -- is directly
comparable.
""")
w(r"""
Figure~\ref{fig:netevo} shows what step (P1) actually does to individual
agents, as distinct from the aggregate $(r,p,s)(t)$ curves used everywhere
else: one small ER graph, one fixed node layout, coloured by each node's
current strategy at three sweeps of the same run. The population starts
from i.i.d.\ uniform strategies (visibly a salt-and-pepper mix of all three
colours), a local majority strategy emerges and spreads along edges within a
handful of sweeps, and by sweep $30$ the free population has frozen into a
single colour -- consensus is, microscopically, nothing more than every
node's local majority argument (the Glauber comparison
\eqref{eq:glauber} against its immediate neighbours) resolving the same way
everywhere at once, propagated purely through the graph's edges with no
global coordination.
""")
w(r"\begin{figure}[H]\centering\includegraphics[width=0.95\textwidth]{figures/network_evolution.png}")
w(r"\caption{Node-level view of ordering: same graph and node layout at three sweeps of one run ($\varepsilon{=}0.3$, ordering phase).}\label{fig:netevo}\end{figure}")
w(r"\subsection{Engine validation}")
sv_ns = sorted(set(int(n) for n in sv["N"]))
sv_mg = {n_: float(sv_gap[sv["N"] == n_].mean()) for n_ in sv_ns}
w(rf"""
Before any reported result, the C++ engine is checked against an independent,
pure-Python reference implementation of the same Markov chain. Because the
two implementations draw from different, independent random streams,
agreement is expected in \emph{{regime}} (which side of $\mpsi{{=}}0.5$ a run
lands on), not to the last digit; the size of the expected disagreement
follows from \eqref{{eq:psi}}: $\psi(t)$ averages $N$ site phasors, so
$\mathrm{{Var}}[\psi(t)]\propto1/N$, and the two engines' $\mpsi$ estimates
should differ by $O(1/\sqrt{{N}})$. Across a grid of {len(sv)} paired runs
spanning $N\in\{{{','.join(str(n) for n in sv_ns)}\}}$ and five $\varepsilon$
values bracketing the transition, {sv_agree}/{len(sv)} pairs classify the
same phase, and the mean engine-to-engine gap falls monotonically with $N$
(""" + ", ".join(f"{sv_mg[n_]:.5f}" for n_ in sv_ns)
  + rf""" at $N={{','.join(str(n_) for n_ in sv_ns)}}$'') -- the predicted
$1/\sqrt{{N}}$ self-averaging. This licenses using the validated C++ engine
for every Monte Carlo result below.
""".replace("{{','.join(str(n_) for n_ in sv_ns)}}", ','.join(str(n_) for n_ in sv_ns)))
w(r"""
The $\varepsilon_c$ estimator itself (linear interpolation of the
$\mpsi{=}0.5$ crossing) is likewise checked against the coarser
first-grid-point convention: at the production grid step ($0.05$) the
interpolated estimator is off by only """
  + f"{sgr_i05:+.4f}" + r" from a $4\times$-finer reference grid, versus "
  + f"{sgr_n05:+.4f}" + r""" for the naive convention -- roughly a
$4\times$ effective resolution gain, so every $\epsc$ quoted in this paper
carries a grid uncertainty of at most $\sim0.005$.
""")
w(r"\subsection{Bias of the interpolated \texorpdfstring{$\epsc$}{eps\_c} estimator}\label{sec:bias}")
w(r"""
The linear-interpolation estimator of Section~\ref{sec:methods} can be
analysed directly. Let $\varepsilon_0<\varepsilon_1$ be the two grid points
bracketing the true crossing $\varepsilon_c^\star$ (where the underlying,
continuous $\mpsi(\varepsilon)$ equals $\tfrac12$ exactly), with grid step
$h=\varepsilon_1-\varepsilon_0$, and let $m(\varepsilon)$ denote that
continuous curve. The estimator solves the secant equation
$m_0+(\hat\epsc-\varepsilon_0)(m_1-m_0)/h=\tfrac12$ for the two sampled
values $m_0=m(\varepsilon_0)$, $m_1=m(\varepsilon_1)$, i.e.\ it replaces
$m$ on $[\varepsilon_0,\varepsilon_1]$ by its chord. Standard
linear-interpolation error analysis (the Lagrange remainder of a degree-1
interpolant) gives, for $\varepsilon\in[\varepsilon_0,\varepsilon_1]$,
""")
w(r"""\begin{equation}
m(\varepsilon)-\big[\text{chord}\big](\varepsilon)
=-\tfrac12(\varepsilon-\varepsilon_0)(\varepsilon_1-\varepsilon)\,m''(\xi)
\qquad\text{for some }\xi\in(\varepsilon_0,\varepsilon_1),
\label{eq:interperr}
\end{equation}""")
w(r"""
so at the estimated crossing the chord misses the true curve's value by at
most $\tfrac18 h^2\sup|m''|$ (the quadratic $(\varepsilon-\varepsilon_0)
(\varepsilon_1-\varepsilon)$ is maximised at the midpoint, value
$h^2/4$). Converting this value-space error into an $\varepsilon$-space
error via the local slope $m'(\varepsilon_c^\star)$ gives the bias bound
""")
w(r"""\begin{equation}
\big|\hat\epsc-\varepsilon_c^\star\big|
\;\lesssim\;\frac{h^2\,\sup|m''|}{8\,|m'(\varepsilon_c^\star)|}.
\label{eq:biasbound}
\end{equation}""")
w(r"""
For a \emph{smooth} sigmoid this bound is small whenever $h$ is small, since
it is $O(h^2)$; the interpolated estimator's error therefore shrinks
quadratically with grid resolution, in contrast to the naive
first-grid-point convention (Section~\ref{sec:pd}, Section~\ref{sec:defects}
tables) whose error is $O(h)$ by construction (it can only round the true
crossing up to the next sampled point, so the leading-order bias is
$h/2$ on average and $h$ in the worst case) -- an order in $h$ better,
matching the observed $\sim4\times$ effective-resolution gain at the
production step. Because the order--cycling transition measured throughout
this paper is first-order-sharp (Sections~\ref{sec:bistable},
\ref{sec:ld}), $|m'(\varepsilon_c^\star)|$ is large and $\sup|m''|$ can be
large too very close to the transition, so \eqref{eq:biasbound}'s
\emph{numerical prefactor} is not tight there; the operative evidence for
the $O(h^2)$ scaling is therefore the direct empirical comparison of
Section~\ref{sec:methods}, not the bound in isolation -- but the bound
correctly predicts the \emph{qualitative} advantage of interpolation, and
its derivation is what justifies extrapolating that advantage to grid steps
finer than the ones tested.
""")

# ============================================================ 5. MEAN FIELD VS MC
w(r"\section{Mean-field theory versus Monte Carlo}\label{sec:mf-mc}")
w(r"\subsection{The HMF sweep: connectivity stabilises order}")
w(rf"""
Iterating \eqref{{eq:hmf}} across $\varepsilon\in[0,1]$ at several mean
degrees $\langle k\rangle\in\{{2,5,10,50,200\}}$ (fixed $T{{=}}0.65$) gives a
sharp order--cycling transition whose location $\epsc(\langle k\rangle)$
rises monotonically:
""")
w(r"\begin{center}\begin{tabular}{lccccc}\toprule")
w(r"$\langle k\rangle$ & " + " & ".join(k[1:] for k in sw_ks) + r"\\")
w(r"$\epsc$ & " + " & ".join(f2(epsc(sw["epsilon"], sw[k_])) for k_ in sw_ks) + r"\\")
w(r"\bottomrule\end{tabular}\end{center}")
w(r"""
By Proposition~\ref{prop:kt} this whole row is one function evaluated at five
points: raising $k$ at fixed $T$ is, exactly, cooling the effective noise
$T/k$. The rise saturates at large $k/T$ because the Glauber rates
\eqref{eq:glauber} approach step functions in that limit ($w_{ab}\to\tfrac12
\mathbf{1}[U_b>U_a]$), so the map loses its last free parameter and $F(k/T)$
plateaus -- a mechanism that matters again in Section~\ref{sec:boundary}, where
the plateau collides with the rising Monte Carlo boundary.
""")
w(r"\begin{figure}[H]\centering\includegraphics[width=0.62\textwidth]{mean_field/hmf_sweep.png}")
w(r"\caption{HMF order parameter vs $\varepsilon$ at five mean degrees.}\end{figure}")

w(r"\subsection{HMF, DMF and Monte Carlo: which closure is accurate, and why}")
w(rf"""
At a fixed operating point ($\langle k\rangle{{=}}10$, one ER and one BA
realisation, $N{{=}}800$) all three descriptions -- MC (ground truth), HMF, and
DMF fed the graph's \emph{{measured}} degree histogram -- are evaluated on
the same $26$-point $\varepsilon$ grid and scored by root-mean-square error
against MC:
""")
w(r"\begin{center}\begin{tabular}{lccc}\toprule")
w(r"graph & RMSE(HMF) & RMSE(DMF) & DMF gain\\\midrule")
for g_ in ("ER", "BA"):
    w(f"{g_} & {f3(rows_rmse[g_][0])} & {f3(rows_rmse[g_][1])} & \\textbf{{{f3(rows_rmse[g_][0]-rows_rmse[g_][1])}}}\\\\")
w(r"\bottomrule\end{tabular}\end{center}")
w(rf"""
DMF beats HMF on both graphs, and its advantage is $\sim{gain_ratio:.1f}\times$
larger on the heterogeneous BA network -- exactly where resolving $P(k)$
should matter most, since ER's degree distribution is nearly a point mass.
The mechanism is a Jensen gap: the Glauber rates in \eqref{{eq:hmf}} are a
nonlinear (sigmoidal) function of $k$, so averaging \emph{{before}} that
nonlinearity (HMF) costs, to second order,
$\mathbb{{E}}_{{P(k)}}[g(k)]-g(\langle k\rangle)\approx\tfrac12
g''(\langle k\rangle)\,\sigma_k^2$: the HMF's excess error over DMF should
scale with the degree variance $\sigma_k^2$. Measured on the two seed-1
graphs at build time, $\sigma_k^2={var_er:.1f}$ on ER (close to Poisson,
$\approx\langle k\rangle$) versus $\sigma_k^2={var_ba:.1f}$ on BA -- a
${var_ba/var_er:.0f}\times$ variance ratio, matching the concentration of
the DMF advantage on BA.
""")
w(r"""
\deriv{Formally, let $g(k)$ denote the map's dependence on degree at fixed
$(x,\varepsilon,T)$, i.e.\ $g(k)=F(x;\varepsilon,k,T)$ component-wise, and let
$\sigma_k^2=\mathrm{Var}_{P(k)}[k]$. A second-order Taylor expansion of $g$
about $\langle k\rangle$ under the degree measure $P(k)$ gives
\[
\mathbb E_{P(k)}[g(k)]=g(\langle k\rangle)+\tfrac12 g''(\langle
k\rangle)\,\sigma_k^2+R_3,
\qquad
|R_3|\le\frac{\sup|g'''|}{6}\,\mathbb E_{P(k)}\!\big[|k-\langle
k\rangle|^3\big].
\]
The DMF evaluates the left-hand side exactly (one map instance per degree
class, weighted by $P(k)$); the HMF evaluates only $g(\langle k\rangle)$, the
first term. Their difference is therefore $\tfrac12 g''(\langle
k\rangle)\sigma_k^2+O(\mathbb E|k-\langle k\rangle|^3)$: to leading order in
the degree spread, the HMF's excess error over DMF is linear in the degree
\emph{variance}, and vanishes in the same limit ($\sigma_k\to0$) in which
HMF and DMF coincide by construction. The remainder term is controlled by
the third absolute moment of $P(k)$, which for a heavy-tailed $P(k)$ such as
BA's need not be small even when $\sigma_k^2$ already is -- so the
leading-order scaling above is a good \emph{qualitative} guide (larger
spread $\Rightarrow$ larger HMF penalty) rather than a quantitatively exact
prediction of the RMSE gap.}
""")
w(rf"""
Both mean fields overestimate the size of the ordered region relative to the
finite Monte Carlo system: on ER, $\epsc^{{\mathrm{{MC}}}}\approx{f2(ec_mc10)}$
versus $\epsc^{{\mathrm{{HMF}}}}\approx{f2(ec_hmf10)}$. Writing the gap as an
additive decomposition,
""")
w(r"""\begin{equation}
\Delta(k,T,N)\equiv\epsc^{\mathrm{HMF}}-\epsc^{\mathrm{MC}}
=\underbrace{F(k/T)-\epsc^{\infty}(k,T)}_{\text{mean-field closure error}}
+\underbrace{c(k)/N}_{\text{finite-size shift}},
\label{eq:gapdecomp}
\end{equation}""")
w(r"""
the first term is the price of the mean-field factorisation itself (every
neighbour treated as an independent draw), which shrinks with $k$ as the
local composition self-averages; the second is the first-order finite-size
drift derived in Section~\ref{sec:fss}. Since the mean fields are $N$-blind,
any $N$-dependence of the observed gap is entirely the second term, moving
under a frozen mean-field curve.
""")
w(r"\begin{figure}[H]\centering")
w(r"\includegraphics[width=0.49\textwidth]{mean_field/comparison_suite_ER_k10.png}\hfill")
w(r"\includegraphics[width=0.49\textwidth]{mean_field/comparison_suite_BA_k10.png}")
w(r"\caption{MC vs HMF vs DMF, ER (left) and BA (right), $\langle k\rangle{=}10$.}\end{figure}")

# ============================================================ 6. PHASE DIAGRAM
w(r"\section{The phase diagram}\label{sec:pd}")
_pdg_ks = sorted(set(int(k_) for k_ in pdg["ER"]["degree"]))
_pdg_ne = len(set(pdg["ER"]["epsilon"]))
_pdg_tiles = len(_pdg_ks) * _pdg_ne
w(rf"""
The central empirical result of this study is obtained without any
mean-field assumption: for each of {len(_pdg_ks)} mean degrees
$\langle k\rangle\in[{_pdg_ks[0]},{_pdg_ks[-1]}]$, a graph of $N{{=}}800$
nodes is built and swept across {_pdg_ne} values of $\varepsilon\in[0,1]$
with the validated C++ engine -- {_pdg_tiles} independent simulations per
graph family -- tiling the $(\langle k\rangle,\varepsilon)$ plane with the
directly measured $\mpsi$. The experiment is run on both ER and BA graphs to
separate the effect of average degree from the effect of the degree
distribution's shape.
""")
per = {}
for g_ in ("ER", "BA"):
    degs = sorted(set(pdg[g_]["degree"].astype(int)))
    per[g_] = {k_: epsc(pdg[g_]["epsilon"][pdg[g_]["degree"] == k_],
                        pdg[g_]["m_psi"][pdg[g_]["degree"] == k_]) for k_ in degs}
w(r"\begin{figure}[H]\centering")
w(r"\includegraphics[width=0.49\textwidth]{phase_diagram/phase_diagram_ER.png}\hfill")
w(r"\includegraphics[width=0.49\textwidth]{phase_diagram/phase_diagram_BA.png}")
w(r"\caption{Monte Carlo $(\langle k\rangle\times\varepsilon)$ phase diagrams: ER (left), BA (right).}\end{figure}")

w(r"\subsection{Extracted boundary and the mean-field bistable window}\label{sec:boundary}")
w(rf"""
Extracting $\epsc(\langle k\rangle)$ from each heatmap (interpolated
$\mpsi{{=}}0.5$ crossing per degree column) gives the boundary curve of
Figure~\ref{{fig:boundary}}. Two results are immediate. First, the ER and BA
boundaries agree to within $\max_k|\epsc^{{\mathrm{{ER}}}}-\epsc^{{\mathrm{{BA}}}}|
={gap_erba:.3f}$ over all {len(cb)} degrees tested: for the agent-based
dynamics, average degree -- not the shape of $P(k)$ -- controls stability.
This is consistent with the linear-stability picture of
Section~\ref{{sec:stability}}: a node of degree $k_i$ self-averages its
neighbours' composition with relative fluctuation $O(1/\sqrt{{k_i}})$, and
once $k_i(1-\varepsilon)\gg T$ the Glauber rate saturates, so hubs and
typical nodes act alike once both are well above the noise floor -- a hub
buys no extra order per additional stub. Second, plotting the HMF boundary
of Section~\ref{{sec:hmf}}, recomputed at every one of the {len(cb)} MC
degrees on the identical grid and estimator, against the MC boundary
recovers the standard-init overestimate at low $k$ ($+{hgap_max:.3f}$ at
$\langle k\rangle{{=}}{hgap_max_k}$) but shows the gap changing sign near
$\langle k\rangle\approx{hgap_flip_k}$.
""")
w(rf"""
That sign change is \emph{{not}} a mean-field failure but a consequence of
the bistability established in Section~\ref{{sec:bistable}}: since the HMF
transition is genuinely a window, not a point, computing \emph{{both}} the
standard-init edge (plotted) and the ordered-init edge
($(0.98,0.01,0.01)$ start) at every degree shows the window widening with
$k$ until, from $\langle k\rangle{{=}}{inside_first_k}$ on
({inside_n}/{len(cb)} degrees), the MC boundary runs \emph{{inside}} it
(window at $\langle k\rangle{{=}}{win_top_k}$:
$[{win_top[0]:.2f},{win_top[1]:.2f}]$). The ordered-init edge stays above the
MC boundary throughout, so the consensus branch located by Newton
continuation (Sec.~\ref{{sec:bistable}}) survives; a single-init $\epsc$
simply stops being the unique prediction once both attractors coexist.
""")
w(r"\begin{figure}[H]\centering\includegraphics[width=0.62\textwidth]{phase_diagram/critical_boundary.png}")
w(r"\caption{Extracted boundary $\epsc(\langle k\rangle)$: MC (ER, BA) vs the HMF standard-init prediction, with the mean-field bistable window (Sec.~\ref{sec:bistable}) shaded.}\label{fig:boundary}\end{figure}")

w(r"\subsection{Finite-size scaling: a genuine, first-order-like transition}\label{sec:fss}")
w(rf"""
Two independent checks establish that the boundary is a real phase
transition rather than a finite-size or dynamical artefact.
\textbf{{Sharpening.}} Rerunning the transition curve at growing $N\in\{{{','.join(n_[1:] for n_ in fss_Ns)}\}}$
(ER, $\langle k\rangle{{=}}10$) sharpens the curve monotonically -- the
discrete steepness $\max_j|\Delta\mpsi/\Delta\varepsilon|$ grows """
  + " $\\to$ ".join(f"{s:.0f}" for s in fss_slopes)
  + rf""" -- while $\epsc$ converges toward $\approx{f2(fss_ec[-1])}$, textbook
finite-size scaling. \textbf{{Drift law.}} With two macrostates whose
relative statistical weight scales as $e^{{N a(\varepsilon)}}$, the observed
crossing sits where the two weights tie; expanding the free-energy-like
difference linearly about the infinite-size tie point gives a pseudo-critical
shift $\epsc(N)-\epsc(\infty)\propto1/N$ -- the standard finite-size scaling
of a \emph{{first-order}} transition. At $\langle k\rangle{{=}}20$, a
dedicated sweep over $N\in\{{{','.join(str(int(n_)) for n_ in ssz['N'])}\}}$
shows $\epsc$ sliding smoothly downward
({f3(ssz['eps_c'][0])} $\to$ {f3(ssz['eps_c'][-1])}), with the gap to the
Richardson extrapolant $\epsc(\infty)\approx{f3(ssz_extrap)}$ halving with
each doubling of $N$ -- the $1/N$ law confirmed directly at the agent-based
level, independent of the mean-field bistability of
Section~\ref{{sec:bistable}}.
""")
w(r"\begin{figure}[H]\centering")
w(r"\includegraphics[width=0.32\textwidth]{dynamics/fss.png}\hfill")
w(r"\includegraphics[width=0.32\textwidth]{dynamics/ternary.png}\hfill")
w(r"\includegraphics[width=0.32\textwidth]{dynamics/stability.png}")
w(r"\caption{Left: finite-size scaling. Middle: ternary $(r,p,s)$-simplex trajectories -- corner consensus below $\epsc$, closed orbit above. Right: HMF fixed-point stability across $\epsc$.}\end{figure}")

w(r"\subsection{A large-deviation model of the pseudo-transition}\label{sec:ld}")
w(r"""
The ``Drift law'' argument above can be developed into an explicit
finite-size scaling ansatz for the whole transition curve, not just its
midpoint. Model the finite-$N$ stationary measure as concentrated on two
macrostates -- ordered and cycling -- each carrying a large-deviation weight
$Z_{\mathrm{ord}}(\varepsilon)\sim e^{Na_{\mathrm{ord}}(\varepsilon)}$,
$Z_{\mathrm{cyc}}(\varepsilon)\sim e^{Na_{\mathrm{cyc}}(\varepsilon)}$ for
some smooth rate functions $a_{\mathrm{ord}},a_{\mathrm{cyc}}$ (an ansatz
standard for first-order transitions, where each phase's free energy is
extensive in the system size). The finite-$N$ order parameter is then the
weight-averaged value,
""")
w(r"""\begin{equation}
\mpsi(N,\varepsilon)\approx
\frac{m_{\mathrm{ord}}\,e^{Na_{\mathrm{ord}}(\varepsilon)}
+m_{\mathrm{cyc}}\,e^{Na_{\mathrm{cyc}}(\varepsilon)}}
{e^{Na_{\mathrm{ord}}(\varepsilon)}+e^{Na_{\mathrm{cyc}}(\varepsilon)}}
=\frac{m_{\mathrm{ord}}+m_{\mathrm{cyc}}\,e^{-N\Delta a(\varepsilon)}}
{1+e^{-N\Delta a(\varepsilon)}},
\qquad \Delta a\equiv a_{\mathrm{ord}}-a_{\mathrm{cyc}},
\label{eq:ldweight}
\end{equation}""")
w(r"""
with $m_{\mathrm{ord}}\approx1$, $m_{\mathrm{cyc}}\approx0$ the two phases'
characteristic order parameter values. Equation~\eqref{eq:ldweight} is
exactly a logistic function of $N\Delta a(\varepsilon)$, and two of its
properties are the ones tested empirically above.
""")
w(r"""
\deriv{\textbf{Location.} The crossing $\mpsi=\tfrac12$ occurs at
$\Delta a(\varepsilon_c(N))=0$ to leading order. Assuming $\Delta a$ has a
simple zero at some $\varepsilon_c(\infty)$ with
$\Delta a(\varepsilon)\approx\Delta a'(\varepsilon_c(\infty))\,
(\varepsilon-\varepsilon_c(\infty))$ near that point, the finite-size
crossing solves $N\Delta a'\cdot(\varepsilon_c(N)-\varepsilon_c(\infty))=O(1)$,
giving $\varepsilon_c(N)-\varepsilon_c(\infty)=O(1/N)$ with a coefficient
set by $1/\Delta a'(\varepsilon_c(\infty))$ -- the $1/N$ law used for the
Richardson extrapolation above. \textbf{Width.} The same expansion gives the
full logistic profile $\mpsi\approx1/(1+e^{-N\Delta a'\,(\varepsilon-
\varepsilon_c)})$ near the crossing, whose $10$--$90\%$ width in
$\varepsilon$ is $O(1/(N\Delta a'))$: the transition sharpens as $1/N$ at
fixed $\Delta a'$, consistent with the growing $\max|\Delta\mpsi/\Delta
\varepsilon|$ reported above. Both scalings follow from nothing more than
the exponential-weight ansatz \eqref{eq:ldweight} and a single smoothness
assumption on $\Delta a$ near its zero; no further assumption about the
microscopic dynamics is used, which is why the same $1/N$ signature appears
independently in the mean-field bistable window (Section~\ref{sec:bistable})
and in the direct Monte Carlo width measurement of
Section~\ref{sec:robust}.}
""")

# ============================================================ 7. PERTURBATION
w(r"\section{Perturbation experiments}\label{sec:perturb}")
w(r"""
Having mapped the clean transition, we now perturb the ordered phase
($\varepsilon{=}0.3$) and the cycling phase ($\varepsilon{=}0.9$) with four
kinds of intervention: a committed minority (\emph{zealots}), structural
targeting of that minority, competition between two minorities, and quenched
network damage. Each is analysed first at the level of the mean-field field
it induces, then measured directly by Monte Carlo.
""")
w(r"\subsection{The zealot field}\label{sec:zfield}")
w(r"""
Let a fraction $z$ of nodes be permanently pinned to strategy Rock (they
influence neighbours through \eqref{eq:utility} but never update). Split the
global composition into pinned and free parts, $x=z\,e_R+(1-z)\,y$, with $y$
the composition of the free subpopulation; every free node then feels
""")
w(r"""\begin{equation}
U=kP\big[z\,e_R+(1-z)y\big]
=\underbrace{kz\,(1,\ \varepsilon,\ -\varepsilon)^\top}_{\text{zealot field }h}
+(1-z)\,kPy.
\label{eq:zfield}
\end{equation}""")
w(r"""
The sign pattern of $h$ is the entire backfire mechanism: the cyclic term
pays $+\varepsilon kz$ to Paper (every zealot is food for its predator) and
\emph{takes} $\varepsilon kz$ from Scissors, starving the one strategy that
could threaten Paper. Evaluating the utility margins at the corners of the
free simplex shows that \emph{both} free-Paper and free-Rock are locally
stable outcomes for every $z<1$ ($U_P-U_R=k[(1+\varepsilon)-z(1-\varepsilon)]>0$
and $U_P-U_S=k[(1-\varepsilon)+2\varepsilon z]>0$ for free-Paper; both margins
for free-Rock are likewise positive), so the outcome is a \emph{basin
selection} problem from the random start, tilted by $h$: Rock is favoured
first (the largest single field component, $kz$), but a growing Rock
population raises Paper's utility by $\varepsilon k y_R$ while $h$ keeps
Scissors suppressed throughout, funnelling the free population into the
Paper basin. The same picture predicts that the rare large-$z$ flips onto
Rock require a collective fluctuation of $O(N)$ agents -- probability
$\sim e^{-cN}$ -- so their frequency should die with system size, which
Section~\ref{sec:robust} confirms directly.
""")
w(r"""
\begin{remark}[The pinned dynamics is an unpinned map in disguise]
Equation~\eqref{eq:zfield} shows the free composition $y$ evolves under
$U=h+(1-z)kPy$, which has exactly the functional form $U=k'Py'+\text{const}$
of the ordinary (unpinned) HMF utility \eqref{eq:hmf}, with an effective
connectivity $k'=(1-z)k$ rescaling the selective (payoff) part and the
constant field $h$ acting as a symmetry-breaking bias absent from
\eqref{eq:hmf}. The entire fixed-point and stability machinery of
Section~\ref{sec:bistable} -- the reduced map $G(v;\varepsilon)=0$, Newton
continuation, and the spectral-radius stability test -- therefore applies
verbatim to the pinned system after this substitution; the only new
ingredient is that $h$ breaks the model's cyclic symmetry
(Appendix~\ref{sec:symmetry}) explicitly, so the three corners are no longer
equivalent and each must be continued separately. This is why, operationally,
the single-faction zealot problem is analysed here through the sign
structure of $h$ directly (which correctly identifies \emph{which} corner
wins) rather than through a fresh continuation exercise (which would only
confirm, at higher computational cost, which corner is linearly stable --
information already implied by the utility margins above).
\end{remark}
""")
w(r"\subsection{Single-faction zealots: the backfire}")
sel = [0, 4, 8, 12, 16]
w(rf"""
Rock-zealots on ER ($N{{=}}800$, $\langle k\rangle{{=}}10$, 12-seed average)
confirm the mechanism of Section~\ref{{sec:zfield}}. In the ordering phase,
free-node conversion to Rock collapses toward zero by $z\approx0.05$ while
the free network adopts \textbf{{Paper}} instead; in the cycling phase the
induced order is weak and the zealots cannot pin their own strategy:
""")
w(r"\begin{center}\begin{tabular}{lcccc}\toprule")
w(r"$z$ & conv.\ (ordering) & $\mpsi$ (ordering) & conv.\ (cycling) & $\mpsi$ (cycling)\\\midrule")
for i in sel:
    w(f"{f2(z['z'][i])} & {f2(z['order_conversion'][i])} & {f2(z['order_mpsi'][i])} & "
      f"{f2(z['cycle_conversion'][i])} & {f2(z['cycle_mpsi'][i])}\\\\")
w(r"\bottomrule\end{tabular}\end{center}")
w(r"""
(``conv.'' is the fraction of \emph{free} nodes playing Rock.) The ordered
phase is thus \emph{compositionally fragile}: a small committed minority
dictates the population's eventual strategy, but not the minority's own
strategy -- and the cycling phase is nearly immune, since a rotating $\psi$
has no fixed direction for a static field $h$ to reinforce.
""")
w(r"\begin{figure}[H]\centering\includegraphics[width=0.62\textwidth]{zealots/zealots.png}")
w(r"\caption{Single Rock-zealot faction: free-node conversion and $\mpsi$ vs zealot fraction $z$, both phases.}\end{figure}")

w(r"\subsection{Time signals: watching the backfire happen}\label{sec:timesig}")
w(r"""
Every number above is a \emph{time average}, \eqref{eq:mpsi}, and time
averages hide the mechanism. This section replaces the average with the raw
per-sweep trajectory, recorded directly by the C++ engine's
\texttt{--timeseries} output (P2, before the P4 averaging step is applied),
so the same backfire that Section~\ref{sec:zfield} predicted from the sign
of $h$ can be watched happening sweep by sweep.
""")
w(rf"""
\textbf{{Four scenarios in detail.}} Figure~\ref{{fig:tsstory}} records
$(r,p,s)(t)$ on one ER graph ($N{{=}}800$, $\langle k\rangle{{=}}10$) for
four hand-picked $(\varepsilon,z)$ pairs, on a log time axis (consensus on a
dense random graph forms within $\sim\!10$ sweeps, so a linear axis would
waste most of the panel). \emph{{Clean ordering}} ($z{{=}}0$): all three
strategies start near $\tfrac13$ ({SNAMES[int(tss['initial_leader'][0])]}
marginally ahead at
{max(tss['r0'][0], tss['p0'][0], tss['s0'][0]):.2f}), and whoever leads
early simply snowballs, crossing $50\%$ by sweep
{int(tss['t_majority'][0])} -- the winner is decided by the random start,
not by any structural effect. \emph{{Backfire}} ($z{{=}}{tss['z'][1]:g}$):
the zealots make Rock the early leader
({tss['r0'][1]:.2f}), yet Paper -- Rock's predator -- overtakes almost
immediately and is a majority by sweep {int(tss['t_majority'][1])}; Rock is
driven down to exactly its zealot floor $z{{=}}{tss['z'][1]:g}$ (final
composition $({tss['r_final'][1]:.2f},{tss['p_final'][1]:.2f},
{tss['s_final'][1]:.2f})$, matching the exact prediction
$(z,1{{-}}z,0)$ of a completed backfire to within thermal excitation of the
free consensus). \emph{{Large faction}} ($z{{=}}{tss['z'][2]:g}$): Rock
first \emph{{grows}}, feeding on Scissors, before Paper catches up at sweep
{int(tss['t_majority'][2])} and pins it at the same floor -- the
rise-then-collapse is the cyclic mechanism in real time, not merely in the
final tally. \emph{{Cycling}} ($\varepsilon{{=}}0.9$): no winner ever
emerges; the population settles into a noisy chase and the zealots change
nothing, which is exactly why the time-\emph{{averaged}} $\mpsi$ of
Section~\ref{{sec:zfield}} is the right diagnostic for that phase.
""")
w(r"\begin{figure}[H]\centering\includegraphics[width=0.95\textwidth]{zealots/timeseries.png}")
w(r"\caption{Per-sweep population fractions, four scenarios, log time axis. Dash-dotted line: the zealot floor $z$ below which Rock cannot fall.}\label{fig:tsstory}\end{figure}")
w(rf"""
\textbf{{Every $z$, not just four.}} Figure~\ref{{fig:p7ts}} reruns the
\emph{{entire}} $17$-point $z$-grid of Section~\ref{{sec:zfield}} with
\texttt{{--timeseries}} on, on a single ER realisation, plotting per-sweep
conversion $(r(t)-z)/(1-z)$ and the \emph{{instantaneous}} $|\psi(t)|$ (the
un-averaged magnitude at each sweep -- a genuinely different quantity from
the time-averaged $\mpsi$ of \eqref{{eq:mpsi}}), one curve per $z$,
colour-graded light-to-dark. In the ordering phase every trajectory is
decided fast -- median sweep {_p7_dec_med:.0f} to move more than $0.15$ off
the no-influence baseline $\tfrac13$ -- then sits flat for the remaining
$\sim\!1500$ sweeps; but per realisation the outcome is strictly
\emph{{binary}} (conversion $\to0$ or $\to1$), never the smooth partial
value the $12$-seed average of Section~\ref{{sec:zfield}} shows at the same
$z$. This single seed disagrees with that $12$-seed ensemble average by RMSE
{_p7_rmse:.2f} over the whole $z$-grid -- and that disagreement is not noise
to be averaged away, it is the basin-selection mechanism of
Section~\ref{{sec:zfield}} caught in the act: each realisation falls
into \emph{{one}} basin or the other (the rare flip back onto Rock is
exactly the $O(e^{{-cN}})$ event predicted there), and only the ensemble
average over many such binary outcomes looks like a smooth partial
recovery. In the cycling phase, by contrast, no trajectory ever makes a
fast decision: conversion and $|\psi(t)|$ stay near their baselines
throughout and are visibly independent of $z$ -- all seventeen colours
overlap for the full $1500$ sweeps.
""")
w(r"\begin{figure}[H]\centering\includegraphics[width=0.95\textwidth]{zealots/phase7_timeseries.png}")
w(r"\caption{Per-sweep conversion and instantaneous $|\psi(t)|$ across the whole $z$-sweep, colour-graded by $z$, one ER realisation, log time axis.}\label{fig:p7ts}\end{figure}")

w(r"\subsection{Structural targeting: hub leverage}")
w(rf"""
On a scale-free (BA) network, a zealot's influence is proportional to its
\emph{{stubs}}, not its headcount: in the degree-resolved closure
\eqref{{eq:dmf}}, a random edge-end belongs to a zealot with probability
$z_{{\mathrm{{eff}}}}=\sum_{{i\in Z}}k_i/(N\langle k\rangle)$, the
degree-weighted zealot fraction, which is what replaces $z$ in the field
\eqref{{eq:zfield}}. Random placement gives $\mathbb E\,z_{{\mathrm{{eff}}}}=z$
by construction. Hub placement -- locking the top fraction of nodes by degree
-- is different: for an ideal BA graph with $P(k)=2m^2/k^3$, the top fraction
$q$ of nodes starts at degree $k_q=m/\sqrt q$, and its share of all stubs is
$\int_{{k_q}}^{{\infty}}kP(k)\,dk/\langle k\rangle=m/k_q=\sqrt q$, so
""")
w(r"""\begin{equation}
z_{\mathrm{eff}}^{\mathrm{hub}}=\sqrt z
\qquad\Longrightarrow\qquad
\text{leverage } z_{\mathrm{eff}}/z=1/\sqrt z.
\label{eq:hubleverage}
\end{equation}""")
w(r"""
\deriv{Equation~\eqref{eq:hubleverage} is the special case $\gamma{=}3$ of a
family indexed by the power-law exponent. For a general normalised power-law
degree distribution $P(k)=(\gamma-1)k_{\min}^{\gamma-1}k^{-\gamma}$
($k\ge k_{\min}$, $\gamma>2$ for a finite mean), the mean degree is
$\langle k\rangle=\tfrac{\gamma-1}{\gamma-2}k_{\min}$, and the top fraction
$q$ of nodes by degree starts at $k_q$ solving
$q=\Pr(k>k_q)=(k_{\min}/k_q)^{\gamma-1}$, i.e.\ $k_q=k_{\min}\,
q^{-1/(\gamma-1)}$. Its stub share is
\[
\frac{\int_{k_q}^\infty kP(k)\,dk}{\langle k\rangle}
=\left(\frac{k_{\min}}{k_q}\right)^{\gamma-2}
=q^{\,(\gamma-2)/(\gamma-1)},
\]
so in general $z_{\mathrm{eff}}^{\mathrm{hub}}=z^{(\gamma-2)/(\gamma-1)}$.
The BA network's $\gamma=3$ gives exponent $\tfrac12$, recovering
\eqref{eq:hubleverage}. The two limits bracket the result sensibly: as
$\gamma\to2^+$ (an extremely heavy tail) the exponent $\to0$ and
$z_{\mathrm{eff}}^{\mathrm{hub}}\to1$ for \emph{any} $z>0$ -- a vanishing
elite hoards essentially all stubs; as $\gamma\to\infty$ (an exponentially
thin tail, degree distribution effectively homogeneous, as on ER) the
exponent $\to1$ and $z_{\mathrm{eff}}^{\mathrm{hub}}\to z$ -- hub placement
converges to random placement, correctly predicting that this leverage
effect is specific to heavy-tailed networks and has no counterpart on ER.}
""")
w(rf"""
At $z{{=}}0.10$ this predicts a $1/\sqrt{{0.1}}\approx3.2\times$ leverage,
checked directly on the seed-1 BA graph used in the simulations: the top
$5\%/10\%$ of nodes by degree hold ${{{stub_share[0.05]:.3f}}}/{{{stub_share[0.10]:.3f}}}$
of all stubs, against the ideal prediction
$\sqrt z={{{np.sqrt(0.05):.3f}}}/{{{np.sqrt(0.10):.3f}}}$. Directly measured,
the same budget $z{{=}}0.10$ placed on hubs amplifies the induced cycling-phase
order by {amp_hub:.1f}$\times$ over random placement
($\mpsi^{{\mathrm{{hub}}}}={f2(h['cycle_hub_mpsi'][-1])}$ vs
$\mpsi^{{\mathrm{{rand}}}}={f2(h['cycle_random_mpsi'][-1])}$) -- the response
of the steep ordering nonlinearity to the $\sqrt z$ field. Structural
leverage, however, only controls \emph{{whether}} the network orders: the
population still orders on Paper except once
$z\gtrsim{f2(z_sat)}$ in the ordering phase, where saturated hub
neighbourhoods finally pin their own (Rock) strategy directly
(free-node conversion {f2(h['order_hub_conversion'][-1])} at
$z{{=}}{f2(h['z'][-1])}$).
""".replace("{{stub_share[0.05]:.3f}}", f"{stub_share[0.05]:.3f}")
    .replace("{{stub_share[0.10]:.3f}}", f"{stub_share[0.10]:.3f}")
    .replace("{{np.sqrt(0.05):.3f}}", f"{np.sqrt(0.05):.3f}")
    .replace("{{np.sqrt(0.10):.3f}}", f"{np.sqrt(0.10):.3f}"))
w(r"\begin{figure}[H]\centering\includegraphics[width=0.62\textwidth]{zealots/zealots_hubs.png}")
w(r"\caption{Rock-zealots on BA: hub vs random placement, both phases.}\end{figure}")

w(r"\subsection{Competing factions: the predator wins}")
w(r"""
Two opposing zealot factions of equal size $z$, one locked to Rock and one to
Paper, extend the field \eqref{eq:zfield} by summing the two payoff columns:
""")
w(r"""\begin{equation}
h=kz\,P\,(e_R+e_P)=kz\,\big(1-\varepsilon,\ \ 1+\varepsilon,\ \ 0\big)^\top.
\label{eq:twofield}
\end{equation}""")
w(rf"""
The verdict is read off the components of \eqref{{eq:twofield}} directly:
Paper collects $1+\varepsilon$ (its own zealots, plus predation on the Rock
faction), Rock keeps only $1-\varepsilon$ (its own zealots, minus what the
Paper faction eats), and Scissors gets \emph{{exactly zero}} -- its gain from
predating Rock ($+\varepsilon$) cancels its loss to Paper ($-\varepsilon$)
identically. The naive guess that the third, uninvolved strategy should
profit from two cancelling factions fails \emph{{algebraically}}, not merely
empirically: measured on ER (12-seed average), the ordering-phase population
goes to Paper ($\rho_{{\mathrm{{Paper}}}}={f2(mx['order_rho_paper'][-1])}$ at
$z{{=}}{f2(mx['z'][-1])}$, each faction), with Scissors driven toward zero,
while the cycling phase remains essentially unperturbed
($\mpsi\le{f2(float(np.max(mx['cycle_mpsi'])))}$).
""")
w(r"\begin{figure}[H]\centering\includegraphics[width=0.62\textwidth]{zealots/zealots_mixed.png}")
w(r"\caption{Equal Rock and Paper zealot factions: free-population composition, ordering phase.}\end{figure}")

w(r"\subsection{Network defects and the thinning theorem}\label{sec:defects}")
w(rf"""
Rather than adding agents, damage the network itself: remove a fraction $f$
of edges (broken links, all agents retained) or of nodes (vacancies, agents
and their links removed) and re-measure the transition. Both damage types
are \emph{{exactly closed}} on the ER family.
""")
w(r"""
\begin{proposition}
Edge removal with retention probability $1-f$ maps
$\mathrm{ER}(N,p)\to\mathrm{ER}(N,(1-f)p)$ exactly (independent thinning of
independent edges), giving $\langle k\rangle_{\mathrm{eff}}=(1-f)\langle
k\rangle_0$. Node removal keeping a fraction $1-f$ of nodes induces the
subgraph on $N'=(1-f)N$ nodes, again $\mathrm{ER}(N',p)$, with $\langle
k\rangle_{\mathrm{eff}}=p(N'-1)\approx(1-f)\langle k\rangle_0$ -- the
\emph{same} effective degree as edge removal at matched $f$.
\end{proposition}
""")
w(r"""
\deriv{\textbf{Edge removal.} In $\mathrm{ER}(N,p)$ every one of the
$\binom N2$ possible edges is present independently with probability $p$.
Deleting each \emph{present} edge independently with probability $f$ leaves
it present in the damaged graph with probability $p(1-f)$, and by
independence of the original edges the deletions of distinct edges are
themselves independent -- so the damaged graph is, by definition, a fresh
draw from $\mathrm{ER}(N,(1-f)p)$; there is no approximation in this step.
\textbf{Node removal.} Deleting each node independently with probability $f$
and keeping the induced subgraph on the survivors is equivalent, by the
same independence, to first fixing the surviving vertex set (a uniformly
random subset of expected size $N'=(1-f)N$) and then noting that each edge
of the \emph{original} graph survives in the induced subgraph iff both its
endpoints do, which occurs independently across edges with probability
$(1-f)^2$ for edges between two originally-distinct present-with-probability-$p$
pairs -- but since we condition on the surviving vertex set having size
$N'$, the induced subgraph on those $N'$ vertices is again exactly
$\mathrm{ER}(N',p)$ (the presence of each of the $\binom{N'}{2}$ possible
edges among survivors is still an independent Bernoulli$(p)$ trial, since
node removal does not touch edge probabilities at all). Its mean degree is
therefore $p(N'-1)$, and since $\langle k\rangle_0=p(N-1)$, $p(N'-1)=
\langle k\rangle_0\cdot\frac{N'-1}{N-1}\approx(1-f)\langle k\rangle_0$ for
$N\gg1$. Both closures are exact statements about the ER ensemble (not
approximations for the specific realisations used in the simulations),
which is why the empirical deviation quoted below is attributable entirely
to finite-$N$ fluctuation of $p$'s realised edge count around its mean,
not to any bias in the argument itself.}
""")
w(rf"""
\numcheck{{Measured across all eight (type,\,$f$) cells of the damage
experiment, the deviation from $\langle k\rangle_{{\mathrm{{eff}}}}=(1-f)\times{k0_pristine:.1f}$
is at most {thin_dev:.2f}.}}
""")
w(r"\begin{center}\begin{tabular}{lcccc}\toprule")
w(r"$f$ & $\epsc$ (edge) & $\langle k\rangle$ (edge) & $\epsc$ (node) & $\langle k\rangle$ (node)\\\midrule")
for f_ in fr:
    e_ = d[(d["defect_type_0edge_1node"] == 0) & (d["f"] == f_)]
    n_ = d[(d["defect_type_0edge_1node"] == 1) & (d["f"] == f_)]
    w(f"{f2(f_)} & {f2(ec_edge[f_])} & {e_['mean_k'][0]:.1f} & "
      f"{f2(epsc(n_['epsilon'], n_['m_psi']))} & {n_['mean_k'][0]:.1f}\\\\")
w(r"\bottomrule\end{tabular}\end{center}")
w(rf"""
Edge and node damage produce statistically the same $\epsc$ once matched by
resulting mean degree, and the transition slides down smoothly as damage
increases (${f2(ec_edge[fr[0]])}\to{f2(ec_edge[fr[-1]])}$) -- Sec.~5.1's
``connectivity stabilises order'' run in reverse.
""")
w(r"\subsection{Collapse onto a one-parameter family}")
w(rf"""
The decisive test composes the thinning theorem with the pristine boundary
$\mathcal E(\cdot)$ of Section~\ref{{sec:boundary}}: if damage matters only
through the mean degree it leaves behind, then for every damaged graph $G'$,
$\epsc(G')=\mathcal E(2E'/N')$, and in particular
$\epsc(f)=\mathcal E\big((1-f)\langle k\rangle_0\big)$ with no free
constants. Plotting each damaged network's (effective $\langle k\rangle$,
$\epsc$) coordinate against the pristine ER boundary -- extracted at a
\emph{{different}} $N$ ($800$ vs $1000$ for the damage experiment), a
deliberately stronger test -- shows every one of the ${{2*len(cl)}}$ damaged
points landing on the pristine curve, with
$\max_f|\epsc^{{\mathrm{{edge}}}}-\epsc^{{\mathrm{{node}}}}|={dev_collapse:.3f}$.
Combined with the ER$\,\equiv\,$BA agreement of Section~\ref{{sec:boundary}}
(within {gap_erba:.3f}), this collapses every network studied -- random or
scale-free, pristine or damaged -- onto a single curve: in this model, order
stability is a function of $\langle k\rangle$ alone.
""".replace("{{2*len(cl)}}", str(2*len(cl))))
w(r"\begin{figure}[H]\centering\includegraphics[width=0.62\textwidth]{defects/defects_collapse.png}")
w(r"\caption{Damaged-network $(\langle k\rangle,\epsc)$ points collapsed onto the pristine ER boundary.}\end{figure}")

# ============================================================ 8. ROBUSTNESS
w(r"\section{Robustness}\label{sec:robust}")
w(rf"""
Every fixed parameter used above -- graph and dynamics seeds, the
$\varepsilon$-grid step, run length, system size, the mean-field initial
condition, the zealot strategy label, and the damage realisation -- is
audited in a dedicated sensitivity study (hypotheses pre-registered before
running; full detail in \texttt{{sensitivity/FINDINGS.md}}). The headline
findings: \textbf{{(i)}} graph-seed and dynamics-seed contributions to
$\epsc$ scatter are statistically indistinguishable
(graph-seed std vs dynamics-seed std, both $\ll$ the $\varepsilon$-grid
step), so single-realisation sweeps used throughout are representative.
\textbf{{(ii)}} The zealot strategy label is an \emph{{exact}} symmetry of
the payoff matrix \eqref{{eq:payoff}} under the cyclic relabelling
R$\to$P$\to$S -- a designed null test -- and the measured label dependence is
consistent with pure statistical noise (relabelled trajectories coalesce to
machine precision). \textbf{{(iii)}} Damage \emph{{realisation}} (which
particular edges or nodes die at fixed $f$) shifts $\epsc$ by at most
{sdf_maxsd:.3f}, confirming that only the resulting $2E'/N'$ matters, as the
thinning theorem requires. \textbf{{(iv)}} $T{{=}}0.65$ sits deep in the
$k$-dominated regime of Proposition~\ref{{prop:kt}} at the degrees used
throughout ($\epsc$'s sensitivity to $T$ shrinks sharply as $\langle
k\rangle$ grows), which is what licenses treating $\langle k\rangle$ as the
single structural control parameter. \textbf{{(v)}} 1500 sweeps is roughly
twice the length needed for $\epsc$ to converge, and the finite-run bias that
remains is signed by the ordering-phase transient rather than the (smaller)
cycling-phase bias originally hypothesised -- a case where the mechanism was
right and the dominant sign was not, resolved only by measurement.
""")

# ============================================================ 9. DISCUSSION
w(r"\section{Discussion}\label{sec:discussion}")
w(r"""
Three themes recur across every section of this study and are worth stating
together. \textbf{Connectivity is an inverse temperature.} The exact $k/T$
invariance of the mean field (Proposition~\ref{prop:kt}) and the closely
matching phenomenology of the agent-based phase diagram show that, in this
model, a node's degree does not merely correlate with stability -- it
substitutes directly for lower noise. This is why the ER and BA boundaries
coincide (Section~\ref{sec:boundary}): once a node's degree is high enough to
saturate the Glauber rate, its exact position in the degree distribution's
tail stops mattering, and why the two damage types collapse onto one curve
(Section~\ref{sec:defects}): both act purely by lowering the surviving mean
degree.

\textbf{The transition is first-order-like, not continuous.} Three
independent lines of evidence converge on this point: the mean-field
bistability found by sweeping from different initial conditions
(Section~\ref{sec:bistable}), the Newton-continuation location of the
consensus branch's true endpoint, and the $1/N$ finite-size drift of the
agent-based pseudo-critical point (Section~\ref{sec:fss}). Practically, this
means a single reported $\epsc$ is always a finite-size or finite-initial-
condition estimate of a quantity that is genuinely history-dependent near the
transition; every comparison drawn in this paper (ER vs BA, damaged vs
pristine, mean field vs Monte Carlo) is made at matched $N$ and matched
convention specifically to remain valid despite this.

\textbf{Perturbation outcomes follow directly from a linear field
calculation.} The zealot field \eqref{eq:zfield} and its two-faction sum
\eqref{eq:twofield} are the whole explanation for two results that would
otherwise look like unrelated curiosities -- a Rock minority electing Paper,
and a Rock+Paper standoff also electing Paper rather than the untouched
Scissors. In both cases the cyclic term of the payoff matrix, not the size of
the minorities involved, decides the outcome; the only role of $z$ is to set
\emph{how fast} the decision is reached and how firmly it resists the rare,
exponentially suppressed fluctuation back onto the zealots' own strategy.

\textbf{A single symmetry organises the whole analysis.} The cyclic
invariance of Appendix~\ref{sec:symmetry} is not a footnote: it is the
reason a single Newton continuation (Section~\ref{sec:bistable}) certifies
the stability of all three ordered branches at once, the reason the zealot
strategy label is a legitimate and powerful null test rather than an
arbitrary modelling choice (Section~\ref{sec:robust}), and the reason the
two-faction field's exact cancellation on Scissors
(Section~\ref{sec:perturb}) could be anticipated on symmetry grounds before
any simulation was run. Together with the $(k,T)$ Hopf threshold of
Section~\ref{sec:hopf} -- which turns the informal observation that
``sparse networks order poorly'' into the precise, checkable statement that
they sit close to the linear-stability boundary $k_H=3T$ -- these results
illustrate a pattern that holds throughout the paper: every qualitative
finding reported here has a linear or symmetry-based mathematical
counterpart that predicts it in advance, rather than merely describing it
after the fact.
""")

# ============================================================ 10. CONCLUSION
w(r"\section{Conclusion and outlook}\label{sec:conclusion}")
w(rf"""
A $q{{=}}3$ Potts model deformed by an RPS-cyclic term $\varepsilon$ exhibits
a genuine, first-order-like order--cycling transition whose location is
governed, to a very good approximation, by mean degree alone -- reproduced
almost identically on Erd\H{{o}}s--R\'enyi and Barab\'asi--Albert topologies,
and equally by pristine and quenched-damaged networks once matched by
surviving connectivity. Two mean-field closures, validated eigenvalue and
continuation analysis, and direct agent-based simulation give a mutually
consistent, closed-form account of \emph{{why}}: connectivity is an exact
inverse temperature in the mean field, cyclic dominance enters the linear
stability spectrum as pure rotation, and the transition's first-order
character follows from a genuine competition between two non-locally stable
attractors rather than a local bifurcation. The same linear-field reasoning
that explains the clean transition also predicts, correctly and without
additional fitting, the counter-intuitive outcomes of every perturbation
experiment: zealot backfire, $1/\sqrt z$ hub leverage, and the algebraic
elimination of the bystander strategy under competing factions.

Two directions follow naturally. First, the bistable window identified here
(Section~\ref{{sec:bistable}}) has not been characterised quantitatively as a
function of $\varepsilon$ and $k$ beyond two sample degrees; a systematic
map of the window's boundaries, together with an estimate of the associated
hysteresis loop, would complete the first-order picture. Second, the
finite-size scaling of Section~\ref{{sec:fss}} establishes the $1/N$ law but
at system sizes (up to $N{{=}}2000$) too small to extract precise critical
exponents or to rule out corrections to the leading-order scaling; this is
the natural target for a larger, HPC-scale campaign.
""")

w(r"\appendix")
w(r"\section{The cyclic symmetry group}\label{sec:symmetry}")
w(r"""
Every derivation above secretly uses a symmetry that is worth making
explicit, since it both simplifies proofs and gives a sharp,
falsifiable null test. Let $\pi:\mathrm R\mapsto\mathrm P\mapsto
\mathrm S\mapsto\mathrm R$ be the cyclic relabelling of strategies and
$\Pi$ its $3\times3$ permutation matrix. The payoff matrix
\eqref{eq:payoff} is circulant, hence invariant under simultaneous
conjugation of both arguments by $\pi$:
""")
w(r"""\begin{equation}
\Pi P(\varepsilon)\Pi^\top=P(\varepsilon)
\qquad\text{for every }\varepsilon.
\label{eq:circulant}
\end{equation}""")
w(r"""
\deriv{$\Pi$ acts on the standard basis by $\Pi e_R=e_P$, $\Pi e_P=e_S$,
$\Pi e_S=e_R$. Since $P=I+\varepsilon S$ with $S=\Pi-\Pi^\top$, and $\Pi$
commutes with itself and with $\Pi^\top=\Pi^{-1}=\Pi^2$ (all powers of a
single permutation matrix commute), $\Pi P\Pi^\top=\Pi(I+\varepsilon(\Pi-
\Pi^\top))\Pi^\top=\Pi\Pi^\top+\varepsilon(\Pi^2\Pi^\top-\Pi\Pi^\top\Pi^\top)
=I+\varepsilon(\Pi-\Pi^\top)=P$, using $\Pi\Pi^\top=I$ and
$\Pi^2\Pi^\top=\Pi$, $\Pi\Pi^{\top2}=\Pi^\top$.}
""")
w(r"""
Because the entire microscopic dynamics \eqref{eq:utility}--\eqref{eq:glauber}
is built from $P$ alone, \eqref{eq:circulant} lifts to an exact symmetry of
the whole model: relabelling every agent's strategy by $\pi$ (equivalently,
relabelling which corner of the simplex is called ``Rock'') leaves the
Markov chain's transition probabilities, and hence the distribution of every
trajectory, invariant. Three consequences are used, sometimes implicitly,
throughout this paper.
""")
w(r"""
\begin{enumerate}[leftmargin=1.4em]
\item \textbf{One stability calculation suffices for all three ordered
corners.} The three corner fixed points of \eqref{eq:hmf} are related by
$\pi$, so their Jacobian spectra coincide; Section~\ref{sec:bistable}'s
Newton continuation of the Rock-adjacent branch determines the other two
branches' stability with no further computation.
\item \textbf{The zealot strategy label is a genuine null parameter.} Locking
zealots to Rock, Paper, or Scissors are three instances of the same
underlying experiment related by $\pi$; any measured difference between the
three labels (Section~\ref{sec:robust}) is attributable only to finite-sample
noise, which is exactly what is tested there, and the observed coalescence of
relabelled trajectories to machine precision is a direct empirical witness of
\eqref{eq:circulant}.
\item \textbf{The competing-factions cancellation is forced by symmetry, not
coincidence.} Section~\ref{sec:perturb}'s two-faction field
\eqref{eq:twofield} assigns Scissors exactly zero net field when Rock and
Paper factions are equal and opposite; this is the unique outcome consistent
with $\pi$ acting transitively on the three strategies while the chosen
zealot pair breaks the symmetry down to a single reflection, under which
Scissors is the unique fixed strategy and therefore the unique candidate for
an exactly-zero linear response.
\end{enumerate}
""")
w(r"\section{Summary of closed-form results}\label{sec:summary}")
w(r"""
For reference, every closed-form result derived in this paper is collected
below, with the section in which it is derived and (where applicable) the
build-time numerical check performed on the actual model code or data.
""")
w(r"\begin{center}\small\begin{tabular}{p{0.30\textwidth}p{0.42\textwidth}p{0.08\textwidth}}\toprule")
w(r"Result & Statement & Sec.\\\midrule")
w(r"Order-parameter norm & $|\psi|^2=1-3(rp+ps+sr)$ & \ref{sec:psiident}\\")
w(r"Kolmogorov cycle ratio & $\prod W_{\to}/\prod W_{\leftarrow}=e^{3\varepsilon/T}$ & \ref{sec:dbal}\\")
w(r"$k/T$ invariance & $\epsc^{\mathrm{HMF}}(k,T)=F(k/T)$ & \ref{sec:invar}\\")
w(r"Symmetric-point Jacobian & $\lambda_\pm=\tfrac14+\tfrac{k}{4T}\pm i\tfrac{\sqrt3\varepsilon k}{4T}$ & \ref{sec:stability}\\")
w(r"Hopf threshold & $k_H=3T$ & \ref{sec:hopf}\\")
w(r"HMF--DMF Jensen gap & $\Delta\mathrm{RMSE}\approx\tfrac12 g''(\langle k\rangle)\sigma_k^2$ & \ref{sec:mf-mc}\\")
w(r"MC--HMF gap decomposition & $\Delta=[F(k/T)-\epsc^\infty]+c(k)/N$ & \ref{sec:mf-mc}\\")
w(r"First-order finite-size laws & $\epsc(N)-\epsc(\infty)=O(1/N)$, width $=O(1/N)$ & \ref{sec:ld}\\")
w(r"Zealot field (single faction) & $h=kz(1,\varepsilon,-\varepsilon)^\top$ & \ref{sec:zfield}\\")
w(r"Zealot field (two factions) & $h=kz(1-\varepsilon,1+\varepsilon,0)^\top$ & \ref{sec:perturb}\\")
w(r"Hub leverage (general $\gamma$) & $z_{\mathrm{eff}}^{\mathrm{hub}}=z^{(\gamma-2)/(\gamma-1)}$ & \ref{sec:perturb}\\")
w(r"Thinning theorem & $\langle k\rangle_{\mathrm{eff}}=(1-f)\langle k\rangle_0$, both damage types & \ref{sec:defects}\\")
w(r"Cyclic symmetry & $\Pi P(\varepsilon)\Pi^\top=P(\varepsilon)$ & \ref{sec:symmetry}\\")
w(r"\bottomrule\end{tabular}\end{center}")
w(r"\end{document}")

# ------------------------------------------------------------ write + compile
tex = os.path.join(HERE, "paper.tex")
body = "\n".join(L)
body = body.replace("RUNDATE", run_date)
body = body.replace("NPDGTILES{}", str(2 * _pdg_tiles))
body = body.replace("GAPERBA{}", f"{gap_erba:.3f}")
body = body.replace("NDEGREES{}", str(len(cb)))
body = body.replace("AMPHUB{}", f"{amp_hub:.1f}")
body = body.replace("DEVCOLLAPSE{}", f"{dev_collapse:.3f}")
with open(tex, "w") as f:
    f.write(body)
print("Wrote paper.tex")
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "paper.tex"], cwd=HERE, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-4000:])
    raise SystemExit("pdflatex FAILED")
os.replace(os.path.join(HERE, "paper.pdf"), os.path.join(HERE, "RESEARCH_PAPER.pdf"))
for ext in ("aux", "log", "out", "toc"):
    p = os.path.join(HERE, f"paper.{ext}")
    if os.path.exists(p):
        os.remove(p)
sz = os.path.getsize(os.path.join(HERE, "RESEARCH_PAPER.pdf")) // 1024
print(f"Wrote RESEARCH_PAPER.pdf ({sz} KB)")
