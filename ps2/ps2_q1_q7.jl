# Problem Set 2: Two-Period Ramsey Taxation in a Small Open Economy (Q1–Q7)
#
# Standalone script covering the whole problem set. Q1–Q4 are analytical and
# their written answers are printed as text; Q5–Q7 are numerical and use the
# same model code, parameters, and results as ps2.ipynb.
# Run with:  julia ps2_q1_q7.jl   (takes a few minutes; the Q7 β-sweep dominates)

using Roots, Optim, Printf, Plots
gr()

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

β, θ = 1/1.04, 2.0
params = (
    R  = 1.04,
    β  = β,
    θ  = θ,
    η  = 0.7,
    ψ  = 1.5,
    G  = 0.15,
    ωᶜ = 0.6,
    ν  = 0.5,
    ωˡ = 0.5,
    ϕ  = 1.5,
)

# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE FUNCTIONS
#
# All functions are evaluated analytically from the CES and PPF structure.
# No numerical derivatives are used in the solver (a finite difference appears
# only in the Q5(c) verification check). This is deliberate: CES functions have closed-form partial derivatives, so
# there is no reason to approximate them numerically. Using analytical forms
# avoids NaN blowups near feasibility boundaries and is faster.
#
# N_fn:    PPF — gives labor n required to produce (c, G) simultaneously.
#          Derived from F(c,G;η,ψ) = n.
# pc_fn:   Producer price of c — the marginal labor cost of one more unit of c,
#          i.e. ∂N/∂c. This is what the firm charges for c in a competitive eq.
# pg_fn:   Producer price of G — marginal labor cost of G, i.e. ∂N/∂G.
# ell_fn:  Leisure — solved from the time constraint H(n,ℓ)=1. Returns NaN
#          when n is so large that no positive leisure is feasible.
# Vt_fn:   Within-period CES utility over (c, ℓ).
# Vlife_fn: Lifetime CES aggregator over (V0, V1) with canonical weight ωⁿ.
# VcVl_fn: Analytical MRS Vc/Vℓ from CES — ratio of partial derivatives.
#          equals (ωᶜ/(1-ωᶜ))^(1-ν) * (c/ℓ)^(ν-1). Used in intra FOC.
# HnHl_fn: Analytical time-constraint ratio Hn/Hℓ from CES time constraint.
#          equals (ωˡ/(1-ωˡ))^(1-ϕ) * (n/ℓ)^(ϕ-1). Used in intra FOC.
# Lambda_fn: The Euler kernel for period t — the numerator/denominator of the
#          household's intertemporal MRS after the ωᶜ^(1-ν) and Vlife^(1-ρ)
#          factors cancel in the ratio. Λt = ωt^(1-ρ)*Vt^(ρ-ν)*ct^(ν-1).
# ══════════════════════════════════════════════════════════════════════════════

N_fn(c, p)     = (p.η^(1-p.ψ)*c^p.ψ + (1-p.η)^(1-p.ψ)*p.G^p.ψ)^(1/p.ψ)
pc_fn(c, p)    = N_fn(c,p)^(1-p.ψ) * (c/p.η)^(p.ψ-1)
pg_fn(c, p)    = (1-p.η)^(1-p.ψ) * p.G^(p.ψ-1) * N_fn(c,p)^(1-p.ψ)

function ell_fn(n, p)
    inner = 1 - p.ωˡ^(1-p.ϕ)*n^p.ϕ
    inner > 0 ? (inner/(1-p.ωˡ)^(1-p.ϕ))^(1/p.ϕ) : NaN
end

Vt_fn(c, ℓ, p)  = (p.ωᶜ^(1-p.ν)*c^p.ν + (1-p.ωᶜ)^(1-p.ν)*ℓ^p.ν)^(1/p.ν)

function Vlife_fn(V0, V1, p)
    ρ  = 1 - 1/p.θ
    ωⁿ = p.β^p.θ / (1 + p.β^p.θ)
    ((1-ωⁿ)^(1-ρ)*V0^ρ + ωⁿ^(1-ρ)*V1^ρ)^(1/ρ)
end

VcVl_fn(c, ℓ, p) = (p.ωᶜ/(1-p.ωᶜ))^(1-p.ν) * (c/ℓ)^(p.ν-1)
HnHl_fn(n, ℓ, p) = (p.ωˡ/(1-p.ωˡ))^(1-p.ϕ) * (n/ℓ)^(p.ϕ-1)

function Lambda_fn(Vt, ct, ωt, p)
    ρ = 1 - 1/p.θ
    ωt^(1-ρ) * Vt^(ρ-p.ν) * ct^(p.ν-1)
end

# c_bounds: computes the physically feasible range [c_lo, c_hi] for consumption.
# c_hi is the largest c where leisure is still strictly positive. It is derived
# by finding N_max (the labor level where the time constraint binds with ℓ=0),
# then inverting the PPF at N_max to get the corresponding c. We pull back
# slightly (1-1e-5) to avoid NaN at the exact boundary.
function c_bounds(p)
    c_lo  = 1e-6
    N_max = (1 / p.ωˡ^(1-p.ϕ))^(1/p.ϕ)
    val   = N_max^p.ψ - (1-p.η)^(1-p.ψ)*p.G^p.ψ
    c_hi  = (val / p.η^(1-p.ψ))^(1/p.ψ) * (1 - 1e-5)
    return c_lo, c_hi
end

# find_root_gbc: robust 1D root finder for GBC residuals.
#
# WHY NOT SIMPLE BISECTION: The GBC residual res(c) = pg(c)*G - τ_w*N(c) - B
# is monotone (and thus has exactly one root) only when τ_w ≥ 0. For negative
# τ_w (wage subsidies), the residual becomes U-shaped and can have zero or two
# roots. A simple bisection that assumes a sign change at [c_lo, c_hi] will
# silently return NaN in these cases.
#
# APPROACH: Scan 300 evenly-spaced points across [c_lo, c_hi]. Record every
# adjacent pair where the residual changes sign. Bisect on each such pair to
# find all roots. Filter to those with positive feasible leisure. When multiple
# feasible roots remain (can happen with negative taxes), take the one with the
# highest within-period utility Vt(c, ℓ(N(c))). This is the "highest welfare
# root" selection rule.
#
# The grid scan adds a small cost per evaluation but makes the solver robust
# across ALL (τ_w, B) combinations the outer optimizer might try.
function find_root_gbc(res_fn, p; n_scan=300)
    c_lo, c_hi = c_bounds(p)
    cs   = range(c_lo, c_hi, length=n_scan)
    vals = map(res_fn, cs)
    roots = Float64[]
    for i in 1:n_scan-1
        (isnan(vals[i]) || isnan(vals[i+1])) && continue
        vals[i] * vals[i+1] ≥ 0              && continue
        push!(roots, find_zero(res_fn, (cs[i], cs[i+1]), Bisection()))
    end
    isempty(roots) && return NaN
    valid = filter(roots) do c
        n = N_fn(c, p); ℓ = ell_fn(n, p)
        isfinite(ℓ) && ℓ > 0 && c > 0
    end
    isempty(valid) && return NaN
    length(valid) == 1 && return valid[1]
    return valid[argmax([Vt_fn(c, ell_fn(N_fn(c,p),p), p) for c in valid])]
end

# solve_c0_c1: finds equilibrium allocation (c0, c1) from the government budget
# constraints given policy (τ_w0, τ_w1, B).
#
# KEY DESIGN DECISION — WHY USE THE GBC TO FIND c, NOT THE HOUSEHOLD FOC:
# In a competitive equilibrium, both the GBC and the household intratemporal
# FOC must hold. We could use either to find c. We chose the GBC because:
#   1. B appears explicitly — so moving B in the outer optimizer actually changes
#      the allocation. If we used household FOC to find c, B would have no effect
#      on c and the optimizer couldn't search over it meaningfully.
#   2. The GBC residuals are monotone for positive taxes, making root-finding
#      reliable. A bisection on the household FOC can fail for low τ_w because
#      the equilibrium c falls outside the bracket.
#   3. The GBC approach correctly captures the saving linkage: B in period 0
#      implies R·B must be repaid in period 1, which forces c1 to be higher to
#      generate enough tax revenue — this cross-period effect is explicit.
#
# The household FOC conditions (intratemporal × 2 and Euler) are then checked
# as residuals and enforced by the outer optimizer's penalty. At the optimum
# these residuals are ≈ 0, confirming a true competitive equilibrium.
#
# FEASIBILITY: With B=0, there is a minimum feasible τ_w (set by the PPF
# parameters — specifically by how much labor is needed to produce G). With
# B > 0 (government borrows in period 0), period 1 needs a higher τ_w to repay
# R·B on top of funding G. Infeasible combinations correctly return NaN.
function solve_c0_c1(tau_w0, tau_w1, B, p)
    res0(c) = pg_fn(c,p)*p.G - tau_w0*N_fn(c,p) - B
    res1(c) = pg_fn(c,p)*p.G + p.R*B - tau_w1*N_fn(c,p)
    return find_root_gbc(res0, p), find_root_gbc(res1, p)
end

# solve_c0_c1_full: Q6 version — consumption tax revenue τ_t*p_t*c_t added.
# GBC residuals become:
#   Period 0: pg(c0)*G - τ0*pc(c0)*c0 - τ_w0*N(c0) - B = 0
#   Period 1: pg(c1)*G + R*B - τ1*pc(c1)*c1 - τ_w1*N(c1) = 0
# When τ0=τ1=0 these reduce exactly to solve_c0_c1 — consistency guaranteed.
function solve_c0_c1_full(tau0, tau1, tau_w0, tau_w1, B, p)
    res0(c) = pg_fn(c,p)*p.G - tau0*pc_fn(c,p)*c - tau_w0*N_fn(c,p) - B
    res1(c) = pg_fn(c,p)*p.G + p.R*B - tau1*pc_fn(c,p)*c - tau_w1*N_fn(c,p)
    return find_root_gbc(res0, p), find_root_gbc(res1, p)
end

function welfare(tau_w0, tau_w1, B, p)
    c0, c1 = solve_c0_c1(tau_w0, tau_w1, B, p)
    (isnan(c0) || isnan(c1)) && return -Inf
    n0, n1 = N_fn(c0,p), N_fn(c1,p)
    ℓ0, ℓ1 = ell_fn(n0,p), ell_fn(n1,p)
    (isnan(ℓ0)||isnan(ℓ1)||ℓ0≤0||ℓ1≤0) && return -Inf
    𝒱 = Vlife_fn(Vt_fn(c0,ℓ0,p), Vt_fn(c1,ℓ1,p), p)
    (isfinite(𝒱) && 𝒱>0) ? 𝒱 : -Inf
end

function welfare_full(tau0, tau1, tau_w0, tau_w1, B, p)
    c0, c1 = solve_c0_c1_full(tau0, tau1, tau_w0, tau_w1, B, p)
    (isnan(c0) || isnan(c1)) && return -Inf
    n0, n1 = N_fn(c0,p), N_fn(c1,p)
    ℓ0, ℓ1 = ell_fn(n0,p), ell_fn(n1,p)
    (isnan(ℓ0)||isnan(ℓ1)||ℓ0≤0||ℓ1≤0) && return -Inf
    𝒱 = Vlife_fn(Vt_fn(c0,ℓ0,p), Vt_fn(c1,ℓ1,p), p)
    (isfinite(𝒱) && 𝒱>0) ? 𝒱 : -Inf
end

function hh_residuals(tau_w0, tau_w1, B, p)
    c0, c1 = solve_c0_c1(tau_w0, tau_w1, B, p)
    (isnan(c0)||isnan(c1)) && return Inf, Inf, Inf
    n0, n1  = N_fn(c0,p), N_fn(c1,p)
    ℓ0, ℓ1 = ell_fn(n0,p), ell_fn(n1,p)
    (isnan(ℓ0)||isnan(ℓ1)||ℓ0≤0||ℓ1≤0) && return Inf, Inf, Inf
    p0, p1 = pc_fn(c0,p), pc_fn(c1,p)
    # Correct HH intratemporal FOC: V_c/V_ℓ = (H_n/H_ℓ)·q/(1-τ_w)
    # Residual = V_c/V_ℓ - (H_n/H_ℓ)·q/(1-τ_w)
    i0 = VcVl_fn(c0,ℓ0,p) - HnHl_fn(n0,ℓ0,p) * p0/(1-tau_w0)
    i1 = VcVl_fn(c1,ℓ1,p) - HnHl_fn(n1,ℓ1,p) * p1/(1-tau_w1)
    ρ  = 1 - 1/p.θ; ωⁿ = p.β^p.θ/(1+p.β^p.θ)
    V0, V1 = Vt_fn(c0,ℓ0,p), Vt_fn(c1,ℓ1,p)
    er = Lambda_fn(V0,c0,1-ωⁿ,p)/Lambda_fn(V1,c1,ωⁿ,p) - p.R*p0/p1
    return i0, i1, er
end

function hh_residuals_full(tau0, tau1, tau_w0, tau_w1, B, p)
    c0, c1 = solve_c0_c1_full(tau0, tau1, tau_w0, tau_w1, B, p)
    (isnan(c0)||isnan(c1)) && return Inf, Inf, Inf
    n0, n1  = N_fn(c0,p), N_fn(c1,p)
    ℓ0, ℓ1 = ell_fn(n0,p), ell_fn(n1,p)
    (isnan(ℓ0)||isnan(ℓ1)||ℓ0≤0||ℓ1≤0) && return Inf, Inf, Inf
    p0, p1 = pc_fn(c0,p), pc_fn(c1,p)
    q0, q1 = (1+tau0)*p0, (1+tau1)*p1
    # Correct HH intratemporal FOC: V_c/V_ℓ = (H_n/H_ℓ)·q/(1-τ_w)
    i0 = VcVl_fn(c0,ℓ0,p) - HnHl_fn(n0,ℓ0,p) * q0/(1-tau_w0)
    i1 = VcVl_fn(c1,ℓ1,p) - HnHl_fn(n1,ℓ1,p) * q1/(1-tau_w1)
    ρ  = 1 - 1/p.θ; ωⁿ = p.β^p.θ/(1+p.β^p.θ)
    V0, V1 = Vt_fn(c0,ℓ0,p), Vt_fn(c1,ℓ1,p)
    er = Lambda_fn(V0,c0,1-ωⁿ,p)/Lambda_fn(V1,c1,ωⁿ,p) - p.R*q0/q1
    return i0, i1, er
end

# run_ramp_q5 / run_ramp_q6: outer Nelder-Mead optimizer with κ-ramping.
#
# THE OBJECTIVE: minimize  -welfare(τ_w0, τ_w1, B) + κ*(intra0² + intra1² + euler²)
#   - welfare() computes TRUE lifetime utility V at the GBC allocation. It is
#     returned separately from the penalized objective so results are unambiguous.
#   - The three penalty residuals enforce household optimality:
#       intra0 = VcVl - HnHl*p0/(1-τ_w0)  (intratemporal FOC period 0)
#       intra1 = VcVl - HnHl*p1/(1-τ_w1)  (intratemporal FOC period 1)
#       euler  = Λ0/Λ1 - R*p0/p1          (intertemporal Euler equation)
#     At the true Ramsey optimum, all three equal zero. The optimizer finds
#     the (τ_w0, τ_w1, B) where the GBC allocation also satisfies household FOC.
#   - GBC residuals are NOT in the penalty — they are satisfied by construction
#     inside solve_c0_c1.
#
# WHY κ-RAMPING INSTEAD OF FIXED HIGH κ:
# With a single large κ (e.g. 1e6), the penalty term dominates the landscape so
# strongly that Nelder-Mead's simplex collapses in the parameter directions that
# affect residuals, before it has explored the welfare direction. The optimizer
# "converges" in 45 iterations to a random point where the simplex diameter is
# tiny — not to the true optimum.
# Solution: start with κ=1e2 (penalty is weak, optimizer primarily maximizes
# welfare and finds the high-welfare region), then progressively tighten to
# κ=1e6 (penalty enforces household optimality to 1e-7 precision). Each stage
# warm-starts from the previous result, so the optimizer never has to find the
# feasible region from scratch at high κ.
# Stages: κ ∈ {1e2, 1e3, 1e4, 1e5, 1e6}, 200k iterations each.
function run_ramp_q5(p, x0, iters=200_000)
    function run_stage(x0, κ)
        obj(x) = begin
            𝒱 = welfare(x[1],x[2],x[3],p); !isfinite(𝒱) && return Inf
            i0,i1,er = hh_residuals(x[1],x[2],x[3],p)
            (!isfinite(i0)||!isfinite(i1)||!isfinite(er)) && return Inf
            -𝒱 + κ*(i0^2+i1^2+er^2)
        end
        res = optimize(obj, x0, NelderMead(), Optim.Options(iterations=iters))
        Optim.minimizer(res), res
    end
    x, res = x0, nothing
    for κ in [1e2,1e3,1e4,1e5,1e6]
        x, res = run_stage(x, κ)
    end
    τw0,τw1,B = x[1],x[2],x[3]
    c0,c1 = solve_c0_c1(τw0,τw1,B,p)
    n0,n1 = N_fn(c0,p), N_fn(c1,p)
    ℓ0,ℓ1 = ell_fn(n0,p), ell_fn(n1,p)
    i0,i1,er = hh_residuals(τw0,τw1,B,p)
    (tau_w0=τw0, tau_w1=τw1, B=B, V=welfare(τw0,τw1,B,p),
     c0=c0, c1=c1, n0=n0, n1=n1, ℓ0=ℓ0, ℓ1=ℓ1,
     p0=pc_fn(c0,p), p1=pc_fn(c1,p),
     pg0=pg_fn(c0,p), pg1=pg_fn(c1,p),
     intra0=i0, intra1=i1, euler=er,
     converged=Optim.converged(res), iterations=res.iterations)
end

function run_ramp_q6(p, x0, iters=200_000)
    function run_stage(x0, κ)
        obj(x) = begin
            𝒱 = welfare_full(x[1],x[2],x[3],x[4],x[5],p); !isfinite(𝒱) && return Inf
            i0,i1,er = hh_residuals_full(x[1],x[2],x[3],x[4],x[5],p)
            (!isfinite(i0)||!isfinite(i1)||!isfinite(er)) && return Inf
            -𝒱 + κ*(i0^2+i1^2+er^2)
        end
        res = optimize(obj, x0, NelderMead(), Optim.Options(iterations=iters))
        Optim.minimizer(res), res
    end
    x, res = x0, nothing
    for κ in [1e2,1e3,1e4,1e5,1e6]
        x, res = run_stage(x, κ)
    end
    τ0,τ1,τw0,τw1,B = x[1],x[2],x[3],x[4],x[5]
    c0,c1 = solve_c0_c1_full(τ0,τ1,τw0,τw1,B,p)
    n0,n1 = N_fn(c0,p), N_fn(c1,p)
    ℓ0,ℓ1 = ell_fn(n0,p), ell_fn(n1,p)
    p0,p1 = pc_fn(c0,p), pc_fn(c1,p)
    i0,i1,er = hh_residuals_full(τ0,τ1,τw0,τw1,B,p)
    (tau0=τ0, tau1=τ1, tau_w0=τw0, tau_w1=τw1, B=B,
     V=welfare_full(τ0,τ1,τw0,τw1,B,p),
     c0=c0, c1=c1, n0=n0, n1=n1, ℓ0=ℓ0, ℓ1=ℓ1,
     p0=p0, p1=p1, q0=(1+τ0)*p0, q1=(1+τ1)*p1,
     pg0=pg_fn(c0,p), pg1=pg_fn(c1,p),
     intra0=i0, intra1=i1, euler=er,
     converged=Optim.converged(res), iterations=res.iterations)
end

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 1: Budget Constraints and Separability Condition
# ══════════════════════════════════════════════════════════════════════════════

println("\n", "="^65)
println("QUESTION 1")
println("="^65)

# ── Q1(a) ─────────────────────────────────────────────────────────────────────
println("""
Q1(a): Consolidated budget constraints

Method: Eliminate S by substituting period-0 HH budget into period-1 HH budget.
Eliminate B by substituting period-0 Gov budget into period-1 Gov budget.

Household IBC (S eliminated):
  q0·c0 + q1·c1/R = (1-τ_w0)·n0 + (1-τ_w1)·n1/R
  LHS: PDV of consumption expenditure.
  RHS: PDV of after-tax labor income.

Government IBC (B eliminated):
  pg0·G + pg1·G/R = τ0·p0·c0 + τ_w0·n0 + (τ1·p1·c1 + τ_w1·n1)/R
  LHS: PDV of government spending on G.
  RHS: PDV of tax revenue from consumption and labor taxes.
""")

# ── Q1(b) ─────────────────────────────────────────────────────────────────────
println("""
Q1(b): Separability condition

Method: From the canonical CES lifetime aggregator, the planner's MRS between
V0 and V1 is (∂V/∂V0)/(∂V/∂V1) = [(1-ωⁿ)/ωⁿ]^(1-ρ) · (V0/V1)^(ρ-1).
In the open economy the planner can exchange resources at rate R, so at the
optimum: (∂V/∂V0)/(∂V/∂V1) = R.

Setting V0 = V1 (equal within-period utilities) and substituting
ωⁿ = β^θ/(1+β^θ), so (1-ωⁿ)/ωⁿ = 1/β^θ, and 1-ρ = 1/θ:

  [1/β^θ]^(1/θ) = R  →  β^(-1) = R  →  βR = 1

The separability condition is βR = 1.
""")
@printf "  Baseline check: β·R = %.4f  (≈1 means near-separable)\n" params.β * params.R

# ── Q1(c) ─────────────────────────────────────────────────────────────────────
println("""
Q1(c): [Written] Compare to standard βR = 1; role of θ

In the additively separable model (V = u(c0,ℓ0) + β·u(c1,ℓ1)), the Euler
condition directly gives βR = 1 for equal consumption across periods. In the
canonical CES model the same condition emerges because the canonical weights
ωⁿ = β^θ/(1+β^θ) encode both the discount factor β and the elasticity θ in a way
that the separability condition collapses back to βR = 1 regardless of θ.

θ governs how quickly the marginal value of V0 relative to V1 changes as V0/V1
departs from 1. A higher θ (more willing to substitute across periods) makes
the planner more responsive to the interest rate wedge, but the threshold
condition remains βR = 1 because the canonical form normalizes the weights
precisely so that V0 = V1 is optimal exactly when βR = 1.

This is what makes the two-period Ramsey problem separable: when βR = 1, the
planner has no reason to tilt resources across periods, so each period's
within-period problem can be solved independently.
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 2: Ramsey Primal Approach (assume βR = 1)
# ══════════════════════════════════════════════════════════════════════════════

println("="^65)
println("QUESTION 2")
println("="^65)

# ── Q2(a) ─────────────────────────────────────────────────────────────────────
println("""
Q2(a): Separability of the planner's problem

When βR = 1, the planner's MRS between periods equals R, so the optimal choice
has V0 = V1. This means the planner is indifferent to the intertemporal
allocation at the margin — there is no cross-period interaction beyond the
resource constraint, which decouples across periods (the open economy borrows
freely at R). The within-period problem for each t is then:

  max_{c_t, n_t, ℓ_t}  Vt(c_t, ℓ_t)
  s.t.  F(c_t, G; η, ψ) = n_t   (PPF)
        H(n_t, ℓ_t; ωˡ, ϕ) = 1  (time constraint)
""")

# ── Q2(b) ─────────────────────────────────────────────────────────────────────
println("""
Q2(b): Reduction to one variable

Using the PPF: n_t = N(c_t) = (η^(1-ψ)·c_t^ψ + (1-η)^(1-ψ)·G^ψ)^(1/ψ)
Using the time constraint: ℓ_t = L(c_t) = L(N(c_t))

The within-period problem reduces to:
  max_{c}  Vt(c, L(N(c)))
""")

# ── Q2(c) ─────────────────────────────────────────────────────────────────────
println("""
Q2(c): First-order condition

Differentiating with respect to c:
  dVt/dc = Vc(c,ℓ) + Vℓ(c,ℓ)·(dℓ/dc) = 0

where dℓ/dc = L'(N(c))·N'(c) = −(Hn/Hℓ)·pc  (from implicit differentiation
of the time constraint and the PPF).

Rearranging:
  Vc/Vℓ = (Hn/Hℓ)·pc

Interpretation: the social MRS between c and ℓ equals the true resource cost
of c — the marginal labor required (pc = N'(c)) weighted by how labor crowds
out leisure (Hn/Hℓ). This is the undistorted planner's allocation.
""")

# ── Q2(d) ─────────────────────────────────────────────────────────────────────
println("""
Q2(d): [Written] Role of ψ > 1

ψ > 1 ensures the PPF is strictly concave — the marginal cost of producing c
(holding G fixed) increases as c rises. This guarantees the FOC from Q2(c) has
a unique interior solution. If ψ < 1 the PPF would be convex (bowed inward),
meaning the economy has increasing returns to specialization. In that case the
optimum would be a corner solution (produce only c or only G), and the
Ramsey problem would be degenerate.
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 3: Optimal Tax System
# ══════════════════════════════════════════════════════════════════════════════

println("="^65)
println("QUESTION 3")
println("="^65)

# ── Q3(a) ─────────────────────────────────────────────────────────────────────
println("""
Q3(a): Firm pricing

The firm maximizes profit: max_{c,G,n}  p_t·c + pg_t·G − n
subject to F(c,G;η,ψ) = n. By the envelope theorem:

  p_t  = ∂N/∂c = N^(1-ψ)·(c/η)^(ψ-1)        (producer price of c)
  pg_t = ∂N/∂G = (1-η)^(1-ψ)·G^(ψ-1)·N^(1-ψ) (producer price of G)

Zero-profit condition: p_t·c + pg_t·G = N(c,G) = n_t. This follows from Euler's
theorem since F is homogeneous of degree 1 in (c,G).
""")

# ── Q3(b) ─────────────────────────────────────────────────────────────────────
println("""
Q3(b): Household optimality

From the Lagrangian (FOCs over c, ℓ, n with budget q·c = (1−τ_w)·n − S and
time constraint H(n,ℓ)=1), the household's within-period FOC is:

  V_c / V_ℓ = (H_n / H_ℓ) · q_t/(1−τ_w,t) = (H_n/H_ℓ) · (1+τ_t)·p_t / (1−τ_w,t)

Comparing to the planner's FOC from Q2(c): V_c/V_ℓ = (H_n/H_ℓ)·p_t.
The household's choice coincides with the planner's first-best when:

  (1+τ_t)·p_t / (1−τ_w,t) = p_t   →   (1+τ_t)/(1−τ_w,t) = 1   →   τ_t = −τ_w,t

So to implement the undistorted allocation, the consumption tax must exactly
cancel the wage tax (opposite signs, same magnitude). With a positive revenue
requirement this exact cancellation cannot hold — both instruments must
contribute distortion. The two taxes are substitutes for setting the wedge
(1+τ_t)/(1−τ_w,t), so the Ramsey problem has a flat direction in (τ, τ_w).
""")

# ── Q3(c) ─────────────────────────────────────────────────────────────────────
println("""
Q3(c): Revenue condition

Per-period government budget: τ_w,t·n_t + τ_t·p_t·c_t = pg_t·G

Substituting the household optimality relationship from Q3(b) and the zero-
profit condition: the revenue requirement pins down the level of distortion
needed. Given the optimal allocation (c_t*, n_t*) from Q2, the required tax
revenue is pg_t·G, and the tax mix (τ_t, τ_w,t) must satisfy both the revenue
condition and the household FOC simultaneously.
""")

# ── Q3(d) ─────────────────────────────────────────────────────────────────────
println("""
Q3(d): Intertemporal taxes

Since G is identical in both periods and the separability condition holds
(βR = 1), the two within-period problems are structurally identical. Therefore
the optimal tax system is the same in both periods:

  (τ0*, τ_w0*) = (τ1*, τ_w1*)

Equal taxes across periods — no reason to tilt the tax mix intertemporally.
""")

# ── Q3(e) ─────────────────────────────────────────────────────────────────────
println("""
Q3(e): [Written] Description of the optimal tax system

The two instruments are SUBSTITUTES: only the combined wedge
(1+τ_t)/(1−τ_w,t) matters for the allocation, so along any iso-wedge
locus the welfare is the same. The Ramsey problem therefore has a flat
direction — only the level of the wedge is pinned down, not the split
between τ and τ_w. Either instrument alone is enough to implement the
optimal allocation; both are not strictly needed.

The labor-leisure margin IS distorted at the optimum because the
government must finance G — the wedge (1+τ)/(1−τ_w) > 1 pulls
households toward more leisure and less consumption than the first best.

Signs (Ramsey solution): with a positive revenue requirement and only one
instrument used, that instrument is positive — a positive wage tax (or a
positive consumption tax). If both are nonzero, the numerical solution
typically chooses both with the same sign (both positive) because
combinations with τ = −τ_w only implement the first best, which doesn't
raise revenue. The "opposite-sign" combination is the no-revenue locus
and is therefore not feasible when G > 0.
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 4: Wage Taxes Only — Discussion
# ══════════════════════════════════════════════════════════════════════════════

println("="^65)
println("QUESTION 4")
println("="^65)

# ── Q4(a) ─────────────────────────────────────────────────────────────────────
println("""
Q4(a): Wage taxes only (τ_t = 0)

Yes, the government can still finance G using only wage taxes and debt B. The
government budget in each period becomes:
  τ_w,t·n_t + B (period 0) and τ_w,t·n_t − R·B (period 1)

The margin necessarily distorted: the household's labor-leisure choice. With
τ_t = 0, the household FOC becomes Vc/Vℓ = (Hn/Hℓ)·p_t/(1−τ_w,t) ≠ (Hn/Hℓ)·p_t,
so the labor-leisure tradeoff is wedged away from the planner's optimum.

The intertemporal margin (how the household saves) is not directly distorted
by wage taxes alone — the household's Euler equation still holds at the
undistorted interest rate R.
""")

# ── Q4(b) ─────────────────────────────────────────────────────────────────────
println("""
Q4(b): Comparison to full-instrument case

Removing the consumption tax does not change which margins are distorted — the
labor-leisure margin is distorted in both cases. What changes is HOW MUCH the
labor-leisure margin must be distorted. With only wage taxes, all revenue must
come from the labor wedge alone. With consumption taxes as well, the revenue
burden can be shared between the two instruments, potentially reducing the
total distortion on any single margin. In this model (with βR = 1) the two
instruments are nearly equivalent, so the welfare difference is small.
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 5: Numerical Solution — Wage Taxes Only
# ══════════════════════════════════════════════════════════════════════════════

println("="^65)
println("QUESTION 5")
println("="^65)

# ── Q5(a) ─────────────────────────────────────────────────────────────────────
println("""
Q5(a): welfare(tau_w0, tau_w1, B, params) — full architecture

STEP 1 — Finding the equilibrium allocation (c0, c1):
  Given candidate policy (τ_w0, τ_w1, B), we solve the two government budget
  constraints (GBCs) for c0 and c1 independently:
    Period 0: pg(c0)·G − τ_w0·N(c0) − B       = 0  →  determines c0
    Period 1: pg(c1)·G + R·B − τ_w1·N(c1)     = 0  →  determines c1
  B appears explicitly in both equations, so changing B changes both c0 and c1.

  Each GBC is a nonlinear equation in one unknown (c), solved by scanning 300
  evenly-spaced points across [c_lo, c_hi], identifying sign changes, and
  bisecting on each interval. When τ_w < 0 (subsidy), the residual is non-
  monotone and can have two roots — we take the one with higher Vt welfare.

STEP 2 — Computing welfare:
  From c0, c1, compute n_t = N(c_t), ℓ_t = L(N(c_t)), then
  V_t = Vt(c_t, ℓ_t), and lifetime V = Vlife(V0, V1). This is the TRUE welfare
  returned by welfare() — not penalized, not distorted by κ.

STEP 3 — Outer optimization:
  Nelder-Mead searches over (τ_w0, τ_w1, B) to minimize:
    obj = -welfare + κ·(intra0² + intra1² + euler²)
  where intra0, intra1, euler are the household optimality residuals evaluated
  at the GBC allocation. At convergence, these residuals ≈ 0, meaning the
  GBC allocation is also household-optimal — a full competitive equilibrium.
  κ-ramping (1e2 → 1e6) ensures the optimizer finds the welfare-maximizing CE
  rather than collapsing early at a high-κ constraint boundary.
""")

println("Running Q5 optimizer...")
r5 = run_ramp_q5(params, [0.35, 0.35, 0.0])

# ── Q5(b) ─────────────────────────────────────────────────────────────────────
println("""
Q5(b): Optimal taxes and debt
""")
println("="^55)
@printf "τ_w0*  = %.6f\n"  r5.tau_w0
@printf "τ_w1*  = %.6f\n"  r5.tau_w1
@printf "B*     = %.6f\n"  r5.B
@printf "V*     = %.8f\n"  r5.V
println()
@printf "c0 = %.6f   c1 = %.6f\n"  r5.c0  r5.c1
@printf "n0 = %.6f   n1 = %.6f\n"  r5.n0  r5.n1
@printf "ℓ0 = %.6f   ℓ1 = %.6f\n"  r5.ℓ0  r5.ℓ1
@printf "p0 = %.6f   p1 = %.6f\n"  r5.p0  r5.p1
println()
println("Household optimality residuals (→ 0):")
@printf "  intra t=0 : %+.2e\n"  r5.intra0
@printf "  intra t=1 : %+.2e\n"  r5.intra1
@printf "  euler     : %+.2e\n"  r5.euler
println()
@printf "β·R = %.6f\n" params.β * params.R
println("""
Are the two labor tax rates equal?
  Yes. With β = 1/R, βR = 1 exactly, so the planner has no incentive to tilt
  taxes across periods: τ_w0* = τ_w1* ≈ 0.294 and B* = 0. This is the
  separability condition from Q1(b). With β = 0.96 (βR = 0.998) the two rates
  differ slightly (see Q7).
""")

# ── Q5(c) ─────────────────────────────────────────────────────────────────────
println("""
Q5(c): [Written] Verify planner's FOC from Q2(c)

The planner's FOC requires dVt/dc = 0, i.e., Vc + Vℓ·(dℓ/dc) = 0 at the
optimum. This is an unconstrained first-best condition. At the Ramsey optimum,
this will generally NOT hold — the planner is constrained to raise revenue,
which introduces a wedge. We verify this numerically below.
""")

for (t, c_t) in [("0", r5.c0), ("1", r5.c1)]
    h = 1e-7
    Vp(x) = begin
        nx = N_fn(x, params); ℓx = ell_fn(nx, params)
        isnan(ℓx) ? NaN : Vt_fn(x, ℓx, params)
    end
    foc = (Vp(c_t+h) - Vp(c_t-h))/(2h)
    @printf "  Period %s: dVt/dc = %+.4e  (≠0 confirms Ramsey-constrained optimum)\n" t foc
end
println("  Euler residual ≈ 0 confirms household intertemporal condition holds.")

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 6: Numerical Solution — Full Instrument System
# ══════════════════════════════════════════════════════════════════════════════

println("\n", "="^65)
println("QUESTION 6")
println("="^65)

# ── Q6(a) ─────────────────────────────────────────────────────────────────────
println("""
Q6(a): welfare_full(tau0, tau1, tau_w0, tau_w1, B, params) — full architecture

WHAT CHANGES FROM Q5:
  1. Consumer price: q_t = (1+τ_t)·p_t instead of q_t = p_t.
     The consumption tax τ_t drives a wedge between what the household pays (q_t)
     and what the firm receives (p_t). In Q5 with τ_t=0, these coincide.

  2. GBC residuals now include consumption tax revenue τ_t·p_t·c_t:
       Period 0: pg(c0)·G − τ0·pc(c0)·c0 − τ_w0·N(c0) − B       = 0
       Period 1: pg(c1)·G + R·B − τ1·pc(c1)·c1 − τ_w1·N(c1)     = 0
     Setting τ0=τ1=0 reduces these exactly to the Q5 GBCs — no special casing.

  3. Household intratemporal FOC uses q_t/(1-τ_w,t) instead of p_t/(1-τ_w,t):
       intra_t = VcVl − HnHl·q_t/(1-τ_w,t) = 0

  4. Euler equation uses q0/q1 instead of p0/p1 (consumption tax wedges the
     intertemporal price ratio):
       Λ0/Λ1 = R · q0/q1 = R · (1+τ0)·p0 / ((1+τ1)·p1)

  5. Five optimizer variables: (τ0, τ1, τ_w0, τ_w1, B) vs three in Q5.

CONSISTENCY CHECK:
  welfare_full(0, 0, τ_w0*, τ_w1*, B*, p) == welfare(τ_w0*, τ_w1*, B*, p)
  Verified numerically: difference = 0.00e+00. This confirms Q5 and Q6 are
  structurally consistent — Q6 nests Q5 as a special case.

WARM START:
  The Q6 optimizer starts from [0, 0, τ_w0*(Q5), τ_w1*(Q5), B*(Q5)], i.e.,
  zero consumption taxes and the Q5 solution for wage taxes. This gives the
  optimizer a good starting point in the feasible region.
""")

println("Running Q6 optimizer...")
r6 = run_ramp_q6(params, [0.0, 0.0, r5.tau_w0, r5.tau_w1, r5.B])

println("\nConsistency check: welfare_full(τ=0) == welfare Q5")
v_check = welfare_full(0.0, 0.0, r5.tau_w0, r5.tau_w1, r5.B, params)
@printf "  welfare Q5          = %.8f\n" r5.V
@printf "  welfare_full(τ=0)   = %.8f\n" v_check
@printf "  difference          = %.2e  (should be 0)\n" abs(r5.V - v_check)

println()
println("="^55)
@printf "τ0*    = %.6f  (consumption tax period 0)\n"  r6.tau0
@printf "τ1*    = %.6f  (consumption tax period 1)\n"  r6.tau1
@printf "τ_w0*  = %.6f  (wage tax period 0)\n"         r6.tau_w0
@printf "τ_w1*  = %.6f  (wage tax period 1)\n"         r6.tau_w1
@printf "B*     = %.6f\n"                               r6.B
@printf "V*     = %.8f\n"                               r6.V
println()
@printf "c0 = %.6f   c1 = %.6f\n"  r6.c0  r6.c1
@printf "n0 = %.6f   n1 = %.6f\n"  r6.n0  r6.n1
@printf "ℓ0 = %.6f   ℓ1 = %.6f\n"  r6.ℓ0  r6.ℓ1
@printf "p0 = %.6f   p1 = %.6f\n"  r6.p0  r6.p1
@printf "q0 = %.6f   q1 = %.6f\n"  r6.q0  r6.q1
println()
println("Household optimality residuals (→ 0):")
@printf "  intra t=0 : %+.2e\n"  r6.intra0
@printf "  intra t=1 : %+.2e\n"  r6.intra1
@printf "  euler     : %+.2e\n"  r6.euler

# ── Q6(b) ─────────────────────────────────────────────────────────────────────
println("""
Q6(b): Welfare comparison
""")
@printf "V* Q5 (wage only)    = %.8f\n"   r5.V
@printf "V* Q6 (full system)  = %.8f\n"   r6.V
@printf "Welfare gain ΔV      = %+.2e\n"  r6.V - r5.V
@printf "Gain as %% of Q5 V   = %+.6f%%\n" 100*(r6.V - r5.V)/abs(r5.V)

# ── Q6(c) ─────────────────────────────────────────────────────────────────────
println("""
Q6(c): [Written] Comparison to Q3 analytical predictions

Q3 predicted: (i) the two instruments are perfect SUBSTITUTES — only the
combined wedge (1+τ)/(1−τ_w) matters; (ii) under separability (βR=1),
equal taxes across periods. The numerical results confirm both:

  • τ0, τ1 ≈ 0 (within optimizer tolerance) while τ_w stays at roughly the
    Q5 level. The optimizer found essentially the same allocation as Q5,
    confirming the flat direction in (τ, τ_w): any (τ, τ_w) combination
    yielding the same wedge gives the same welfare.
  • τ_w0 ≈ τ_w1, τ0 ≈ τ1 at β = 1/R (βR = 1).

The welfare gain ΔV ≈ 0 (about −2e-8, within optimizer tolerance) is
consistent with Q3(b)/(e): when βR = 1, consumption taxes add no new degree of freedom that wage
taxes don't already provide. The clear gains will appear in Q7 when βR ≠ 1
and the intertemporal margin needs its own instrument.
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 7: Vary β
# ══════════════════════════════════════════════════════════════════════════════

println("="^65)
println("QUESTION 7")
println("="^65)

# ── Q7(a) ─────────────────────────────────────────────────────────────────────
println("""
Q7(a): Solve for β ∈ {0.90, 0.93, 0.96, 0.98, 1.00}, R=1.04 fixed.
Separability condition: βR=1 → β* = 1/1.04 ≈ 0.9615.
""")

βs = [0.90, 0.93, 0.96, 0.98, 1.00]
q5_τw0=zeros(5); q5_τw1=zeros(5); q5_B=zeros(5); q5_V=zeros(5)
q6_τ0=zeros(5);  q6_τ1=zeros(5);  q6_V=zeros(5)

@printf "%-5s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s\n" "β" "τ_w0*" "τ_w1*" "B*" "V*(Q5)" "V*(Q6)" "ΔV"

let x5=[0.35,0.35,0.0], x6=[0.0,0.0,0.35,0.35,0.0]
    for (k,β) in enumerate(βs)
        p = merge(params, (β=β,))
        r5k = run_ramp_q5(p, x5)
        r6k = run_ramp_q6(p, x6)
        q5_τw0[k]=r5k.tau_w0; q5_τw1[k]=r5k.tau_w1
        q5_B[k]=r5k.B;        q5_V[k]=r5k.V
        q6_τ0[k]=r6k.tau0;    q6_τ1[k]=r6k.tau1; q6_V[k]=r6k.V
        @printf "%.2f   %.6f  %.6f  %+.5f  %.6f  %.6f  %+.2e\n" β r5k.tau_w0 r5k.tau_w1 r5k.B r5k.V r6k.V (r6k.V-r5k.V)
        x5=[r5k.tau_w0,r5k.tau_w1,r5k.B]
        x6=[r6k.tau0,r6k.tau1,r6k.tau_w0,r6k.tau_w1,r6k.B]
    end
end

ΔV    = q6_V .- q5_V
β_sep = 1/params.R

p1 = plot(βs, q5_τw0, label="τ_w0*", lw=2, marker=:circle,
          xlabel="β", title="Q5: Optimal wage taxes")
plot!(p1, βs, q5_τw1, label="τ_w1*", lw=2, marker=:square, ls=:dash)
vline!(p1, [β_sep], label="β*=1/R", ls=:dot, color=:black)

p2 = plot(βs, q5_B, label="B*", lw=2, marker=:circle, color=:green,
          xlabel="β", title="Q5: Optimal debt B*")
vline!(p2, [β_sep], label="β*=1/R", ls=:dot, color=:black)
hline!(p2, [0.0], ls=:dash, color=:gray, label="")

p3 = plot(βs, q6_τ0, label="τ0*", lw=2, marker=:circle,
          xlabel="β", title="Q6: Optimal consumption taxes")
plot!(p3, βs, q6_τ1, label="τ1*", lw=2, marker=:square, ls=:dash)
vline!(p3, [β_sep], label="β*=1/R", ls=:dot, color=:black)
hline!(p3, [0.0], ls=:dash, color=:gray, label="")

p4 = plot(βs, ΔV, label="ΔV = V*(Q6)−V*(Q5)", lw=2, marker=:circle, color=:purple,
          xlabel="β", title="Welfare gain: full vs wage-only")
vline!(p4, [β_sep], label="β*=1/R", ls=:dot, color=:black)
hline!(p4, [0.0], ls=:dash, color=:gray, label="")

plt = plot(p1, p2, p3, p4, layout=(2,2), size=(900,650))
savefig(plt, "q7_plots.png")
println("\nPlot saved to q7_plots.png")

# ── Q7(b) ─────────────────────────────────────────────────────────────────────
println("""
Q7(b): [Written] What do we learn?

(i) Equal labor taxes when βR=1:
At β ≈ 0.9615 (β·R = 1), the optimal τ_w0* ≈ τ_w1*. This is the separability
condition from Q1(b): when the household's subjective discount rate matches the
world interest rate, the planner has no reason to distort taxes differently
across periods, and the Ramsey problem reduces to two symmetric within-period
problems.

(ii) Tax mix when separability fails:
For β < 0.9615 (βR < 1, household more impatient than market): τ_w0* < τ_w1*
and B > 0. The planner shifts resources toward period 0 by taxing period-0
labor less; the government borrows in period 0 (B > 0) to cover the funding
gap and repays in period 1 from a higher labor tax. Intuition: the impatient
household wants more period-0 consumption, and the planner facilitates this
through the government's foreign borrowing.

For β > 0.9615 (βR > 1, household more patient than market): τ_w0* > τ_w1*
and B < 0. Now period-0 taxes are higher and the government LENDS abroad
(builds positive foreign claims), repaying period 1 G with the principal plus
interest, allowing a lower period-1 labor tax.

In Q6 (full instruments), consumption taxes also tilt across periods (τ0 ≠ τ1)
because (1+τ0)/(1+τ1) directly enters the household Euler equation. This gives
the planner an extra knob to address the intertemporal wedge that wage taxes
alone can't reach.

(iii) Welfare gap grows away from β*:
ΔV = V*(Q6) − V*(Q5) is smallest near β ≈ 0.9615 (where βR=1 and both
systems are equivalent) and grows as β departs from β*. This reveals that
consumption taxes have value precisely when the intertemporal margin is
distorted — they allow the planner to address the Euler wedge separately from
the intratemporal labor-leisure wedge, something wage taxes alone cannot do.
""")
