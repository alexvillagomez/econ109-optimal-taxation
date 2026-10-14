# Econ 109: Optimal Taxation in Julia

My problem sets from Econ 109 at UCLA. Each one sets up a small general-equilibrium model, solves it numerically in Julia, and uses the results to test a result from optimal tax theory: when a uniform VAT is optimal, when wage taxes should differ across periods, and how a wealth tax compares to a capital income tax.

All of the work here is my own. The notebooks have the code, outputs, and my written answers together, and each one has a PDF export next to it.

## Problem sets

**PS1: The case for a flat VAT** — [ps1/ps1.ipynb](ps1/ps1.ipynb)

A static Ramsey problem. A household with CES preferences over three goods, a CES production frontier that also produces fixed government purchases, and a government that has to fund those purchases with commodity taxes.

- Builds the household and firm problems and solves for the competitive equilibrium.
- Solves the Ramsey problem in primal form (choose the allocation subject to the resource constraint), then backs out the taxes that implement it from consumer and producer prices.
- The optimal taxes come out uniform at about 11% on every good (largest gap between rates is under 1e-3), which is the numerical version of the flat-VAT result.
- Sweeps the substitution parameters in preferences and production to show the tax stays uniform even as the allocation and the tax level move.

**PS2: Two-period Ramsey taxation in a small open economy** — [ps2/ps2.ipynb](ps2/ps2.ipynb)

A two-period economy at a world interest rate R, where the government can tax wages and consumption in each period and borrow or lend B between periods.

- For a candidate tax policy, solves the government budget constraints for consumption, then enforces the household's intratemporal conditions and Euler equation with a penalty. The penalty weight is ramped from 1e2 to 1e6 across Nelder-Mead stages so the optimizer doesn't collapse early onto a point that isn't an equilibrium.
- With βR = 1 the optimal wage tax is the same in both periods (about 29.4%) and the government doesn't borrow, matching the separability condition from the analytical part.
- Adding consumption taxes gives essentially no welfare gain at βR = 1.
- Sweeping β from 0.90 to 1.00: impatient households (β < 1/R) get a lower wage tax today, a higher one tomorrow, and government borrowing; patient households get the reverse. The welfare value of having consumption taxes grows as β moves away from 1/R.
- [ps2/ps2_q1_q7.jl](ps2/ps2_q1_q7.jl) is a standalone script covering the whole problem set, including the written answers to the analytical questions (Q1 to Q4).

**PS3: Portfolio choice under a wealth tax vs a capital income tax** — [ps3/ps3.ipynb](ps3/ps3.ipynb)

An infinitely lived investor with Epstein-Zin preferences chooses between a safe asset and a risky one.

- Solves the no-tax portfolio share and consumption rate from the portfolio first-order condition and the certainty-equivalent return.
- A 2% wealth tax leaves the portfolio unchanged and lowers expected growth. A capital income tax raising the same expected revenue needs a rate of about 55%, but it pushes the investor into the risky asset (risky share goes from 0.28 to 0.61), and expected growth ends up higher than under the wealth tax.
- Monte Carlo of 10,000 wealth paths over 100 periods under each regime. Simulated growth rates match the analytical ones to within 2e-4.
- Compares the exact risky share to the Merton approximation (0.267 vs 0.278).

## Running it

Requires Julia 1.10 or later. From the repo root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then open the notebooks with Jupyter and an IJulia kernel, or run the PS2 script directly:

```bash
julia --project=. ps2/ps2_q1_q7.jl
```

Packages: Optim, Roots, Plots, plus LinearAlgebra, Statistics, Printf, and Random from the standard library. Optim is pinned to 1.x since that's what the notebooks were run with.

## Layout

```
ps1/   ps1.ipynb, ps1.pdf
ps2/   ps2.ipynb, ps2.pdf, ps2_q1_q7.jl
ps3/   ps3.ipynb, ps3.pdf, figures/
```
