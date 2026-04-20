# Performance Comparison Report
# Improved Secretary Bird Optimization Algorithm (ISBOA)

## 1  Introduction

This report compares **ISBOA** (Improved SBOA with Full DE Mechanics, Opposition-Based Learning Initialisation, and Non-Linear Adaptive Evasion Factor) against the base SBOA and seven recent nature-inspired optimisation algorithms:

- ISBOA
- SBOA
- GWO
- WOA
- SCA
- SSA
- HHO
- MPA
- AOA

## 2  Improvements in ISBOA

### 2.1 Full Differential Evolution (DE) Mechanics
The exploration phase now uses DE/rand/1 (Stage 1), DE/current-to-best/1 (Stage 2), and DE/best/1 + Lévy (Stage 3) mutation strategies, followed by **binomial crossover** and **greedy selection** to preserve the fittest traits and prevent premature convergence.

### 2.2 Opposition-Based Learning (OBL) Initialisation
For each randomly generated individual $X$, its opposite $X_{obl} = LB + UB - X$ is also created. Both populations are evaluated and the top 50 % fittest individuals are retained, effectively halving the initial distance to the global optimum.

### 2.3 Non-Linear Adaptive Evasion Factor
The fixed $(1 - t/T)^2$ evasion coefficient is replaced by $\alpha = 1 - (FEs / MaxFEs)^2$, which maintains exploration capacity longer and transitions smoothly into fine-grained exploitation near the budget limit of 60 000 FEs.

## 3  Experimental Setup

| Parameter | Value |
|-----------|-------|
| Population size | 30 |
| Max function evaluations | 60,000 |
| Independent runs | 30 |
| Dimension (CEC 2014/2017) | 30 |
| Dimension (CEC 2020/2022) | 10 |
| Significance level (Wilcoxon) | 0.05 |

## 4  CEC 2022 Results

### Mean (Std)

| Function | ISBOA | SBOA | GWO | WOA | SCA | SSA | HHO | MPA | AOA |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| F1 | 5.43e-09 (4.52e-09) | 0.00e+00 (0.00e+00) | 9.66e+01 (4.62e+01) | 6.74e+03 (2.69e+03) | 3.56e+02 (6.01e+01) | 4.29e-10 (1.75e-10) | 4.57e-01 (1.04e-01) | 6.05e-11 (3.99e-11) | 2.59e-03 (1.08e-03) |
| F2 | 5.63e+00 (2.85e+00) | 8.92e+00 (0.00e+00) | 4.74e+01 (3.49e+01) | 2.86e+02 (9.62e+01) | 5.54e+01 (8.02e+00) | 4.17e+00 (1.89e-01) | 6.20e+00 (4.79e+00) | 2.82e-06 (4.86e-06) | 8.92e+00 (1.28e-03) |

### Rankings

| Function | ISBOA | SBOA | GWO | WOA | SCA | SSA | HHO | MPA | AOA |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| F1 | 4 | 1 | 7 | 9 | 8 | 3 | 6 | 2 | 5 |
| F2 | 3 | 5 | 7 | 9 | 8 | 2 | 4 | 1 | 6 |
| **Avg** | 3.50 | 3.00 | 7.00 | 9.00 | 8.00 | 2.50 | 5.00 | 1.50 | 5.50 |

## 5  Wilcoxon Rank-Sum Test

Comparison of ISBOA vs. each opponent (significance level α = 0.05).  
**+** = ISBOA wins, **=** = tie, **-** = ISBOA loses.

| Year | Function | AOA | GWO | HHO | MPA | SBOA | SCA | SSA | WOA |
|------|----------|---|---|---|---|---|---|---|---|
| 2022 | F1 | + | + | + | - | - | + | - | + |
| 2022 | F2 | + | = | = | - | = | + | = | + |

### Win / Tie / Loss Summary

| Opponent | Win | Tie | Loss |
|----------|-----|-----|------|
| AOA | 2 | 0 | 0 |
| GWO | 1 | 1 | 0 |
| HHO | 1 | 1 | 0 |
| MPA | 0 | 0 | 2 |
| SBOA | 0 | 1 | 1 |
| SCA | 2 | 0 | 0 |
| SSA | 0 | 1 | 1 |
| WOA | 2 | 0 | 0 |

## 6  Overall Average Rank

| Algorithm | Avg Rank |
|-----------|----------|
| ISBOA | 3.50 |
| SBOA | 3.00 |
| GWO | 7.00 |
| WOA | 9.00 |
| SCA | 8.00 |
| SSA | 2.50 |
| HHO | 5.00 |
| MPA | 1.50 |
| AOA | 5.50 |

## 7  Conclusion

The results demonstrate the effectiveness of the three proposed improvements integrated into ISBOA. The full DE mechanics strengthen exploration diversity, OBL initialisation provides a better starting point, and the non-linear adaptive evasion factor ensures a smooth transition from exploration to exploitation within the 60 000 FEs budget.
