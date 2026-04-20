"""
Optimization Algorithm Implementations
========================================
  SBOA  : Secretary Bird Optimization Algorithm (base)
  ISBOA : Improved SBOA (DE + OBL + non-linear adaptive evasion)
  GWO   : Grey Wolf Optimizer
  WOA   : Whale Optimization Algorithm
  SCA   : Sine Cosine Algorithm
  SSA   : Salp Swarm Algorithm
  HHO   : Harris Hawks Optimization
  MPA   : Marine Predators Algorithm
  AOA   : Arithmetic Optimization Algorithm
"""

import numpy as np
from scipy.special import gamma


# ======================== Utilities ========================

def levy_flight(dim, beta=1.5):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta) + 1e-30)


def _bounds(lb, ub, dim):
    lb = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, dtype=float)
    return lb, ub


# ======================== Base SBOA ========================

def sboa(obj_func, lb, ub, dim, pop_size=30, max_fes=60000):
    """Secretary Bird Optimization Algorithm (Fu et al., 2024)."""
    lb, ub = _bounds(lb, ub, dim)
    fes = 0

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.array([obj_func(pop[i]) for i in range(pop_size)])
    fes += pop_size

    bi = np.argmin(fit)
    best_pos, best_fit = pop[bi].copy(), fit[bi]
    conv = [best_fit]

    T = max((max_fes - fes) // (pop_size * 2), 1)
    t = 0

    while fes < max_fes:
        t += 1
        ratio = min(t / T, 1.0)
        CF = max(0.0, 1 - ratio) ** (2 * ratio) if ratio < 1 else 0.0

        for i in range(pop_size):
            if fes >= max_fes:
                break

            # ---- Exploration (Predation) ----
            if t < T / 3:
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                xn = pop[i] + np.random.rand(dim) * (pop[r1] - pop[r2])
            elif t < 2 * T / 3:
                RB = np.random.randn(dim)
                xn = best_pos + np.exp(ratio ** 4) * (RB - 0.5) * (best_pos - pop[i])
            else:
                RL = 0.5 * levy_flight(dim)
                xn = best_pos + CF * pop[i] * RL

            xn = np.clip(xn, lb, ub)
            fn = obj_func(xn); fes += 1

            if fn < fit[i]:
                pop[i], fit[i] = xn, fn
                if fn < best_fit:
                    best_pos, best_fit = xn.copy(), fn

            if fes >= max_fes:
                break

            # ---- Exploitation (Escape) ----
            if np.random.rand() < 0.5:
                RB = np.random.rand(dim)
                xn = best_pos + (1 - ratio) ** 2 * (2 * RB - 1) * pop[i]
            else:
                K = np.round(1 + np.random.rand())
                R2 = np.random.rand(dim)
                ri = np.random.randint(pop_size)
                xn = pop[i] + R2 * (pop[ri] - K * pop[i])

            xn = np.clip(xn, lb, ub)
            fn = obj_func(xn); fes += 1

            if fn < fit[i]:
                pop[i], fit[i] = xn, fn
                if fn < best_fit:
                    best_pos, best_fit = xn.copy(), fn

        conv.append(best_fit)

    return best_pos, best_fit, conv


# ======================== Improved SBOA ========================

def isboa(obj_func, lb, ub, dim, pop_size=30, max_fes=60000):
    """
    Improved SBOA:
      1) Full DE mutation + binomial crossover in exploration
      2) Opposition-Based Learning initialisation
      3) Non-linear adaptive evasion factor  alpha = 1-(FEs/MaxFEs)^2
    """
    lb, ub = _bounds(lb, ub, dim)
    fes = 0
    F_de = 0.5
    CR = 0.9

    # -- Improvement 2: OBL --
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    obl = lb + ub - pop
    comb = np.vstack([pop, obl])
    cfit = np.array([obj_func(comb[j]) for j in range(2 * pop_size)])
    fes += 2 * pop_size

    idx = np.argsort(cfit)[:pop_size]
    pop, fit = comb[idx].copy(), cfit[idx].copy()

    bi = np.argmin(fit)
    best_pos, best_fit = pop[bi].copy(), fit[bi]
    conv = [best_fit]

    T = max((max_fes - fes) // (pop_size * 2), 1)
    t = 0

    while fes < max_fes:
        t += 1
        ratio = min(t / T, 1.0)
        CF = max(0.0, 1 - ratio) ** (2 * ratio) if ratio < 1 else 0.0
        alpha = 1.0 - (fes / max_fes) ** 2          # Improvement 3

        for i in range(pop_size):
            if fes >= max_fes:
                break

            # -- Improvement 1: DE Mechanics --
            cands = list(range(pop_size)); cands.remove(i)
            r1, r2, r3 = np.random.choice(cands, 3, replace=False)

            if t < T / 3:
                mutant = pop[r1] + F_de * (pop[r2] - pop[r3])
            elif t < 2 * T / 3:
                RB = np.random.randn(dim)
                mutant = pop[i] + F_de * (best_pos - pop[i]) + F_de * (pop[r1] - pop[r2]) + 0.01 * RB
            else:
                RL = 0.5 * levy_flight(dim)
                mutant = best_pos + F_de * (pop[r1] - pop[r2]) + CF * RL

            # Binomial crossover
            trial = pop[i].copy()
            jr = np.random.randint(dim)
            mask = (np.random.rand(dim) < CR) | (np.arange(dim) == jr)
            trial[mask] = mutant[mask]

            trial = np.clip(trial, lb, ub)
            ft = obj_func(trial); fes += 1

            if ft < fit[i]:
                pop[i], fit[i] = trial, ft
                if ft < best_fit:
                    best_pos, best_fit = trial.copy(), ft

            if fes >= max_fes:
                break

            # -- Exploitation with non-linear adaptive evasion --
            if np.random.rand() < 0.5:
                RB = np.random.rand(dim)
                xn = best_pos + alpha * (2 * RB - 1) * pop[i]
            else:
                K = np.round(1 + np.random.rand())
                R2 = np.random.rand(dim)
                ri = np.random.randint(pop_size)
                xn = pop[i] + alpha * R2 * (pop[ri] - K * pop[i])

            xn = np.clip(xn, lb, ub)
            fn = obj_func(xn); fes += 1

            if fn < fit[i]:
                pop[i], fit[i] = xn, fn
                if fn < best_fit:
                    best_pos, best_fit = xn.copy(), fn

        conv.append(best_fit)

    return best_pos, best_fit, conv


# ======================== Grey Wolf Optimizer ========================

def gwo(obj_func, lb, ub, dim, pop_size=30, max_fes=60000):
    """Mirjalili et al., 2014."""
    lb, ub = _bounds(lb, ub, dim)
    fes = 0

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.array([obj_func(pop[i]) for i in range(pop_size)])
    fes += pop_size

    si = np.argsort(fit)
    alpha_pos, alpha_fit = pop[si[0]].copy(), fit[si[0]]
    beta_pos = pop[si[1]].copy()
    delta_pos = pop[si[2]].copy()
    conv = [alpha_fit]

    T = max((max_fes - fes) // pop_size, 1)

    for t in range(1, T + 1):
        if fes >= max_fes:
            break
        a = 2 - 2 * t / T

        for i in range(pop_size):
            if fes >= max_fes:
                break
            r1 = np.random.rand(dim); r2 = np.random.rand(dim)
            A1 = 2*a*r1 - a; C1 = 2*r2
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - pop[i])

            r1 = np.random.rand(dim); r2 = np.random.rand(dim)
            A2 = 2*a*r1 - a; C2 = 2*r2
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - pop[i])

            r1 = np.random.rand(dim); r2 = np.random.rand(dim)
            A3 = 2*a*r1 - a; C3 = 2*r2
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - pop[i])

            xn = np.clip((X1 + X2 + X3) / 3, lb, ub)
            fn = obj_func(xn); fes += 1
            pop[i], fit[i] = xn, fn

        si = np.argsort(fit)
        if fit[si[0]] < alpha_fit:
            alpha_pos, alpha_fit = pop[si[0]].copy(), fit[si[0]]
        beta_pos = pop[si[1]].copy()
        delta_pos = pop[si[2]].copy()
        conv.append(alpha_fit)

    return alpha_pos, alpha_fit, conv


# ======================== Whale Optimization Algorithm ========================

def woa(obj_func, lb, ub, dim, pop_size=30, max_fes=60000):
    """Mirjalili & Lewis, 2016."""
    lb, ub = _bounds(lb, ub, dim)
    fes = 0

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.array([obj_func(pop[i]) for i in range(pop_size)])
    fes += pop_size

    bi = np.argmin(fit)
    best_pos, best_fit = pop[bi].copy(), fit[bi]
    conv = [best_fit]

    T = max((max_fes - fes) // pop_size, 1)

    for t in range(1, T + 1):
        if fes >= max_fes:
            break
        a = 2 - 2 * t / T

        for i in range(pop_size):
            if fes >= max_fes:
                break
            A = 2 * a * np.random.rand() - a
            C = 2 * np.random.rand()
            p = np.random.rand()
            l = np.random.uniform(-1, 1)

            if p < 0.5:
                if np.abs(A) < 1:
                    D = np.abs(C * best_pos - pop[i])
                    xn = best_pos - A * D
                else:
                    ri = np.random.randint(pop_size)
                    D = np.abs(C * pop[ri] - pop[i])
                    xn = pop[ri] - A * D
            else:
                D = np.abs(best_pos - pop[i])
                xn = D * np.exp(l) * np.cos(2 * np.pi * l) + best_pos

            xn = np.clip(xn, lb, ub)
            fn = obj_func(xn); fes += 1
            pop[i], fit[i] = xn, fn

            if fn < best_fit:
                best_pos, best_fit = xn.copy(), fn

        conv.append(best_fit)

    return best_pos, best_fit, conv


# ======================== Sine Cosine Algorithm ========================

def sca(obj_func, lb, ub, dim, pop_size=30, max_fes=60000):
    """Mirjalili, 2016."""
    lb, ub = _bounds(lb, ub, dim)
    fes = 0

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.array([obj_func(pop[i]) for i in range(pop_size)])
    fes += pop_size

    bi = np.argmin(fit)
    best_pos, best_fit = pop[bi].copy(), fit[bi]
    conv = [best_fit]

    T = max((max_fes - fes) // pop_size, 1)

    for t in range(1, T + 1):
        if fes >= max_fes:
            break
        r1 = 2.0 - t * 2.0 / T

        for i in range(pop_size):
            if fes >= max_fes:
                break
            r2 = 2 * np.pi * np.random.rand(dim)
            r3 = 2 * np.random.rand(dim)
            r4 = np.random.rand()

            if r4 < 0.5:
                xn = pop[i] + r1 * np.sin(r2) * np.abs(r3 * best_pos - pop[i])
            else:
                xn = pop[i] + r1 * np.cos(r2) * np.abs(r3 * best_pos - pop[i])

            xn = np.clip(xn, lb, ub)
            fn = obj_func(xn); fes += 1
            pop[i], fit[i] = xn, fn

            if fn < best_fit:
                best_pos, best_fit = xn.copy(), fn

        conv.append(best_fit)

    return best_pos, best_fit, conv


# ======================== Salp Swarm Algorithm ========================

def ssa(obj_func, lb, ub, dim, pop_size=30, max_fes=60000):
    """Mirjalili et al., 2017."""
    lb, ub = _bounds(lb, ub, dim)
    fes = 0

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.array([obj_func(pop[i]) for i in range(pop_size)])
    fes += pop_size

    bi = np.argmin(fit)
    food_pos, food_fit = pop[bi].copy(), fit[bi]
    conv = [food_fit]

    T = max((max_fes - fes) // pop_size, 1)

    for t in range(1, T + 1):
        if fes >= max_fes:
            break
        c1 = 2 * np.exp(-(4 * t / T) ** 2)

        for i in range(pop_size):
            if fes >= max_fes:
                break
            if i < pop_size // 2:
                c2 = np.random.rand(dim)
                c3 = np.random.rand(dim)
                xn = np.where(c3 < 0.5,
                              food_pos + c1 * (c2 * (ub - lb) + lb),
                              food_pos - c1 * (c2 * (ub - lb) + lb))
            else:
                xn = (pop[i] + pop[i - 1]) / 2.0

            xn = np.clip(xn, lb, ub)
            fn = obj_func(xn); fes += 1
            pop[i], fit[i] = xn, fn

            if fn < food_fit:
                food_pos, food_fit = xn.copy(), fn

        conv.append(food_fit)

    return food_pos, food_fit, conv


# ======================== Harris Hawks Optimization ========================

def hho(obj_func, lb, ub, dim, pop_size=30, max_fes=60000):
    """Heidari et al., 2019."""
    lb, ub = _bounds(lb, ub, dim)
    fes = 0

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.array([obj_func(pop[i]) for i in range(pop_size)])
    fes += pop_size

    bi = np.argmin(fit)
    rabbit_pos, rabbit_fit = pop[bi].copy(), fit[bi]
    conv = [rabbit_fit]

    T = max((max_fes - fes) // pop_size, 1)

    for t in range(1, T + 1):
        if fes >= max_fes:
            break
        E0 = 2 * np.random.rand() - 1
        E = 2 * E0 * (1 - t / T)

        for i in range(pop_size):
            if fes >= max_fes:
                break

            if np.abs(E) >= 1:
                # --- Exploration ---
                q = np.random.rand()
                if q >= 0.5:
                    ri = np.random.randint(pop_size)
                    xn = pop[ri] - np.random.rand() * np.abs(pop[ri] - 2 * np.random.rand() * pop[i])
                else:
                    xm = pop.mean(axis=0)
                    xn = (rabbit_pos - xm) - np.random.rand() * (lb + np.random.rand(dim) * (ub - lb))

                xn = np.clip(xn, lb, ub)
                fn = obj_func(xn); fes += 1

            else:
                r = np.random.rand()
                J = 2 * (1 - np.random.rand())

                if r >= 0.5 and np.abs(E) >= 0.5:
                    # Soft besiege
                    xn = rabbit_pos - E * np.abs(J * rabbit_pos - pop[i])
                    xn = np.clip(xn, lb, ub)
                    fn = obj_func(xn); fes += 1

                elif r >= 0.5 and np.abs(E) < 0.5:
                    # Hard besiege
                    xn = rabbit_pos - E * np.abs(rabbit_pos - pop[i])
                    xn = np.clip(xn, lb, ub)
                    fn = obj_func(xn); fes += 1

                elif r < 0.5 and np.abs(E) >= 0.5:
                    # Soft besiege + progressive rapid dives
                    Y = rabbit_pos - E * np.abs(J * rabbit_pos - pop[i])
                    Y = np.clip(Y, lb, ub)
                    fY = obj_func(Y); fes += 1
                    if fY < fit[i]:
                        xn, fn = Y, fY
                    else:
                        Z = Y + np.random.rand(dim) * levy_flight(dim)
                        Z = np.clip(Z, lb, ub)
                        if fes < max_fes:
                            fZ = obj_func(Z); fes += 1
                            xn, fn = (Z, fZ) if fZ < fit[i] else (pop[i].copy(), fit[i])
                        else:
                            xn, fn = pop[i].copy(), fit[i]

                else:
                    # Hard besiege + progressive rapid dives
                    xm = pop.mean(axis=0)
                    Y = rabbit_pos - E * np.abs(J * rabbit_pos - xm)
                    Y = np.clip(Y, lb, ub)
                    fY = obj_func(Y); fes += 1
                    if fY < fit[i]:
                        xn, fn = Y, fY
                    else:
                        Z = Y + np.random.rand(dim) * levy_flight(dim)
                        Z = np.clip(Z, lb, ub)
                        if fes < max_fes:
                            fZ = obj_func(Z); fes += 1
                            xn, fn = (Z, fZ) if fZ < fit[i] else (pop[i].copy(), fit[i])
                        else:
                            xn, fn = pop[i].copy(), fit[i]

            pop[i], fit[i] = xn, fn
            if fn < rabbit_fit:
                rabbit_pos, rabbit_fit = xn.copy(), fn

        conv.append(rabbit_fit)

    return rabbit_pos, rabbit_fit, conv


# ======================== Marine Predators Algorithm ========================

def mpa(obj_func, lb, ub, dim, pop_size=30, max_fes=60000):
    """Faramarzi et al., 2020."""
    lb, ub = _bounds(lb, ub, dim)
    fes = 0
    P = 0.5
    FADs = 0.2

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.array([obj_func(pop[i]) for i in range(pop_size)])
    fes += pop_size

    bi = np.argmin(fit)
    elite = pop[bi].copy()
    elite_fit = fit[bi]
    conv = [elite_fit]

    T = max((max_fes - fes) // pop_size, 1)

    for t in range(1, T + 1):
        if fes >= max_fes:
            break
        CF = (1 - t / T) ** (2 * t / T)

        for i in range(pop_size):
            if fes >= max_fes:
                break

            if t < T / 3:
                # Phase 1: prey faster
                RB = np.random.randn(dim)
                stepsize = RB * (elite - RB * pop[i])
                xn = pop[i] + P * np.random.rand(dim) * stepsize
            elif t < 2 * T / 3:
                if i < pop_size // 2:
                    RL = 0.5 * levy_flight(dim)
                    stepsize = RL * (elite - RL * pop[i])
                    xn = elite + P * CF * stepsize
                else:
                    RB = np.random.randn(dim)
                    stepsize = RB * (RB * elite - pop[i])
                    xn = pop[i] + P * np.random.rand(dim) * stepsize
            else:
                # Phase 3: predator faster
                RL = 0.5 * levy_flight(dim)
                stepsize = RL * (RL * elite - pop[i])
                xn = elite + P * CF * stepsize

            xn = np.clip(xn, lb, ub)
            fn = obj_func(xn); fes += 1

            if fn < fit[i]:
                pop[i], fit[i] = xn, fn
                if fn < elite_fit:
                    elite, elite_fit = xn.copy(), fn

            # FADs effect
            if fes >= max_fes:
                break
            if np.random.rand() < FADs:
                U = (np.random.rand(dim) < FADs).astype(float)
                xn = pop[i] + CF * (lb + np.random.rand(dim) * (ub - lb)) * U
            else:
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                xn = pop[i] + (FADs * (1 - np.random.rand()) + np.random.rand()) * (pop[r1] - pop[r2])

            xn = np.clip(xn, lb, ub)
            fn = obj_func(xn); fes += 1

            if fn < fit[i]:
                pop[i], fit[i] = xn, fn
                if fn < elite_fit:
                    elite, elite_fit = xn.copy(), fn

        conv.append(elite_fit)

    return elite, elite_fit, conv


# ======================== Arithmetic Optimization Algorithm ========================

def aoa(obj_func, lb, ub, dim, pop_size=30, max_fes=60000):
    """Abualigah et al., 2021."""
    lb, ub = _bounds(lb, ub, dim)
    fes = 0
    MOA_Min, MOA_Max = 0.2, 0.9
    alpha_aoa = 5
    eps = 1e-30

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.array([obj_func(pop[i]) for i in range(pop_size)])
    fes += pop_size

    bi = np.argmin(fit)
    best_pos, best_fit = pop[bi].copy(), fit[bi]
    conv = [best_fit]

    T = max((max_fes - fes) // pop_size, 1)

    for t in range(1, T + 1):
        if fes >= max_fes:
            break
        MOA = MOA_Min + t * (MOA_Max - MOA_Min) / T
        MOP = 1 - (t ** (1 / alpha_aoa)) / (T ** (1 / alpha_aoa) + eps)

        for i in range(pop_size):
            if fes >= max_fes:
                break
            xn = pop[i].copy()

            for j in range(dim):
                r1 = np.random.rand()
                mu = np.random.rand() * 0.5
                if r1 > MOA:
                    # Exploration (math operations)
                    if np.random.rand() > 0.5:
                        xn[j] = best_pos[j] / (MOP + eps) * ((ub[j] - lb[j]) * mu + lb[j])
                    else:
                        xn[j] = best_pos[j] * MOP * ((ub[j] - lb[j]) * mu + lb[j])
                else:
                    # Exploitation
                    if np.random.rand() > 0.5:
                        xn[j] = best_pos[j] - MOP * ((ub[j] - lb[j]) * mu + lb[j])
                    else:
                        xn[j] = best_pos[j] + MOP * ((ub[j] - lb[j]) * mu + lb[j])

            xn = np.clip(xn, lb, ub)
            fn = obj_func(xn); fes += 1
            pop[i], fit[i] = xn, fn

            if fn < best_fit:
                best_pos, best_fit = xn.copy(), fn

        conv.append(best_fit)

    return best_pos, best_fit, conv


# ======================== Registry ========================

ALGORITHMS = {
    "ISBOA": isboa,
    "SBOA":  sboa,
    "GWO":   gwo,
    "WOA":   woa,
    "SCA":   sca,
    "SSA":   ssa,
    "HHO":   hho,
    "MPA":   mpa,
    "AOA":   aoa,
}
