'''
Author: Nicholas Boulos
This file contains the PSO python implemtation to be imported to test scripts
'''
import numpy as np

def pso(
        obj_func,
        bounds,
        num_particles=30,
        iter=500,
        w=0.7,
        c1=1.5,
        c2=1.5,
        random_state=None
):
    rng = np.random.default_rng(random_state)

    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    #Initialize Particles
    X = rng.random((num_particles, dim)) * (ub - lb) + lb
    V = rng.normal(scale=0.1, size=(num_particles, dim))

    #Initial fitness
    fitness = np.apply_along_axis(obj_func, 1, X)

    pbest_pos = X.copy()
    pbest_val = fitness.copy()

    best_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[best_idx].copy()
    gbest_val = pbest_val[best_idx]

    convergence_curve = [gbest_val]

    #Main Loop
    for t in range(1, iter + 1):
        r1 = rng.random((num_particles, dim))
        r2 = rng.random((num_particles, dim))

        cognitive = c1 * r1 * (pbest_pos - X)
        social = c2 * r2 * (gbest_pos - X)

        V = w * V + cognitive + social
        X = X + V

        #Clip if boundary hit
        out_low = X < lb
        out_high = X > ub
        X = np.clip(X, lb, ub)
        V[out_low | out_high] *= -0.5

        fitness = np.apply_along_axis(obj_func, 1, X)

        improved = fitness < pbest_val
        pbest_pos[improved] = X[improved]
        pbest_val[improved] = fitness[improved]

        best_idx = np.argmin(pbest_val)
        if pbest_val[best_idx] < gbest_val:
            gbest_val = pbest_val[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()

        convergence_curve.append(gbest_val)

    return gbest_pos, gbest_val, convergence_curve