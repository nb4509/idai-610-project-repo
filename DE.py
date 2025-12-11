import numpy as np

def de(
    obj_func,
    bounds,
    pop_size=30,
    max_iter=500,
    F=0.5,          # differential weight
    CR=0.9,         # crossover rate
    random_state=None,
):
    rng = np.random.default_rng(random_state)

    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    if pop_size < 4:
        raise ValueError("pop_size must be at least 4 for DE/rand/1.")

    #Initialization of population along bounds
    pop = rng.random((pop_size, dim)) * (ub - lb) + lb
    fitness = np.apply_along_axis(obj_func, 1, pop)

    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_score = fitness[best_idx]
    convergence_curve = [best_score]

    #Main loop
    for it in range(1, max_iter + 1):
        new_pop = np.empty_like(pop)
        new_fitness = np.empty_like(fitness)

        for i in range(pop_size):
            # choose 3 distinct indices different from i
            idxs = np.arange(pop_size)
            rng.shuffle(idxs)
            r1, r2, r3 = idxs[:3]
            if r1 == i or r2 == i or r3 == i:
                # ensure all distinct and not i
                choices = [idx for idx in idxs if idx != i]
                r1, r2, r3 = choices[:3]

            a = pop[r1]
            b = pop[r2]
            c = pop[r3]

            mutant = a + F * (b - c)

            #clip mutant to bounds to keep it feasible
            mutant = np.clip(mutant, lb, ub)

            trial = pop[i].copy()
            j_rand = rng.integers(0, dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            f_trial = obj_func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
            else:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]

        #replace population with new population on those with higher fitness
        pop = new_pop
        fitness = new_fitness

        #Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_score:
            best_score = fitness[best_idx]
            best_pos = pop[best_idx].copy()

        convergence_curve.append(best_score)

    return best_pos, best_score, convergence_curve