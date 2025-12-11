import numpy as np

def sfoa(
    obj_func,
    bounds,
    n_starfish=30,
    iter=500,
    gp=0.5,
    random_state=None
):
    rng = np.random.default_rng(random_state)

    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    #Initialization
    #Create matrix X with number of columns equal to dimensions and number of rows equal to starfish
    #Scale by width of domain the add lower bounds to ensuree the correct starting point for everything
    X = rng.random((n_starfish, dim)) * (ub - lb) + lb
    #Apply fitness function to each row X
    fitness = np.apply_along_axis(obj_func, 1, X)

    #Current best index is the max of fitness
    best_idx = np.argmin(fitness)
    #Copy row values of X at best index of fitness
    best_pos = X[best_idx].copy()
    #Grab best score from the fitness list
    best_score = fitness[best_idx]

    #Initial best for convergence curve
    convergence_curve = [best_score]

    #Main Loop
    for t in range(1, iter + 1):
        #Create new matrix that's empty in the same shape as current
        new_X = np.empty_like(X)

        for i in range(n_starfish):
            #For each starfish decide between exploring or exploiting
            if rng.random() < gp:
                #Exploration Phase
                if dim > 5:
                    #5 Dimensional Search
                    #Choose 5 random dimensions to update
                    p_idx = rng.choice(dim, size=5, replace=False)
                    #Calculate angular progression
                    a1 = (2 * rng.random() - 1) * np.pi
                    h = (np.pi / 2.0) * (t / iter)

                    #Choose randomly between updating new position along a cosine or sine trajectory
                    if rng.random() < 0.5:
                        new_vals = (
                            X[i, p_idx]
                            + a1 * (best_pos[p_idx] - X[i, p_idx]) * np.cos(h)
                        )
                    else:
                        new_vals = (
                            X[i, p_idx]
                            - a1 * (best_pos[p_idx] - X[i, p_idx]) * np.sin(h)
                        )
                    
                    #Create temporary copy of starfish and change it's position values to the newly calculated values
                    temp = X[i].copy()
                    temp[p_idx] = new_vals


                    #Chekc if it's out of bounds
                    out_low = temp < lb
                    out_high = temp > ub

                    temp[out_low | out_high] = X[i, out_low | out_high]
                    
                    #Add starfish position update to new matrix
                    X_new = temp
                else:
                    #Unidimensional Search
                    p = rng.integers(0, dim)  # one dimension
                    k1, k2 = rng.choice(n_starfish, size=2, replace=False)
                    A1, A2 = rng.uniform(-1.0, 1.0, size=2)

                    h = (np.pi / 2.0) * (t / iter)
                    Et = ((iter - t) / iter) * np.cos(h)

                    temp = X[i].copy()
                    temp[p] = (
                        Et * X[i, p]
                        + A1 * (X[k1, p] - X[i, p])
                        + A2 * (X[k2, p] - X[i, p])
                    )

                    # Boundary handling: revert if out of bounds in that dim
                    if not (lb[p] <= temp[p] <= ub[p]):
                        temp[p] = X[i, p]

                    X_new = temp
            else:
                #Exploitation Phase
                if i < n_starfish - 1:
                    #Preying Behavior
                    #Create list of starfish random dimension coordinates
                    m_idx = rng.choice(n_starfish, size=5, replace=False)
                    dm = [best_pos - X[m] for m in m_idx]
                    #Randomly choose between two of those 5
                    dm1 = dm[rng.integers(0, 5)]
                    dm2 = dm[rng.integers(0, 5)]
                    r1 = rng.random()
                    r2 = rng.random()
                    temp = X[i] + r1 * dm1 + r2 * dm2
                    r1 = rng.random()
                    r2 = rng.random()
                    temp = X[i] + r1 * dm1 + r2 * dm2
                    #Move starfish along weighted vector between those two vectors
                else:
                    #Regeneration
                    #Decay starfish that is furthest from global
                    decay = np.exp(-(t * n_starfish / iter))
                    temp = decay * X[i]
                #Clip bounds
                X_new = np.clip(temp, lb, ub)
            #Set new starfish positions in overall matrix
            new_X[i] = X_new
        
        #Evaluate fitness of new population
        new_fitness = np.apply_along_axis(obj_func, 1, new_X)

        #Greedy take starfish where fitness is minimized
        for i in range(n_starfish):
            if new_fitness[i] < fitness[i]:
                fitness[i] = new_fitness[i]
                X[i] = new_X[i]
        
        #Update Global Best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_score:
            best_score = fitness[best_idx]
            best_pos = X[best_idx].copy()
        
        convergence_curve.append(best_score)

    return best_pos, best_score, convergence_curve

#Benchmark Problem 1
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def sphere_bounds(dim: int, lower: float = -100.0, upper: float = 100.0):
    return [(lower, upper)] * dim

#Benchmark Problem 2
def wsn_coverage_fitness(
        x: np.ndarray,
        num_sensors: int = 10,
        radius: float = 15.0,
        area_size: float = 100.0,
        grid_res: int = 50
) -> float:
    sensors = x.reshape(num_sensors, 2)

    xs = np.linspace(0.0, area_size,  grid_res)
    ys = np.linspace(0.0, area_size, grid_res)
    xx, yy = np.meshgrid(xs, ys)
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    r2 = radius * radius
    covered = np.zeros(len(points), dtype=bool)

    for sx, sy in sensors:
        d2 = (points[:, 0] - sx) ** 2 + (points[:, 1] - sy) ** 2
        covered |= d2 <= r2
    
    coverage_fraction = covered.mean()

    return float(1.0 - coverage_fraction)

def wsn_bounds(num_sensors: int, area_size: float = 100.0):
    return [(0.0, area_size)] * (2 * num_sensors)



