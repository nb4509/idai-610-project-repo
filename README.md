# Metaheuristic Benchmarking: SFOA, PSO, and DE

This project benchmarks three population-based metaheuristic optimization algorithms:

- **SFOA** – Starfish Optimization Algorithm  
- **PSO** – Particle Swarm Optimization  
- **DE** – Differential Evolution  

The algorithms are evaluated with the following benchmark scripts:

1. **Continuous optimization benchmarks**
   - Sphere function
   - Wireless Sensor Network (WSN) coverage optimization
2. **ML hyperparameter optimization**
   - Support Vector Machine (SVM, RBF kernel) on the Iris dataset

Performance is compared using convergence speedd, final fitness, success rates, and Pairwise Wilcoxon Test.

## Benchmarks
### Sphere Function
- Continuous, convex, unimodel test function
- Objective:
    f(x) = sum_{i=1}^{d} x_i^2
- Global Optimum: **0**
- Test abilities on high dimension and smooth mathematical functions
- Benchmarks to compare with industry implementations to confirm implementations work correctly

Metrics:
- Convergence Curves
- Final best fitness
- Success Rate (0.02 within best-known algo)
- Statistical significance (Pairwise Wilcoxon + Holm)

### Wireless Sensor Network (WSN) Coverage Optimization
- Sensors modeled as circles with fixed radius
- Objective:
    Minimize Covered Area
- Fitnesss:
    fitness = 1 - coverage fraction

Metrics:
- Convergence Curves
- Final best fitness
- Boxplots for final best coverage across trials
- Success Rate (0.02 within best-known algo)
- Statistical significance (Pairwise Wilcoxon + Holm)

### SVM Hyperparameter Optimization
- Classifier: **SVM With RBF kernel**
- Dataset: Iris Flowers
- Hyperparameters:
    - log_{10}(C) in bounds [-3, 3]
    - log_{10}(gamma) in bounds [-4, 1]
- Objective:
    Minimize 1 - macro f1 via cross validation

Models Compared:
- Untuned SVM (`C=1.0`, `gamma='scale'`) <- Baseline
- SFOA-tuned SVM
- PSO-tuned SVM
- DE-tuned SVM

Metrics:
- Test accuracy
- Macro F1 score
- Convergence curves

### Visualizations
Project Includes:
- Mean convergence plots
- Boxplots of final fitness
- Success rate
- Statistical Tests:
  - Pairwise Wilcoxon tests
  - Holm-Bonferroni correction

## Module Descriptions
### SFOA
Parameters:
- obj_func - function to evaluate fitness
- bounds - Solution search space
- n_starfish - Initial population count
- iter - number of iterations
- gp - probability of explore or exploit
- random_state - pass rng seed for reproducibility

Returns
- best_pos - returns vector of best starfish
- best_score - returns fitness score of best starfish
- convergence_curve - returns best fitness over all iterations as a list

### PSO
Parameters:
- obj_func - function to evaluate fitness
- bounds - Solution search space
- n_particles - Initial population count
- w - weighting coefficient
- c1 - cognitive coefficient
- c2 - social coefficient
- random_state - pass rng seed for reproducibility

Returns
- gbest_pos - returns vector of best particle
- gbest_val - returns fitness score of best particle
- convergence_curve - returns best fitness over all iterations as a list

### DE
Parameters:
- obj_func - function to evaluate fitness
- bounds - Solution search space
- pop_size - Initial population count
- max_iter - Iteration count
- F - differential weight
- CR - crossover rate
- random_state - pass rng seed for reproducibility

Returns
- best_pos - returns vector of best individual
- best_score - returns fitness score of best individual
- convergence_curve - returns best fitness over all iterations as a list

## How to Run:
### 1. Install Dependencies
pip install -r requirements.txt

### 2. Run Combinatorial Optimization Benchmarks
Can be found in co_benchmark_nb.ipynb
- Default runs 30 different trials with random seed 1 - 30

### 3. Run Classifier Benchmarks
Can be found in clf_benchmark_nb.ipynb
- Default runs 1 trial with random seed 0

## Project Structure
```bash
idai-610-project-repo/
│
├── data/
│   └── iris.csv #Training and tuning data
│
├── results/ #Directory containing plots and report
│   ├── CO_Results.xlsx
│   ├── Final_Fitness.png
│   ├── Sphere_Convergence_Plot.png
│   ├── WSN_Convergence_Curve.png
│   └── Term_Project_Poster.pptx
│
├── clf_benchmark_nb.ipynb #Test script for co problems
├── co_benchmark_nb.ipynb #Test script for SVM hpo
│
├── SFOA.py #Module containing Starfish Optimization Algorithm implementation
├── PSO.py #Module containing Particle Swarm Optimization implementation
├── DE.py #Module containing Differential Evolution implementation
│
├── README.md
├── requirements.txt #pip install -r requirements.txt
```
## References

Ahmad, M. F., Isa, N. A. M., Lim, W. H., & Ang, K. M. (2022). Differential evolution: A recent review based on state-of-the-art works. Alexandria Engineering Journal, 61(5), 3831–3872. https://doi.org/10.1016/j.aej.2022.07.036

Alkhalifa, A. K., Aljebreen, M., Alanazi, R., Ahmad, N., Alrusaini, O., Aljehane, N. O., Alqazzaz, A., & Alkhiri, H. (2025). Leveraging hybrid deep learning with starfish optimization algorithm-based secure mechanism for intelligent edge computing in smart cities environment. Scientific Reports, 15, 33069. https://doi.org/10.1038/s41598-025-11608-4

Sakpere, W., Yisa, F. I., Salami, A., & Olaniyi, G. A. (2025). Particle Swarm Optimization (PSO) and benchmark functions: An extensive analysis. International Journal of Engineering Research in Computer Science and Engineering, 12(1), 1–? (Published 5 January 2025). Retrieved from https://ijercse.com/article/1%20January%2025%20IJERCSE.pdf

Zhong, C., Li, G., Meng, Z., Li, H., Yildiz, A. R., & Mirjalili, S. (2025). Starfish optimization algorithm (SFOA): A bio-inspired metaheuristic algorithm for global optimization compared with 100 optimizers. Neural Computing and Applications, 37, 3641–3683. https://doi.org/10.1007/s00521-024-10694-1