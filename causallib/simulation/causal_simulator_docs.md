# CausalSimulator3 Class - Detailed Documentation

## Overview

The `CausalSimulator3` class is a sophisticated simulator for generating synthetic causal inference datasets. It's designed to create observational data with known ground truth causal relationships, making it ideal for benchmarking and testing causal inference algorithms (like those used in the ACIC 2016 challenge).

---

## Why a Class Instead of Functions?

### Reasons for Using a Class Structure:

1. **State Management**: The simulator maintains complex state including:
   - Graph topology (adjacency matrix)
   - Variable types and properties
   - Link functions for each variable
   - Signal-to-noise ratios
   - Treatment assignment mechanisms

2. **Reusability**: Once initialized with a specific causal structure, you can generate multiple datasets with the same underlying causal mechanism:
   ```python
   sim = CausalSimulator3(topology, ...)
   data1 = sim.generate_data(n_samples=1000)
   data2 = sim.generate_data(n_samples=500)  # Same mechanism, different sample
   ```

3. **Encapsulation**: The class bundles related data generation methods (treatment assignment, outcome generation, linking functions) with their configuration, making the code more maintainable and modular.

4. **Polymorphism**: Can be extended or modified without affecting calling code, following object-oriented design principles.

---

## Class Initialization

```python
CausalSimulator3(
    topology,
    var_types,
    prob_categories,
    link_types,
    snr,
    treatment_importances,
    treatment_methods='gaussian',
    outcome_types='categorical',
    effect_sizes=None,
    survival_distribution='expon',
    survival_baseline=1,
    params=None
)
```

### Parameters:

- **topology** (np.ndarray): Boolean adjacency matrix where `topology[i,j] = True` iff `j` is a parent of `i`
  - Defines the causal graph structure
  - Must be acyclic (DAG)

- **var_types** (Sequence[str]): Type of each variable
  - Options: `"covariate"`, `"hidden"`, `"outcome"`, `"treatment"`, `"censor"`
  - Hidden variables create unobserved confounding

- **prob_categories** (Sequence[float|None]): 
  - For categorical variables: list of category probabilities
  - For continuous variables: None
  - Example: `[0.3, 0.7]` for binary, `[0.25, 0.25, 0.5]` for 3 categories

- **link_types** (Sequence[str]): How parents combine to generate each variable
  - Options: `'linear'`, `'log'`, `'poly'`, `'affine'`, `'exp'`

- **snr** (float): Signal-to-noise ratio (typically 0.5-0.9)
  - Controls how much of variance is explained vs random noise

- **treatment_importances** (float or array): Weight of treatment effect on outcome
  - Values typically between 0 and 1
  - treatment_importance = 0.8 indicates that the outcome will be affected 80% by treatment and 20% by all other predecessors.


- **treatment_methods** (str): Method for treatment assignment
  - Options: `'gaussian'`, `'logistic'`, `'odds_ratio'`, `'quantile_gauss_fit'`, `'random'`

- **outcome_types** (str): Type of outcome variable
  - Options: `'categorical'` (binary), `'continuous'`, `'survival'`

---

## Key Class Attributes

### Static Dictionaries (Class-Level Constants):

```python
G_LINKING_METHODS = {
    'linear': lambda function,
    'log': lambda function,
    'poly': lambda function,
    'affine': lambda function,
    'exp': lambda function
}

TREATMENT_METHODS = {
    'random': lambda function,
    'gaussian': lambda function,
    'logistic': lambda function,
    'odds_ratio': lambda function,
    'quantile_gauss_fit': lambda function
}

O_LINKING_METHODS = {
    'marginal_structural_model': lambda function,
    None: lambda function (identity)
}
```

These dictionaries map method names to their implementation functions, enabling dynamic selection of data generation strategies.

---

## Core Methods

### 1. `generate_data(num_samples, treatment_values=None, counterfactuals=None)`

**Goal**: Generate a complete synthetic dataset with known causal structure.

--> *implements the SCM

--> user supplies either a dataset or the number of samples (num_samples) that will be generated. If both are supplied, num_samples ius ignored and the X_given (baseline dataset) is used.

--> 

**Inner Workings**:
1. **Topological Sorting**: Determines the order to generate variables (parents before children)
2. **Sequential Generation**: For each variable in topological order:
   - Identify its parents from the adjacency matrix
   - Apply the appropriate linking function to combine parent values
   - Add noise based on SNR
   - For categorical variables, threshold the continuous signal
3. **Treatment Assignment**: Uses specified method (gaussian, logistic, etc.)
4. **Outcome Generation**: Generates both observed and counterfactual outcomes
5. **Returns**: Tuple of (X_covariates, propensities, (y0, y1))
   - X: DataFrame of covariates
   - propensities: Treatment assignment probabilities
   - (y0, y1): Potential outcomes under control and treatment

**Example Flow**:
```
For variable Y with parents [X1, X2, Treatment]:
1. Get parent values: parent_data = data[[X1, X2, Treatment]]
2. Apply link function: signal = linear_link(parent_data, coefficients)
3. Add noise: noisy_signal = signal + noise * (1-snr)
4. Threshold (if categorical): Y = (noisy_signal > threshold)
```

### 2. `generate_random_topology(n_covariates, p, n_treatments=1, n_outcomes=1, ...)`

**Goal**: Create a random causal graph structure suitable for simulation.

**Inner Workings**:
1. **Graph Model**: a random graph model G(n,p)
   - Each potential edge exists with probability `p`
2. **Ensures DAG Property**: 
   - Orders variables: given_vars → covariates → treatments → outcomes → censoring
   - Only allows edges from earlier to later variables
3. **Hidden Variables**: Converts some covariates to hidden with probability `p_hidden`
4. **Returns**: 
   - Boolean topology matrix
   - pandas Series of variable types with names

**Parameters**:
- `n_covariates`: Number of observed features
- `p`: Edge probability (density of connections)
- `n_treatments`, `n_outcomes`, `n_censoring`: Number of special nodes
- `given_vars`: Pre-existing variables (from real data)
- `p_hidden`: Probability of making a covariate unobserved

### 3. Treatment Assignment Methods

#### `_treatment_gaussian_dichotomous(X_parents, prob_category, snr)`

**Goal**: Assign binary treatment based on linear combination of parents.

**Inner Workings**:
1. Compute propensity score: `logit(p) = β'X_parents`
2. Generate treatment by thresholding: `A = (Uniform(0,1) < p)`
3. Returns DataFrame with probabilities for control and treatment

#### `_treatment_logistic_dichotomous(X_parents, prob_category, snr)`

**Goal**: Use logistic model for treatment assignment with specified propensity.

**Inner Workings**:
1. Fit logistic curve to data
2. Ensure specified proportion receives treatment
3. Handles confounding by adjusting intercept

#### `_treatment_odds_ratio(X_parents, prob_category, snr)`

**Goal**: Generate treatment with specific odds ratio relative to covariates.

**Inner Workings**:
1. Calculate baseline odds from prob_category
2. Modify odds based on parent values and desired OR
3. Convert odds to probabilities

#### `_treatment_random(X_parents, prob_category)`

**Goal**: Random treatment assignment (like an RCT).

**Inner Workings**:
- Simply samples from Bernoulli(p) independent of covariates
- No confounding

### 4. Linking Functions

#### `_linear_linking(X_parents, beta=None)`

**Goal**: Linear combination of parent variables.

**Inner Workings**:
```
X_new = β₀ + β₁*X₁ + β₂*X₂ + ... + βₙ*Xₙ
```
- If beta not provided, samples from standard normal
- Returns (signal, coefficients)

#### `_log_linking(X_parents, beta=None)`

**Goal**: Logarithmic transformation of linear combination.

**Inner Workings**:
```
X_new = log(1 + |linear_combination|)
```
- Handles negative values by using absolute value
- Compresses large values

#### `_poly_linking(X_parents, beta=None)`

**Goal**: Polynomial (quadratic) relationships.

**Inner Workings**:
```
X_new = β₀ + Σ βᵢXᵢ + Σ βᵢⱼXᵢXⱼ
```
- Creates interaction terms
- Adds non-linearity

#### `_affine_link(X_parents, beta=None)`

**Goal**: Shifted and scaled linear combination.

**Inner Workings**:
- Similar to linear but with additional transformation
- Can model different scales and locations

#### `_exp_linking(X_parents, beta=None)`

**Goal**: Exponential transformation.

**Inner Workings**:
```
X_new = exp(linear_combination / scale)
```
- Creates strong non-linearity
- Can model multiplicative effects

### 5. `format_for_training(X, propensities, cf, headers_chars=None, exclude_hidden_vars=True)`

**Goal**: Prepare generated data for use in causal inference algorithms.

**Inner Workings**:
1. **Separate Variables**: 
   - Observed covariates (X)
   - Treatment assignment (A)
   - Observed outcomes (Y)
   - Counterfactual outcomes (Y0, Y1)

2. **Handle Hidden Variables**:
   - If `exclude_hidden_vars=True`, removes unobserved confounders
   - Otherwise, includes them (for debugging)

3. **Returns**: Two DataFrames
   - **Observed data**: (X, A, Y) - what you'd see in practice
   - **Counterfactual data**: (Y0, Y1) - ground truth for evaluation

---

## Usage Example

```python
import numpy as np
from causallib.datasets import CausalSimulator

# Define causal structure
#     Hidden
#      ↓   ↓
#   Cov → Treatment → Outcome

topology = np.zeros((4, 4), dtype=bool)
topology[1, 0] = True  # Hidden → Covariate
topology[2, 0] = True  # Hidden → Treatment
topology[2, 1] = True  # Covariate → Treatment
topology[3, 2] = True  # Treatment → Outcome

var_types = ["hidden", "covariate", "treatment", "outcome"]
link_types = ['linear', 'linear', 'linear', 'linear']
prob_categories = [[0.5, 0.5], None, [0.5, 0.5], [0.4, 0.6]]

sim = CausalSimulator(
    topology=topology,
    prob_categories=prob_categories,
    link_types=link_types,
    snr=0.9,
    var_types=var_types,
    treatment_importances=0.8,
    outcome_types="binary",
    treatment_methods="gaussian"
)

# Generate data
X, propensities, (y0, y1) = sim.generate_data(num_samples=1000)
```

---

## Key Design Decisions

1. **Flexible Graph Structure**: The adjacency matrix allows arbitrary DAGs, supporting complex confounding scenarios.

2. **Multiple Treatment Mechanisms**: Different treatment methods model various types of treatment assignment (observational vs RCT-like).

3. **Ground Truth Access**: Returns counterfactual outcomes, enabling evaluation of causal methods.

4. **Modular Linking Functions**: Easy to extend with new functional forms.

5. **Signal-to-Noise Control**: The SNR parameter allows controlling how "noisy" vs "deterministic" the relationships are.

---

## Practical Applications

- **Benchmarking**: Compare causal inference algorithms on data with known ground truth
- **Method Development**: Test new estimators before applying to real data
- **Education**: Demonstrate concepts like confounding, effect heterogeneity
- **ACIC Challenges**: Generate standardized datasets for competitions
- **Sensitivity Analysis**: Understand how algorithms perform under different causal structures