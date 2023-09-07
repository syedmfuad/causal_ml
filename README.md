# causal_ml
Codes for Causal ML paper 

## ML_causal_code_simulation.R 

Simulates different data structures based on DGP. The baseline DGP is structured as following:

$`y_i=\theta d_i+x_i' \beta + u_i`$

$`d_i=x_i' \beta + v_i`$

The treatment effect $`\theta`$ is set to $`\theta=1`$. The $`d_i`$ represents the binary treatment variable (approximately 50% of the observations receive treatment); $`x_i`$ represents a vector of k covariates, generated from a multivariate normal distribution; $`\beta`$ is a vector of k parameters. 

The baseline DGP is defined as $`n=150`$, $`k=10`$, $`\theta=1`$ (homogenous treatment). We can then systematically vary these parameters to progressively more challenging estimation problems. In particular, we vary the following parameters, while keeping the others constant: 
* Increase sample size $`(n=150, 500, 5000, 15000)`$
* Increase number of covariates $`(k=10, 100)`$
* Impose treatment heterogeneity $`(\theta=1; \theta \tilde Normal(1,1))`$
* Change structure of data $`(y_i = \theta d_i + x_i' \beta + u_i)`$ and $`(y_i = \theta d_i + sin(x_i' \beta) + u_i)`$


