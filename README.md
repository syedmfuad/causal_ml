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

## ML_causal_code_Dhurandhar.R 

Causal ML replication of Dhurandhar et al. (2014) 

Dhurandhar, E. J., Dawson, J., Alcorn, A., Larsen, L. H., Thomas, E. A., Cardel, M., ... & Allison, D. B. (2014). The effectiveness of breakfast recommendations on weight loss: a randomized controlled trial. The American Journal of Clinical Nutrition, 100(2), 507-513. 

## ML_causal_code_Bryan.R 

Causal ML replication of Bryan et al. (2014) 

Bryan, G., Chowdhury, S., & Mobarak, A. M. (2014). Underinvestment in a profitable technology: The case of seasonal migration in Bangladesh. Econometrica, 82(5), 1671-1748. 

## 36174-0001-Data.rda 

Dhurandhar et al. (2014) paper data. 

## Round2.dta 

Bryan et al. (2014) paper data. 





