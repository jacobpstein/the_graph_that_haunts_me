data {
  int<lower=0> N;                           // number of observations
  int<lower=1> J;                           // number of teams
  array[N] int<lower=1, upper=J> team;      // team index for each obs
  vector[N] x;                              // lagged fg3pct
  vector[N] y;                              // current fg3pct
}
parameters {
  real alpha_bar;                           // grand mean intercept
  real beta;                                // slope for lagged fg3pct
  vector[J] alpha_raw;                      // raw team intercepts
  real<lower=1e-6> sigma_alpha;             // SD of team intercepts (constrained)
  real<lower=1e-6> sigma_y;                 // residual SD (constrained)
}

transformed parameters {
  vector[N] mu;
  vector[J] alpha = alpha_bar + sigma_alpha * alpha_raw;
  for (n in 1:N)
    mu[n] = alpha[team[n]] + beta * x[n];
}
model {
  // Priors
  alpha_bar ~ normal(0.36, 0.05); // this is basically the typical 3pt percentage 
  beta ~ normal(0, 1);
  sigma_alpha ~ exponential(1);
  sigma_y ~ exponential(1);
  alpha_raw ~ normal(0, 1);

  // Likelihood
  y ~ normal(mu, sigma_y);
}
generated quantities {
  real r2;
  {
    real var_y = variance(y);
    real var_res = sigma_y^2;
    r2 = 1 - var_res / var_y;
  }
}
