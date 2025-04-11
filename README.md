<div style="text-align: center;">
    <h1 style="font-size: 20px;">
        Final project for the <strong style="font-size: 26px;">Information theory</strong> course<br>
        <strong style="font-size: 46px; display: block; margin-top: 20px;">Inferring change points in the spread of COVID-19</strong>
    </h1>
</div>

<hr style="border: 1px solid #000;">

<div style="display: flex; justify-content: center; align-items: center; width: 100%; text-align: center; position: relative;">
    <img src="figures/unipd_template.png" alt="Unipd template" style="max-width: 150px; height: auto; margin-left:100px">
    <img src="figures/pod_template_transparent.png" alt="PoD template" style="max-width: 150px; height: auto; margin-left: 50px;">
</div>



<hr style="border: 1px solid #000;">

<div style="text-align: left; margin: 0 auto; width: 80%;">
    <p><strong style="font-size: 26px;">University of Padua - Department of Physics and Astronomy</strong></p>
    <p><strong>Degree course:</strong> Physics of Data</p>
    <p><strong>Course:</strong> Information Theory </p>
    <p><strong>Year:</strong> 2024-2025</p>
    <p><strong>Professor in charge:</strong> Michele Allegra </p>
    <table style="margin: 20px auto; border-collapse: collapse; width: 100%; font-size: 15px; text-align: left;">
        <thead>
            <tr>
                <th colspan="3" style="border: none; padding: 8px; text-align: center; font-weight: bold; font-size: 20px">Students</th>
            </tr>
            <tr>
                <th style="border: none; padding: 8px;">Name</th>
                <th style="border: none; padding: 8px;">ID</th>
                <th style="border: none; padding: 8px;">Email</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: none; padding: 8px;">Ada D'Iorio</td>
                <td style="border: none; padding: 8px;">2092320</td>
                <td style="border: none; padding: 8px;">ada.diorio@studenti.unipd.it</td>
            </tr>
            <tr>
                <td style="border: none; padding: 8px;">Anna Garbo</td>
                <td style="border: none; padding: 8px;">2091751</td>
                <td style="border: none; padding: 8px;">anna.garbo.2@studenti.unipd.it</td>
            </tr>
        </tbody>
    </table>
</div>

### Abstract
In this project two different sampling techninques will be tested. In particular they will be tested in context of SIR model for COVID-19 pandemic at early stage, indeed the reference comes from the article [*Inferring change points in the spread of COVID-19 reveals the effectiveness of interventions*](https://www.science.org/doi/10.1126/science.abb9789) by Jonas Dehning et al. 

The parameter of the SIR model will be estimated through inference techniques. For the Likelihood computation we used synthetic data assuming that the prevalvence (number of infected indivduals during time) followed the SIR model. The article propose a sofisticated model in which the classic SIR model is modified adding changing points for the transmission rate and the estimation of the delay for reporting cases, which is a charachteristic we all observe during COVID-19 pandemic. Considering this we tested the two tecninques in different steps, first for classical SIR model, then for SIR model with the introduction of the delay for reported cases, then with one changing points and no delay and so on until we get the complete model showed in the article.

The two algorithm that have been tested are Metropolis Hastings for **Markov Chain MonteCarlo** and **NUTS** (Not U Turning Sampler) developed in **PYMC** package for python language. What we can observe is that NUTS better explore the parameter space than the Metropolis Hastings algorithm since it can better deal with the denominated 'curse of dimensionality'.  

Here the theoric framework we are dealing with.

## Bayesian Inference

Bayesian inference is a statistical approach in which we update the knowledge that we have about model parameters, described with $\theta$, based on the observed data. 
The final result is the so called **posterior distribution**, given by a combination of our initial assumption (**the prior distribution**) with the evidence (**likelihood**).

### Posterior Distribution (Bayes' rule):

$
p(\theta \mid D) = \frac{p(D \mid \theta) p(\theta)}{p(D)} \propto p(D \mid \theta) p(\theta)
$

Where:
- $(p(D \mid \theta))$ is the **likelihood**.
- $(p(\theta))$ is the **prior distribution**.
- $(p(D))$ is the normalization constant (not needed explicitly in Metropolis-Hastings).

### Steps of the Metropolis-Hastings Algorithm:

Given a current state $(\theta^{(t)})$, the algorithm performs the following steps:

1. Propose a new state $(\theta^*)$ from the proposal distribution $(q(\theta^* \mid \theta^{(t)}))$.

2. Compute the acceptance ratio:
$[
\alpha = \frac{p(D \mid \theta^*) p(\theta^*) q(\theta^{(t)} \mid \theta^*)}{p(D \mid \theta^{(t)}) p(\theta^{(t)}) q(\theta^* \mid \theta^{(t)})}
]$

3. Accept the new state $(\theta^*)$ with probability:
$[
A(\theta^{(t)} \rightarrow \theta^*) = \min\left(1, \alpha\right)
]$

4. If accepted, set $(\theta^{(t+1)} = \theta^*)$, otherwise set $(\theta^{(t+1)} = \theta^{(t)})$.

By repeating these steps, we generate a chain of samples:
$[
\theta^{(0)} \rightarrow \theta^{(1)} \rightarrow \dots \rightarrow \theta^{(N)}
]$

These samples approximate the posterior distribution $(p(\theta \mid D)$), also called limiting distribution of the chain.



### How NUTS Works:

Hamiltonian Monte Carlo is an efficient sampling technique used to implement MCMC algorithms. It employs Hamiltonian dynamics in order to construct the chain. It needs two hyperparameters in order to work efficiently:
- trajectory length (number of integrating steps);
- step size 

NUTS automates the selection of trajectory length in Hamiltonian Monte Carlo by preventing the trajectory from "turning back" on itself (U-turn), thereby improving sampling efficiency.

Given a current state $(\theta^{(t)})$, NUTS performs the following steps:

1. Sample momentum $(r)$ from a normal distribution:
$[
r \sim \mathcal{N}(0, M)
]$
where $(M)$ is typically the identity matrix or an adaptive mass matrix.

2. Evolve the system using Hamiltonian dynamics with leapfrog integration:
$[
\frac{d\theta}{dt} = M^{-1}r, \quad \frac{dr}{dt} = -\nabla U(\theta)
]$
where $(U(\theta) = -\log p(D \mid \theta) - \log p(\theta))$.

3. At each iteration the algorithm is evolving forward and backward in time building a binary tree of points. After each one, it veryfies the **U-turn** criterion:


$[
(\theta^{+} - \theta^{-}) \cdot r^{-} < 0 \quad \text{or} \quad (\theta^{+} - \theta^{-}) \cdot r^{+} < 0
]$

It is checking if the last movement direction is going accordingly to the general one; if it is not respected, the execution is stopped.

4. Choose a new candidate state $(\theta^*)$ uniformly from the set of points explored.

5. Accept $(\theta^*)$ based on the Hamiltonian Metropolis acceptance criterion (energy conservation).

Repeating this process generates a sequence of samples:
$[
\theta^{(0)} \rightarrow \theta^{(1)} \rightarrow \dots \rightarrow \theta^{(N)}
]$
that are approximating the posterior distribution $(p(\theta \mid D))$.

##  Description of the Functions Used in the SIR Model

This notebook contains a complete implementation of a **discrete-time Susceptible-Infected-Recovered (SIR)** model, extended to include realistic dynamics such as reporting delays, gradual transitions in transmission rate, weekend effects, and Bayesian inference.

---

###  Auxiliary Functions

- **`smooth_step_function(start_val, end_val, t_begin, t_end, t_total)`**  
  Produces a gradual linear transition from `start_val` to `end_val` between time steps `t_begin` and `t_end`. Useful to simulate progressive policy changes or behavior shifts over time.

- **`pt_lognormal(x, mu, sigma)`**  
  Constructs a normalized log-normal distribution over `x`, centered at `mu` with scale `sigma`. Used to apply probabilistic delays in the reporting of infections.

- **`make_delay_matrix(n_rows, n_columns)`**  
  Creates a matrix where each element `[i, j]` encodes the delay from time `i` to `j`. Negative delays are clipped to ensure causality.

- **`apply_delay(array, delay, sigma_delay, delay_mat)`**  
  Applies the delay effect on a given array of new infections using a log-normal distribution over the provided delay matrix.

- **`delay_cases_lognormal(input_arr, len_input_arr, len_output_arr, median_delay, scale_delay)`**  
  Combines the previous delay functions into a one-step wrapper to compute observed cases with probabilistic delays.

---

###  Discrete-Time SIR Model Variants

Let $( S(t) )$, $( I(t) )$, and $( R(t) )$ represent the number of susceptible, infected, and recovered individuals at time $( t )$, with:


\begin{aligned}
I_{\text{new}}(t) &= \beta(t) \cdot \frac{S(t-1) \cdot I(t-1)}{N} \\
R_{\text{new}}(t) &= \gamma \cdot I(t-1)
\end{aligned}



- **`sir_discrete_delay_w_switch(beta_before, beta_after, gamma, delay, I0, scale, t_switch, dt_switch, t_max, N)`**  
  Introduces a smooth transition in \( $\beta(t)$ \) from `beta_before` to `beta_after` using a sigmoid function:


\begin{aligned}
  \beta(t) = \beta_{\text{before}} \cdot (1 - T(t)) + \beta_{\text{after}} \cdot T(t), \\ \quad T(t) = \frac{1}{1 + \exp\left( -\frac{t - t_{\text{switch}}}{\Delta t} \right)}
 \end{aligned} 

---

###  Bayesian Inference with MCMC

- **`metropolis_hastings(log_posterior, initial_params, data, n_iter, step_sizes)`**  
  Implements the classic Metropolis-Hastings algorithm to draw samples from the posterior distribution of model parameters.

- **`metropolis_hastings_multi_chain(...)`**  
  Executes multiple independent MCMC chains using different initial guesses and step sizes to improve parameter exploration and convergence diagnostics.



This suite of functions provides a flexible framework for epidemic modeling, allowing for both simulation and inference under realistic assumptions such as delays, gradual interventions, and reporting biases.
