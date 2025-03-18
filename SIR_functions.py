import pytensor.tensor as pt 
import numpy as np
import pytensor  

def smooth_step_function(start_val, end_val, t_begin, t_end, t_total):
    """
        Instead of going from start_val to end_val in one step, make the change a
        gradual linear slope.

        Parameters
        ----------
            start_val : number
                Starting value

            end_val : number
                Target value

            t_begin : number or array (theano)
                Time point (inbetween 0 and t_total) where start_val is placed

            t_end : number or array (theano)
                Time point (inbetween 0 and t_total) where end_val is placed

            t_total : integer
                Total number of time points

        Returns
        -------
            : theano vector
                vector of length t_total with the values of the parameterised f(t)
    """
    t = np.arange(t_total)

    return (
        pt.clip((t - t_begin) / (t_end - t_begin), 0, 1) * (end_val - start_val)
        + start_val
    )

# ---- LOGNORMAL DISTRIBUTION ----
def pt_lognormal(x, mu, sigma):
    distr = 1 / x * pt.exp(-((pt.log(x) - mu) ** 2) / (2 * sigma ** 2))
    return distr / pt.sum(distr, axis=0)

# ---- DELAY FUNCTIONS ----
def make_delay_matrix(n_rows, n_columns):
    mat = np.zeros((n_rows, n_columns))
    for i in range(n_rows):
        for j in range(n_columns):
            mat_val = j - i
            mat_val = np.maximum(mat_val, 0.01)
            mat_val = np.abs(mat_val)
            mat[i, j] = mat_val
    return mat

def apply_delay(array, delay, sigma_delay, delay_mat):
    mat = pt_lognormal(delay_mat, mu=pt.log(delay), sigma=sigma_delay)
    return pt.dot(array, mat)


def delay_cases_lognormal(input_arr, len_input_arr, len_output_arr, median_delay, scale_delay):
    delay_mat = make_delay_matrix(len_input_arr, len_output_arr)
    delayed_arr = apply_delay(input_arr, median_delay, scale_delay, delay_mat)
    return delayed_arr


# Defining SIR model: simplest as possible - no switches and no delay 
def sir_discrete(beta, gamma, I0, scale, t_max, N):
    """ Simulates a discrete-time SIR model incorporating a reporting delay D """
    
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    I_new = np.zeros(t_max)  # New infections per day

    # Initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for t in range(1, t_max):
        # Compute new infections and recoveries
        I_new[t] = beta * S[t-1] * I[t-1] / N  # New infections
        R_new = gamma * I[t-1]  # New recoveries
        
        # Update compartments
        S[t] = S[t-1] - I_new[t]
        I[t] = I[t-1] + I_new[t] - R_new
        R[t] = R[t-1] + R_new
        
    
    return I  # Return both actual infections and delayed reported cases


# ---- Discrete-time SIR MODEL with delay ----
def sir_discrete_delay_no_switch(beta, gamma, delay, I0, scale,t_max, N):
    """ Simulates a discrete-time SIR model incorporating a reporting delay D """
    
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    I_new = np.zeros(t_max)  # New infections per day
    C = np.zeros(t_max)  # Reported cases (delayed)

    # Initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for t in range(1, t_max):
        # Compute new infections and recoveries
        I_new[t] = beta * S[t-1] * I[t-1] / N  # New infections
        R_new = gamma * I[t-1]  # New recoveries
        
        # Update compartments
        S[t] = S[t-1] - I_new[t]
        I[t] = I[t-1] + I_new[t] - R_new
        R[t] = R[t-1] + R_new
        
        # Introduce delay in reported cases
        delay_days = int(np.clip(np.round(delay), 0, t_max-1))  # Ensure valid index
        if t >= delay_days:
            C[t] = I_new[t - delay_days]  # Reported cases with delay
    
    return I, C  # Return both actual infections and delayed reported cases


# ---- Discrete-time SIR MODEL with delay ----
def sir_discrete_delay_w_switch(beta_before, beta_after, gamma, delay, I0, scale, t_switch, dt_switch, t_max, N):
    """ Simulates a discrete-time SIR model incorporating a reporting delay D """
    
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    I_new = np.zeros(t_max)  # New infections per day
    C = np.zeros(t_max)  # Reported cases (delayed)

    # Initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for t in range(1, t_max):
        # Compute new infections and recoveries
        transition = 1 / (1 + np.exp(-(t - t_switch) / dt_switch))
        beta = beta_before * (1 - transition) + beta_after * transition  # Interpolazione pesata

        I_new[t] = beta * S[t-1] * I[t-1] / N  # New infections
        R_new = gamma * I[t-1]  # New recoveries
        
        # Update compartments
        S[t] = S[t-1] - I_new[t]
        I[t] = I[t-1] + I_new[t] - R_new
        R[t] = R[t-1] + R_new
        
        # Introduce delay in reported cases
        delay_days = int(np.clip(np.round(delay), 0, t_max-1))  # Ensure valid index
        if t >= delay_days:
            C[t] = I_new[t - delay_days]  # Reported cases with delay
    
    return I, C  # Return both actual infections and delayed reported cases

# ---- Discrete-time SIR MODEL with delay ----
def sir_discrete_delay_multi_sw(beta_1, beta_2, beta_3, beta_4, gamma, delay, I0, scale, 
                       t_switch1, t_switch2, t_switch3, delta_t, t_max, N):
    """ Simulates a discrete-time SIR model incorporating a reporting delay D """
    
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    I_new = np.zeros(t_max)  # New infections per day
    C = np.zeros(t_max)  # Reported cases (delayed)

    # Initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for t in range(1, t_max):
        # Smooth transitions between beta values
        transition1 = 1 / (1 + np.exp(-(t - t_switch1) / delta_t))
        transition2 = 1 / (1 + np.exp(-(t - t_switch2) / delta_t))
        transition3 = 1 / (1 + np.exp(-(t - t_switch3) / delta_t))

        # Weighted combination of beta values
        beta = (beta_1 * (1 - transition1) + 
                beta_2 * (transition1 * (1 - transition2)) + 
                beta_3 * (transition2 * (1 - transition3)) + 
                beta_4 * transition3)

        I_new[t] = beta * S[t-1] * I[t-1] / N  # New infections
        R_new = gamma * I[t-1]  # New recoveries
        
        # Update compartments
        S[t] = S[t-1] - I_new[t]
        I[t] = I[t-1] + I_new[t] - R_new
        R[t] = R[t-1] + R_new
        
        # Introduce delay in reported cases
        delay_days = int(np.clip(np.round(delay), 0, t_max-1))  # Ensure valid index
        if t >= delay_days:
            C[t] = I_new[t - delay_days]  # Reported cases with delay
    
    return I, C  # Return both actual infections and delayed reported cases

# ---- Discrete-time SIR MODEL with delay ----
def sir_discrete_single_switch(beta_before, beta_after, gamma, I0, scale, t_switch, dt_switch, t_max, N): # t_switch diventa un parametro che deve essere inferito 
    """ Simulates a discrete-time SIR model incorporating a reporting delay D """
    
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    I_new = np.zeros(t_max)  # New infections per day

    # Initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for t in range(1, t_max):
        # Sigmoide per una transizione graduale attorno a t = 50 con larghezza t_switch
        transition = 1 / (1 + np.exp(-(t - t_switch) / dt_switch))
        beta = beta_before * (1 - transition) + beta_after * transition  # Interpolazione pesata
        
        I_new[t] = beta * S[t-1] * I[t-1] / N  # New infections
        R_new = gamma * I[t-1]  # New recoveries
        
        # Update compartments
        S[t] = S[t-1] - I_new[t]
        I[t] = I[t-1] + I_new[t] - R_new
        R[t] = R[t-1] + R_new
        
    return I  # Return actual infections

def sir_discrete_multi_sw(beta_1, beta_2, beta_3, beta_4, gamma, I0, scale, 
                       t_switch1, t_switch2, t_switch3, delta_t, t_max, N):
    """Simulates a discrete-time SIR model incorporating multiple transitions and a reporting delay"""

    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    I_new = np.zeros(t_max)  # New infections per day

    # Initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for t in range(1, t_max):
        # Smooth transitions between beta values
        transition1 = 1 / (1 + np.exp(-(t - t_switch1) / delta_t))
        transition2 = 1 / (1 + np.exp(-(t - t_switch2) / delta_t))
        transition3 = 1 / (1 + np.exp(-(t - t_switch3) / delta_t))

        # Weighted combination of beta values
        beta = (beta_1 * (1 - transition1) + 
                beta_2 * (transition1 * (1 - transition2)) + 
                beta_3 * (transition2 * (1 - transition3)) + 
                beta_4 * transition3)

        I_new[t] = beta * S[t-1] * I[t-1] / N  # New infections
        R_new = gamma * I[t-1]  # New recoveries
        
        # Update compartments
        S[t] = S[t-1] - I_new[t]
        I[t] = I[t-1] + I_new[t] - R_new
        R[t] = R[t-1] + R_new

    return I  # Return both actual infections and delayed reported cases

def metropolis_hastings(log_posterior, initial_params, data, n_iter=5000, step_sizes=None):
    params = np.array(initial_params)
    samples = [params]
    
    if step_sizes is None:
        step_sizes = np.array([0.03, 0.05, 30, 0.05])  # Default step sizes
    
    for _ in range(n_iter):
        proposal = params + np.random.normal(0, step_sizes, size=len(params))
        
        logp_old = log_posterior(params, data)
        logp_new = log_posterior(proposal, data)
        
        accept_ratio = np.exp(logp_new - logp_old)
        if np.random.rand() < accept_ratio:
            params = proposal  # Accept proposal
        
        samples.append(params)
    
    return np.array(samples)

def metropolis_hastings_multi_chain(log_posterior, initial_params_list, step_sizes_list, data, n_iter=5000, n_chains=4):
    """ Esegue più catene di Metropolis-Hastings con step size differenti """
    all_samples = []
    
    for i in range(n_chains):
        print(f"Running chain {i+1}/{n_chains}...")
        initial_params = initial_params_list[i]  # Diversi parametri iniziali
        step_sizes = step_sizes_list[i]  # Diversi step size per ogni catena
        samples = metropolis_hastings(log_posterior, initial_params, data, n_iter, step_sizes)
        all_samples.append(samples)
    
    
    return np.array(all_samples)  # Restituisce tutte le catene

def _SIR_model(lambda_t, mu, S_begin, I_begin, N):
    """
        Implements the susceptible-infected-recovered model

        Parameters
        ----------
        lambda_t : ~numpy.ndarray
            time series of spreading rate, the length of the array sets the
            number of steps to run the model for

        mu : number
            recovery rate

        S_begin : number
            initial number of susceptible at first time step

        I_begin : number
            initial number of infected

        N : number
            population size

        Returns
        -------
        S : array
            time series of the susceptible

        I : array
            time series of the infected

        new_I : array
            time series of the new infected
    """

    new_I_0 = pt.zeros_like(I_begin)

    def next_day(lambda_t, S_t, I_t, _, mu, N):
        new_I_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - mu * I_t
        I_t = pt.clip(I_t, 0, N)  # for stability
        return S_t, I_t, new_I_t

    # theano scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, I, new_I
    outputs, _ = pytensor.scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[S_begin, I_begin, new_I_0],
        non_sequences=[mu, N],
    )
    return outputs