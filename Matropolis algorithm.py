import numpy as np
import matplotlib.pyplot as plt

def metropolis(H, x0, steps, B, delta):
    
    #Metropolis algorithm for sampling from e^(-H(x)/T).
    
    #Parameters:
    #H : function
    #    The energy function H(x).
    #x0 : float
     #   Initial state.
    #steps : int
     #   Number of iterations.
    #T : float
      #  Temperature parameter.
    #delta : float
     #   Maximum step size for the proposal distribution.
    
   # Returns:
    #samples : list
     #   List of sampled values.
    
    x = x0
    samples = []
    
    for _ in range(steps):
        # Propose a new state
        x_new = x + np.random.uniform(-delta, delta)
        
        # Compute energy difference
        dH = H(x_new) - H(x)
        
        # Metropolis criterion
        if dH < 0 or np.random.rand() < np.exp(-dH * B):
            x = x_new  # Accept the new state
        
        samples.append(x)
    
    return samples

# Define energy function H(x) = x^2
def H(x):
    return x**2

# Parameters
T=300
x0 = 0.1         # Initial state
steps = 10000    # Number of iterations
B= 1        # Temperature parameter
delta = 1.0      # Step size

# Run Metropolis algorithm
samples = metropolis(H, x0, steps, B, delta)

# Plot histogram of sampled values
plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', label='Metropolis Samples')

# Theoretical distribution (Boltzmann-like)
x = np.linspace(-3, 3, 1000)
p_x = np.exp(-H(x) *B)
p_x /= np.trapz(p_x, x)  # Normalize
plt.plot(x, p_x, 'r-', label='Theoretical Distribution')

plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.title('Metropolis Sampling for H(x) = x^2')
plt.show()
