import numpy as np
import matplotlib.pyplot as plt

def metropolis(H, x0, steps, B, delta):
    x = x0
    samples = []
    
    for _ in range(steps):
        x_new = x + np.random.uniform(-delta, delta)
        dH = H(x_new) - H(x)
        
        if dH < 0 or np.random.rand() < np.exp(-dH * B):
            x = x_new  
        
        samples.append(x)
    
    return np.array(samples)

# Define energy function H(x) = x^2
def H(x):
    return x**2

# Parameters
T = 300
x0 = 10         # Initial state
steps = 10000    # Number of iterations
B = 1            # Inverse temperature parameter
delta = 1.0      # Step size

# Run Metropolis algorithm
samples = metropolis(H, x0, steps, B, delta)

# Compute expectation value, standard deviation, and standard error of H(x)
H_values = H(samples)
expectation_H = np.mean(H_values)
std_H = np.std(H_values, ddof=1)
std_error_H = std_H / np.sqrt(len(H_values))

# Plot Monte Carlo history to show thermalization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(samples, alpha=0.7, color='b')
plt.axhline(np.sqrt(expectation_H), color='g', linestyle='--', linewidth=2, label=f'⟨H(x)⟩ = {expectation_H:.3f} ± {std_error_H:.3f}')
plt.xlabel('Monte Carlo Step')
plt.ylabel('x')
plt.title('Monte Carlo History')
plt.legend()

# Plot histogram of sampled values
plt.subplot(1, 2, 2)
plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', label='Metropolis Samples')

# Theoretical distribution (Boltzmann-like)
x = np.linspace(-3, 3, 1000)
p_x = np.exp(-H(x) * B)
p_x /= np.trapz(p_x, x)  # Normalize
plt.plot(x, p_x, 'r-', label='Theoretical Distribution')

# Plot expectation value as a vertical line
plt.axvline(np.sqrt(expectation_H), color='g', linestyle='--', linewidth=2, 
            label=f'⟨H(x)⟩ = {expectation_H:.3f} ± {std_error_H:.3f}')

plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.title('Metropolis Sampling for H(x) = x^2')
plt.tight_layout()
plt.show()

