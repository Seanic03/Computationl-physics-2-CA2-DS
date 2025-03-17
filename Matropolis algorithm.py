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
    
    return samples

# Define energy function H(x) = x^2
def H(x):
    return x**2

# Parameters
x0 = 0.0         # Initial state
steps = 10000    # Number of iterations
delta = 1.0      # Step size
B_values = [1, 2, 3, 4]  # Different inverse temperature parameters

plt.figure(figsize=(12, 10))

for i, B in enumerate(B_values, 1):
    # Run Metropolis algorithm
    samples = metropolis(H, x0, steps, B, delta)
    
    # Compute expectation value, standard deviation, and standard error of H(x)
    H_values = np.array([H(x) for x in samples])
    expectation_H = np.mean(H_values)
    std_H = np.std(H_values, ddof=1)
    std_error_H = std_H / np.sqrt(len(H_values))
    
    # Compute running average to show thermalization
    running_avg = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    
    # Plot histogram of sampled values
    plt.subplot(4, 2, 2 * (i - 1) + 1)
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', label=f'B={B} Samples')
    
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
    plt.title(f'Metropolis Sampling for H(x) = x^2, B={B}')
    
    # Plot running average to show thermalization
    plt.subplot(4, 2, 2 * (i - 1) + 2)
    plt.plot(running_avg, label='Running Average', color='b')
    plt.axhline(expectation_H, color='r', linestyle='--', label='Final Expectation Value')
    plt.xlabel('Steps')
    plt.ylabel('Running Average of x')
    plt.legend()
    plt.title(f'Thermalization for B={B}')

plt.tight_layout()
plt.show()
