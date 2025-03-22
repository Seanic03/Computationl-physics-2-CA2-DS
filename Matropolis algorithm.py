import numpy as np
import matplotlib.pyplot as plt

def metropolis(H, x0, steps, B, delta):
    x = x0
    samples = []
    
    for _ in range(steps):
        x_new = x + np.random.uniform(-delta, delta)
        dH = H(x_new) - H(x)
        
        if dH < 0 or np.random.uniform(0,1) < np.exp(-dH * B):
            x = x_new  
        
        samples.append(x)
    
    return samples

# Define energy function H(x) = x^2
def H(x):
    return x**2

# Parameters
x0 = 0.0         # Initial state
steps = 1000     # Number of iterations
delta = 1.0      # Step size
B_values = [1, 2, 3, 4]  # Different inverse temperature parameters
bin_size = 10    # Number of samples per bin

plt.figure(figsize=(15, 20))

for i, B in enumerate(B_values, 1):
    # Run Metropolis algorithm
    samples = metropolis(H, x0, steps, B, delta)
    
    # Compute expectation value, standard deviation, and standard error for non-binned samples
    H_values = np.array([H(x) for x in samples])
    expectation_H = np.mean(H_values)
    std_H = np.std(H_values, ddof=1)
    std_error_H = std_H / np.sqrt(len(H_values))
    
    # Compute running average to show thermalization
    running_avg = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    
    # Bin the samples
    binned_samples = np.array(samples).reshape(-1, bin_size).mean(axis=1)

    # Compute expectation value, standard deviation, and standard error for binned samples
    expectation_H_binned = np.mean([H(x) for x in binned_samples])
    std_H_binned = np.std([H(x) for x in binned_samples], ddof=1)
    std_error_H_binned = std_H_binned / np.sqrt(len(binned_samples))
    
    # Plot histogram of sampled values
    plt.subplot(4, 4, 4 * (i - 1) + 1)
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
    plt.subplot(4, 4, 4 * (i - 1) + 2)
    plt.plot(running_avg, label='Running Average', color='b')
    plt.axhline(np.mean(running_avg), color='r', linestyle='--', label='Final Expectation Value')
    plt.xlabel('Steps')
    plt.ylabel('Running Average of x')
    plt.legend()
    plt.title(f'Thermalization for B={B}')
    
    # Plot binned samples
    plt.subplot(4, 4, 4 * (i - 1) + 3)
    plt.plot(binned_samples, label='Binned Samples', color='m', linewidth=1)
    plt.axhline(np.mean(binned_samples), color='r', linestyle='--', label=f'⟨H(x)⟩ = {expectation_H_binned:.3f} ± {std_error_H_binned:.3f}')
    plt.xlabel('Binned Steps')
    plt.ylabel('Binned Sample Mean')
    plt.legend()
    plt.title(f'Binned Samples for B={B}')
    
    # Plot comparison between binned and non-binned samples
    plt.subplot(4, 4, 4 * (i - 1) + 4)
    plt.plot(samples, label='Non-Binned Samples', color='g', alpha=0.5, linewidth=1)
    plt.plot(np.arange(0, len(samples), bin_size), binned_samples, label='Binned Samples', color='m', linewidth=2)
    plt.axhline(np.mean(binned_samples), color='r', linestyle='--', label=f'Binned ⟨H(x)⟩ ± {std_error_H_binned:.3f}')
    plt.axhline(np.mean(H_values), color='b', linestyle='--', label=f'Non-Binned ⟨H(x)⟩ ± {std_error_H:.3f}')
    plt.xlabel('Steps')
    plt.ylabel('Sample Mean')
    plt.legend()
    plt.title(f'Binned vs. Non-Binned Comparison for B={B}')
    
plt.tight_layout()
plt.show()

