import numpy as np
import matplotlib.pyplot as plt

def metropolis(H, x0, steps, B, delta):
    """Metropolis algorithm for sampling from a given energy function H(x)."""
    x = x0
    samples = []
    
    for _ in range(steps):
        x_new = x + np.random.uniform(-delta, delta)
        dH = H(x_new) - H(x)
        
        if dH < 0 or np.random.uniform(0,1) < np.exp(-dH * B):
            x = x_new  
        
        samples.append(x)
    
    return np.array(samples)

def autocorrelation(samples, max_tau):
    """Compute the autocorrelation function C(tau)."""
    mean = np.mean(samples)
    var = np.var(samples)
    C_tau = np.array([np.mean((samples[:-tau] - mean) * (samples[tau:] - mean)) / var for tau in range(1, max_tau)])
    return C_tau

def integrated_autocorrelation(C_tau):
    """Compute the integrated autocorrelation time tau_int."""
    return 1 + 2 * np.sum(C_tau)

# Define energy function H(x) = x^2
def H(x):
    return x**2

# Parameters
x0 = 0.0         # Initial state
steps = 5000     # Number of iterations
delta = 1.0      # Step size
B = 1            # Inverse temperature
bin_size = 10    # Number of samples per bin
max_tau = 100    # Maximum tau for autocorrelation

# Run Metropolis algorithm
samples = metropolis(H, x0, steps, B, delta)

# Compute autocorrelation function
C_tau = autocorrelation(samples, max_tau)

# Compute integrated autocorrelation time
tau_int = integrated_autocorrelation(C_tau)

# Binning procedure
binned_samples = samples[:steps//bin_size * bin_size].reshape(-1, bin_size).mean(axis=1)

# Compute expectation values and errors
H_values = np.array([H(x) for x in samples])
H_values_binned = np.array([H(x) for x in binned_samples])

expectation_H = np.mean(H_values)
std_error_H = np.std(H_values, ddof=1) / np.sqrt(len(H_values))

expectation_H_binned = np.mean(H_values_binned)
std_error_H_binned = np.std(H_values_binned, ddof=1) / np.sqrt(len(H_values_binned))

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Autocorrelation function
axes[0].plot(range(1, max_tau), C_tau, marker='o', linestyle='-', color='b', label='C(τ)')
axes[0].set_xlabel('τ')
axes[0].set_ylabel('Autocorrelation C(τ)')
axes[0].set_title(f'Autocorrelation Function (τ_int ≈ {tau_int:.2f})')
axes[0].legend()

# Binning comparison
axes[1].plot(samples, label='Non-Binned Samples', color='g', alpha=0.5, linewidth=1)
axes[1].plot(np.arange(0, len(samples), bin_size), binned_samples, label='Binned Samples', color='m', linewidth=2)
axes[1].axhline(expectation_H_binned, color='r', linestyle='--', label=f'Binned ⟨H(x)⟩={expectation_H_binned:.3f} ± {std_error_H_binned:.3f}')
axes[1].axhline(expectation_H, color='b', linestyle='--', label=f'{expectation_H:.3f} ± {std_error_H:.3f}')
axes[1].set_xlabel('Steps')
axes[1].set_ylabel('Sample Mean')
axes[1].set_title('Binned vs. Non-Binned Comparison')
axes[1].legend()

# Binned error analysis
bin_sizes = np.arange(1, 100, 5)
errors = [np.std(samples[:steps//b] * b, ddof=1) / np.sqrt(len(samples[:steps//b] // b)) for b in bin_sizes]

axes[2].plot(bin_sizes, errors, marker='o', linestyle='-', color='r', label='Estimated Error')
axes[2].set_xlabel('Bin Size τ')
axes[2].set_ylabel('Estimated Error')
axes[2].set_title('Error vs. Bin Size')
axes[2].legend()

plt.tight_layout()
plt.show()
