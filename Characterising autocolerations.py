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
    
    return np.array(samples)

def autocorrelation(samples, max_tau):
    mean = np.mean(samples)
    var = np.var(samples)
    C_tau = np.array([np.mean((samples[:len(samples)-tau] - mean) * (samples[tau:] - mean)) for tau in range(max_tau)])
    return C_tau / var

# Define energy function H(x) = x^2
def H(x):
    return x**2

# Parameters
x0 = 0.0         # Initial state
steps = 5000     # Number of iterations
delta = 1.0      # Step size
B = 1.0          # Fixed inverse temperature
bin_sizes = [1, 5, 10, 15, 20,25,30,35,40,45,50,50,60]  # Different bin sizes for comparison
max_tau = 50

# Run Metropolis algorithm
samples = metropolis(H, x0, steps, B, delta)
H_values = H(samples)

# Compute autocorrelation
C_tau = autocorrelation(H_values, max_tau)

tau_int = 1 + 2 * np.sum(C_tau)

# Compute binned statistics
errors = []
for bin_size in bin_sizes:
    binned_samples = samples[:len(samples)//bin_size * bin_size].reshape(-1, bin_size).mean(axis=1)
    std_H_binned = np.std(H(binned_samples), ddof=1)
    std_error_H_binned = std_H_binned / np.sqrt(len(binned_samples))
    errors.append(std_error_H_binned)

# Plot results
plt.figure(figsize=(12, 10))

# Plot autocorrelation function
plt.subplot(2, 2, 1)
plt.plot(C_tau, marker='o', linestyle='-', label='Autocorrelation')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$C(\tau)$')
plt.title('Autocorrelation Function')
plt.legend()

# Plot binned error estimate
plt.subplot(2, 2, 2)
plt.scatter(bin_sizes, errors, marker='o', label='Binned Standard Error')
plt.xlabel('Bin Size')
plt.ylabel('Standard Error')
plt.title('Standard Error vs Bin Size')
plt.legend()

# Plot raw samples
plt.subplot(2, 2, 3)
plt.plot(samples[:500], label='Raw Samples', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('x')
plt.title('Metropolis Sampled Values')
plt.legend()

# Histogram of sampled values
plt.subplot(2, 2, 4)
plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', label='Sampled Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Histogram of Sampled Values')
plt.legend()

plt.tight_layout()
plt.show()

print(f'Estimated Integrated Autocorrelation Time: {tau_int:.2f}')
