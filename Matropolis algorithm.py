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

def detect_equilibrium(samples, window=500):
    running_avg = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    
    for i in range(window, len(samples)):
        recent_avg = np.mean(running_avg[i-window:i])
        previous_avg = np.mean(running_avg[i-2*window:i-window])
        if np.abs(recent_avg - previous_avg) < 0.01 * np.abs(previous_avg):
            return i
    
    return 0

# Define energy function H(x) = x^2
def H(x):
    return x**2

# Parameters
x0 = 1.0         # Initial state
steps = 10000    # Number of iterations
delta = 1.0      # Step size
B_values = [1, 2, 3, 4]  # Different inverse temperature parameters
bin_size = 10    # Number of samples per bin

plt.figure(figsize=(15, 20))

for i, B in enumerate(B_values, 1):
    samples = metropolis(H, x0, steps, B, delta)
    equilibrium_step = detect_equilibrium(samples)
    samples_eq = samples[equilibrium_step:]
    
    H_values = H(samples_eq)
    expectation_H = np.mean(H_values)
    std_H = np.std(H_values, ddof=1)
    std_error_H = std_H / np.sqrt(len(H_values))
    
    running_avg = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    
    # Ensure samples_eq length is divisible by bin_size
    trimmed_length = len(samples_eq) - (len(samples_eq) % bin_size)
    binned_samples = samples_eq[:trimmed_length].reshape(-1, bin_size).mean(axis=1)
    
    expectation_H_binned = np.mean(H(binned_samples))
    std_H_binned = np.std(H(binned_samples), ddof=1)
    std_error_H_binned = std_H_binned / np.sqrt(len(binned_samples))
    
    plt.subplot(4, 4, 4 * (i - 1) + 1)
    plt.hist(samples_eq, bins=50, density=True, alpha=0.6, color='b', label=f'B={B} Samples')
    
    x = np.linspace(-3, 3, 1000)
    p_x = np.exp(-H(x) * B)
    p_x /= np.trapz(p_x, x)
    plt.plot(x, p_x, 'r-', label='Theoretical Distribution')
    
    plt.axvline(np.sqrt(expectation_H), color='g', linestyle='--', linewidth=2, 
                label=f'⟨H(x)⟩ = {expectation_H:.3f} ± {std_error_H:.3f}')
    
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title(f'Metropolis Sampling for H(x) = x^2, B={B}')
    
    plt.subplot(4, 4, 4 * (i - 1) + 2)
    plt.plot(running_avg, label='Running Average', color='b')
    plt.axvline(equilibrium_step, color='r', linestyle='--', label='Equilibrium Point')
    plt.axhline(np.mean(running_avg[equilibrium_step:]), color='g', linestyle='--', label='Final Expectation')
    plt.xlabel('Steps')
    plt.ylabel('Running Average of x')
    plt.legend()
    plt.title(f'Thermalization for B={B}')
    
    plt.subplot(4, 4, 4 * (i - 1) + 3)
    plt.plot(samples_eq, label='Non-Binned Samples', color='g', alpha=0.5, linewidth=1)
    plt.plot(np.arange(0, len(binned_samples) * bin_size, bin_size), binned_samples, label='Binned Samples', color='m', linewidth=2)
    plt.axhline(np.mean(binned_samples), color='r', linestyle='--', label=f'Binned ⟨H(x)⟩={expectation_H_binned:.3f} ± {std_error_H_binned:.3f}')
    plt.axhline(np.mean(H_values), color='b', linestyle='--', label=f'{expectation_H} ± {std_error_H:.3f}')
    plt.xlabel('Steps')
    plt.ylabel('Sample Mean')
    plt.legend()
    plt.title(f'Binned vs. Non-Binned Comparison for B={B}')
    
plt.tight_layout()
plt.show()


