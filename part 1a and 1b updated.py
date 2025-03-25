


import numpy as np
import matplotlib.pyplot as plt

# Defining parameters
x_0 = 1 # Initial position
N = 10000  # Number of steps
delta = 1.0  #only set to 1 to increase step size
beta = [1, 2, 3, 4]  # List of values beta can be for example where beta =1/k*T shows inverse temperature relation
bin_size = 10

def H(x):
    return x**2  # stating that H(x)=x**2(hamiltonian function)


def metropolis(H, x0, steps, beta, delta): #defining metropolis function
    x = x0
    answers = []  # empty list to store values that will later be worked out 
    for _ in range(N): #starting part of the markov chain
        x_new = x + np.random.uniform(-delta, delta)  # Propose new value/state
        dH = H(x_new) - H(x)  # Change in Hamiltonian
        r = np.random.uniform(0, 1)  # Random number to accept between 0 and 1
        if dH < 0 or r < np.exp(-dH * B):  # Metropolis acceptance criteria
            x = x_new  # If line above is true which it will be then it accepts the new state that is defined 
        answers.append(x)  # appends the values to the empty list that answers equals
    return answers # just gives values stored in answers =[]

def detect_equilibrium(samples, window=1000):
    running_avg = np.cumsum(answers) / np.arange(1, len(answers) + 1)
    
    for i in range(window, len(answers)):
        recent_avg = np.mean(running_avg[i-window:i])
        previous_avg = np.mean(running_avg[i-2*window:i-window])
        if np.abs(recent_avg - previous_avg) < 0.01 * np.abs(previous_avg):
            return i
    
    return 0

plt.figure(figsize=(14, 12)) #size of plots in x and y direction
plt.suptitle("Metropolis program for different values of beta ", fontsize=16) #main title

for i, B in enumerate(beta, 1):
    
    answers = metropolis(H, x_0, N, B, delta)
    
    #for non binned samples working out standard error,standard deviation,expectation value
    H_values = np.array([H(x) for x in answers])
    expec_H = np.mean(H_values) # mean of the hamiltonian values
    std_H = np.std(H_values,ddof=1) # standard deviation of hamiltonian values
    uncertanity =std_H/np.sqrt(len(H_values)) #standard error in hamiltonian 
    
    
    x = np.linspace(-5, 5, N) #giving x an arbitary range over the number of steps taken
    p_x = np.exp(-H(x) *B)  # Partition function proportionality
    norm = np.trapz(p_x, x)  # Normalize the distribution
    result = p_x / norm  # Normalized distribution
    
    binned_samples = np.array(answers).reshape(-1, bin_size).mean(axis=1)

   # Compute expectation value, standard deviation, and standard error for binned samples
    expectation_H_binned = np.mean([H(x) for x in binned_samples])
    std_H_binned = np.std([H(x) for x in binned_samples], ddof=1)
    std_error_H_binned = std_H_binned / np.sqrt(len(binned_samples))
    
    plt.subplot(4, 3, 3 * (i - 1) + 1)
    plt.hist(answers, bins=50, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, result, 'r-', label="Distribution", linewidth=1.5)   
    plt.title(f"Histogram for β = {B}", fontsize=12)
    plt.xlabel("Values", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend(fontsize=8)
    plt.grid()
    
    running_avg = np.cumsum(answers) / np.arange(1, len(answers) + 1) # cumulative sum over all the steps program is run over
    plt.subplot(4, 3, 3 * (i - 1) + 2)
    plt.plot(running_avg, color='blue', label='Running Average', linewidth=1.5)
    plt.axhline(np.mean(running_avg), color='red', linestyle='--', label='Mean', linewidth=1.5)
    plt.title(f"Thermalization for β = {B}", fontsize=12)
    plt.xlabel("Steps", fontsize=10)
    plt.ylabel("Average", fontsize=10)
    plt.legend(fontsize=8)
    plt.grid()
    
    plt.subplot(4, 3, 3 * (i - 1) + 3)
    plt.plot(answers, label='Non-Binned Samples', color='b', alpha=0.5, linewidth=1)
    plt.plot(np.arange(0, len(answers), bin_size), binned_samples, label='Binned Samples', color='g', linewidth=2)
    plt.axhline(np.mean(binned_samples), color='r', linestyle='--', label=f'Binned ⟨H(x)⟩={expectation_H_binned:.3f} ± {std_error_H_binned:.3f}')
    plt.axhline(np.mean(H_values), color='b', linestyle='--', label=f'{expec_H} ± {std_H:.3f}')
    plt.xlabel('Steps')
    plt.ylabel('Sample Mean')
    plt.legend()
    plt.title(f'Binned vs. Non-Binned Comparison for β = {B}')
    

plt.tight_layout()  
plt.savefig("metropolis for different values of beta.pdf")
plt.show()