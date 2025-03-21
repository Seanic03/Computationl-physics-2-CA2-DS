#student number 2115701


import numpy as np
import matplotlib.pyplot as plt

# Defining parameters
x_0 = 0.0  # Initial position
N = 1000  # Number of steps
delta = 1.0  #only set to 1 to increase step size
beta = [1, 2, 3, 4]  # List of values beta can be for example where beta =1/k*T shows inverse temperature relation

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


plt.figure(figsize=(14, 12)) #size of plots in x and y direction
plt.suptitle("Metropolis program for different values of beta ", fontsize=16) #main title

for i, B in enumerate(beta, 1):
    
    answers = metropolis(H, x_0, N, B, delta)
    
    
    x = np.linspace(-5, 5, N) #giving x an arbitary range over the number of steps taken
    p_x = np.exp(-H(x) *B)  # Partition function proportionality
    norm = np.trapz(p_x, x)  # Normalize the distribution
    result = p_x / norm  # Normalized distribution
    
    
    plt.subplot(4, 3, 3 * (i - 1) + 1)
    plt.hist(answers, bins=50, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, result, 'r-', label="Distribution", linewidth=1.5)
    plt.title(f"Histogram for β = {B}", fontsize=12)
    plt.xlabel("Values", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend(fontsize=8)
    plt.grid()
    
    running_avg = np.cumsum(answers) / np.arange(1, len(answers) + 1)# cumulative sum over all the steps program is run over
    plt.subplot(4, 3, 3 * (i - 1) + 2)
    plt.plot(running_avg, color='blue', label='Running Average', linewidth=1.5)
    plt.axhline(np.mean(running_avg), color='red', linestyle='--', label='Mean', linewidth=1.5)
    plt.title(f"weighted data from histogram for β = {B}", fontsize=12)
    plt.xlabel("Steps", fontsize=10)
    plt.ylabel("Average", fontsize=10)
    plt.legend(fontsize=8)
    plt.grid()
    
    plt.subplot(4, 3, 3 * (i - 1) + 3)
    plt.plot(answers, 'b-', label='Data from answers ', linewidth=0.7)
    plt.title(f"data for β = {B}", fontsize=12)
    plt.xlabel("Step", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    plt.legend(fontsize=8)
    plt.grid()

plt.tight_layout()  
plt.savefig("metropolis for different values of beta.pdf")
plt.show()