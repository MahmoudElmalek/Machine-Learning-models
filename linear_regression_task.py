import numpy as np
import matplotlib.pyplot as plt
# Given data
x = np.array([5, 10, 20, 30])
y = np.array([10, 15, 25, 35])
theta = np.array([1, 1]) # initial parameters
alpha = 0.005 # learning rate
# Print initial parameters
print("Linear Regression algorithm Q1")
print(f"Initial parameter θ = {theta}")
print(f"Learning rate α = {alpha}")
print("inputs x =")
print(x)
print("Target output y =")
print(y)
print("\n"*2)
# Perform gradient descent
num_iterations = 10
m = len(y)
for iteration in range(num_iterations):
    h = theta[0] + theta[1] * x
    error = h - y
 
    # Print iteration header
    print(f"Iteration ({iteration+1})")
    print("="*14)
    print("x\t\t y\t\t hθ(x)=θ*x\t ek=h-y\t       dJ/dθ0=ek*1\t dJ/dθ1=ek*x\t\t Cost")
    print("="*120)
 
    # Initialize gradients and cost
    dJ_dθ0_sum = 0
    dJ_dθ1_sum = 0
    cost_sum = 0
 
 # Compute gradients and accumulate cost
    for i in range(m):
        dJ_dθ0 = error[i]
        dJ_dθ1 = error[i] * x[i]
        cost = (error[i] ** 2)
 
        dJ_dθ0_sum += dJ_dθ0
        dJ_dθ1_sum += dJ_dθ1
        cost_sum += cost
 
        print(f"{x[i]}\t \t {y[i]}\t \t {h[i]:.3f}\t \t {error[i]:.3f}\t \t {dJ_dθ0:.3f}\t \t{dJ_dθ1:.3f} \t\t{cost:.3f}")
 
    # Update parameters
    new_theta0 = theta[0] - alpha * dJ_dθ0_sum / m
    new_theta1 = theta[1] - alpha * dJ_dθ1_sum / m
    theta = np.array([new_theta0, new_theta1])
 
        # Print total and updated parameters
    print("-" * 120)

    print(f"Total\t\t\t\t\t\t\t\t{dJ_dθ0_sum:.2f}\t\t{dJ_dθ1_sum:.2f}\t\t\t{cost_sum:.3f}\n")
    print(f"Cost\t= 1/(2*m)*∑(hθ(x)-y)^2= {(cost_sum / (2 * m)):.4f}")
    print(f"dθ0\t= -α/m*∑(dj/dθ0) = {(dJ_dθ0_sum * alpha / m):.6f}")
    print(f"dθ1\t= -α/m*∑(dj/dθ1) = {(dJ_dθ1_sum * alpha / m):.6f}")
    print(f"θ0\t= θ0 + dθ0 = {theta[0]:.4f}")
    print(f"θ1\t= θ1 + dθ1 = {theta[1]:.4f}")
    print("-" * 120)
    print("\n"*2)
plt.style.use('ggplot')
plt.plot(x,y,label='true value',marker='*')
plt.plot(x,h,label='predicted value',marker='^')
plt.title('Linear Regression Algorithm')
plt.xlabel('Input')
plt.ylabel('Output') 
plt.legend()
plt.show()
    


