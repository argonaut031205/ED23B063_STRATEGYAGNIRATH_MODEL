import numpy as np
import random
import matplotlib.pyplot as plt

#I am assuming that the competition takes place on 30th june 2025,(just assuming a random date)

M = 300
a = []  # acceleration (del v / del t)
c1 = 0.004  # coefficient
g = 9.81  # m/s^2
d = 1.225  # kg/m^3
A = 2  # m^2 (frontal area)
efficiency = 0.65
# I = 1000  # Solar irradiance
A1 = 4  # Panel area


learning_rate = 0.01
num_iterations = 10000
tolerance = 1e-6

v_values = np.array([0, 2, 3, 3, 3, 4, 5, 5, 9, 15, 15, 16, 17, 17, 19, 12, 13, 14, 18, 24, 24, 24, 24, 24, 29, 27, 29, 33, 35, 39, 40, 42, 44, 47, 50, 33, 24, 21, 18, 22, 22, 22, 22, 26, 26, 29, 29, 29, 30, 30, 34, 35,40,48,52,55,55,55,55,60,58,63])

I = []
t = []
optimized_v = []

#the next few lines of code is used in finding the cos of angle of inclination
delta=23.20 #degrees (found out from declination angle calculator in google)
La= np.linspace(-18,-15,62) #latitude 
Lo= np.linspace(132,136,62) #longitude

for i in range(len(v_values)):
    I.append(random.randint(1300, 1400))

for i in range(len(v_values)):
    t.append(i)#here im just creating an array for time
    tsolar= 4*(136-Lo) + t[i]#calculating solar time, this is the formula.if this is not clear i have explained about this in previous qns
    if i == 0:
        a.append(v_values[i] / 1)
    else:
        delta_v = v_values[i] - v_values[i - 1]
        delta_t = t[i] - t[i - 1]
        if delta_t == 0:
            a.append(0)  # Avoid division by zero
        else:
            a.append(delta_v / delta_t)



omega= 15*(np.array(tsolar)-0)/60 #we use solar time to calculate omega
cos_theta=np.zeros_like(t)# Angle of inclination of sun assuming that the tilted angle of the solar panel is 0

cos_theta=((np.sin(delta)*np.sin(La))+np.cos(delta)*np.cos(La)*np.cos(omega))
for i in range(len(cos_theta)):
    if cos_theta[i]<0:
        cos_theta[i]=-cos_theta[i]#preventing negative vaues of costheta

   
I = np.array(I)
a = np.array(a)


def power_drained(v, a, I, costheta):#this the power drained from battery
    return (v * (M * a + c1 * M * g + 0.5 * A * d * (v) * (v)) / efficiency) - ((I) * A1 * costheta)



# Define the gradient of f(x, i) with respect to x
def grad(v, a):
    return (M * np.array(a) + c1 * M * g + 3.5 * A * d * v * v) / efficiency

#defining a function for the adam optimizer
def adam_optimizer(velocity, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=1000,
                    tolerance=1e-6):
    
    m = np.zeros_like(velocity)  # First moment estimate
    v = np.zeros_like(velocity)  # Second moment estimate
    t = 0  # Time step
    v_optimized = []  # Optimized v values

    for _ in range(num_iterations):
        t += 1
        gradient = grad(velocity, a)  # Gradient of power_drained
        
        m = beta1 * m + (1 - beta1 ) * gradient  # Update first moment estimate
        v = beta2 * v + (1 - beta2 ) * (gradient * gradient)  # Update second moment estimate

        m_hat = m / (1 - np.power(beta1,t) )  # Bias-corrected first moment estimate
        v_hat = v / (1 - np.power(beta2,t) )  # Bias-corrected second moment estimate
        
        velocity = velocity.astype(float)  # Convert to float to allow for floating-point arithmetic
        velocity -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # Update velocity

        if np.all(np.abs(gradient)) < tolerance:
            break
    v_optimized.append(velocity)
    return v_optimized


# Optimizing x using Adam optimizer
optimized_v_values = adam_optimizer(v_values)
energy_drain=[]
soc=[]
energy=0
#print(power_drained(v_values,a,I,theta))
p=power_drained(v_values,a,I,cos_theta)
for i in range(0,62):
    if i<60:
        energy_interval = 0.5 * (p[i] + p[i + 1]) * (t[i + 1] - t[i])
        #energy += energy_interval
        #energy_drain.append(power_drained(v_values,a,I,theta))
        energy_drain.append(energy_interval)
    if i==61:
        energy_interval = 0.5 * (p[i]) * (t[i])

# Initialize soc array with zeros
soc = np.zeros(len(t))

for i, j in enumerate(energy_drain):
    ratio = (j)/ 645000
    soc[i] = 100 - (ratio * 100)
    if soc[i]>100:
        soc[i]=100

print("soc:",soc,"\n")
# Print optimized x values
print("Optimized v values:", optimized_v_values,"\n")
print("energy drained:",energy_drain,"\n")
plt.plot(t,v_values, label='Initial v values')

# Plotting optimized x values
for i, v_optimized in enumerate(optimized_v_values):
    plt.plot(t, v_optimized, label=f'Optimized v values ')

plt.xlabel('time(secs)')
plt.ylabel('Value of v')
plt.title('Comparison of Initial and Optimized v values')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(t, soc, label='State of Charge (SOC)')
plt.xlabel('Time')
plt.ylabel('State of Charge (%)')
plt.title('State of Charge vs Time')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(t[:-2], energy_drain, label='Energy (milli J)')
plt.xlabel('Time')
plt.ylabel('State of Charge (%)')
plt.title('Energy drained/gievn from battery vs Time')
plt.legend()
plt.grid(True)
plt.show()

