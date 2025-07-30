import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81
l = 1.0
m = 1.0
T_char = np.sqrt(l / g)
E_sep = 2 * m * g * l 


def f1(t, y):
    theta, omega = y
    return [omega, -(g / l) * np.sin(theta)]

def solv(theta0, omega0, t_max=50.0, n_p=10000):
    y0 = [theta0, omega0]
    t_eval = np.linspace(0, t_max, n_p)
    sol = solve_ivp(f1, [0, t_max], y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-8)
    return sol.t, sol.y[0], sol.y[1]

def energy(theta, omega):
    return 0.5 * m * (l * omega)**2 + m * g * l * (1 - np.cos(theta))

def Period(t, theta, omega, E):
    theta_norm = np.mod(theta + np.pi, 2 * np.pi) - np.pi

    crossings = []
    for i in range(1, len(theta_norm)):
        if theta_norm[i - 1] < 0 and theta_norm[i] >= 0 and omega[i] > 0:
            dt = t[i] - t[i - 1]
            dtheta = theta_norm[i] - theta_norm[i - 1]
            t_cross = t[i - 1] - theta_norm[i - 1] * (dt / dtheta)
            crossings.append(t_cross)

    if len(crossings) >= 2:
        periods = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
        if periods:
            return np.median(periods)

    return None

def T1(E): 
    return 2 * np.pi * np.sqrt(l / g) * (1 + E / (8 * m * g * l))

def T2(E): 
    return 2 * np.sqrt(l / g) * np.log(32 * m * g * l / (2 * m * g * l - E))

def T3(E): 
    return np.sqrt(l / g) * np.log(32 * m * g * l / (E - 2 * m * g * l))

def T4(E):  
    return np.pi * np.sqrt(2 * m * l**2 / E)


target_omega = 2 * np.sqrt(g/l)

init_cond = [
    (0.4, 0.4),
    (1.5, 0.5),
    (0.0, 1.0),
    (0.1, 2.0),
    (0.0, target_omega), 
    (0.0, 4.0),
    (0.0, 6.0),
    (0.1, 8.0),
    (0.0, 12.0),
]


plt.figure(figsize=(10, 6))
for theta0, omega0 in init_cond:
    t_arr, theta_arr, omega_arr = solv(theta0, omega0, t_max=20, n_p=5000)
    if theta0 == 0.0 and abs(omega0 - target_omega) < 0.01:
        plt.plot(theta_arr, omega_arr, 'r-', linewidth=2, label=f'φ₀=0, ω₀=2√(g/l)≈{omega0:.2f}')
    else:
        plt.plot(theta_arr, omega_arr, label=f'θ₀={theta0:.1f}, ω₀={omega0:.1f}')

plt.grid(True)
plt.xlabel('θ (рад)', fontsize=14)
plt.ylabel('ω (рад/с)', fontsize=14)
plt.title('Фазовый портрет маятника', fontsize=16)
plt.legend()
plt.xlim(-2 * np.pi, 2 * np.pi)
plt.tight_layout()
plt.show()

E_val = np.linspace(0.01, 10, 500) * (m * g * l)  
T_val = np.zeros_like(E_val)

for i, E in enumerate(E_val):
    if E < 0.5 * E_sep:
        T_val[i] = T1(E)  
    elif E < 0.99 * E_sep:
        T_val[i] = T2(E)  
    elif E > 1.01 * E_sep and E < 2 * E_sep:
        T_val[i] = T3(E)  
    else:
        T_val[i] = T4(E)  

t_max = 50  
n_p = 1000  

omega0_list = np.concatenate([
    np.linspace(0.1, 1.5, 8) * np.sqrt(g / l),  
    np.linspace(1.5, 1.95, 5) * np.sqrt(g / l), 
    np.linspace(2.05, 2.5, 5) * np.sqrt(g / l),  
    np.linspace(2.5, 4.0, 6) * np.sqrt(g / l),  
    np.linspace(4.0, 10.0, 8) * np.sqrt(g / l)  
])


omega0_list = np.append(omega0_list, target_omega)
omega0_list = np.sort(omega0_list)

ch_E = []
ch_T = []

for omega0 in omega0_list:
    theta0 = 0.0
    E = energy(theta0, omega0)

    t_arr, theta_arr, omega_arr = solv(theta0, omega0, t_max=t_max, n_p=n_p)
    period = Period(t_arr, theta_arr, omega_arr, E)

    ch_E.append(E)
    ch_T.append(period)

ch_E = np.array(ch_E)
ch_T = np.array(ch_T)

plt.figure(figsize=(12, 7))
plt.plot(E_val[E_val < 0.5*E_sep]/(m*g*l), T_val[E_val < 0.5*E_sep], 'blue', label='Малые колебания')
plt.plot(E_val[(E_val >= 0.5*E_sep) & (E_val < 0.99*E_sep)]/(m*g*l),
         T_val[(E_val >= 0.5*E_sep) & (E_val < 0.99*E_sep)], 'green',
         label='Колебания вблизи сепаратрисы')
plt.plot(E_val[(E_val > 1.01*E_sep) & (E_val < 2*E_sep)]/(m*g*l),
         T_val[(E_val > 1.01*E_sep) & (E_val < 2*E_sep)], 'purple',
         label='Перевороты вблизи сепаратрисы')
plt.plot(E_val[E_val >= 2*E_sep]/(m*g*l),
         T_val[E_val >= 2*E_sep], 'brown', label='Быстрое вращение')

plt.scatter(ch_E / (m * g * l), ch_T, color='black', s=50, zorder=5, label='Численное решение')


target_E = energy(0.0, target_omega)
target_index = np.where(np.isclose(ch_E, target_E))[0]
if len(target_index) > 0:
    target_period = ch_T[target_index[0]]
    plt.scatter(target_E / (m * g * l), target_period, color='red', s=100, zorder=6, 
                label=f'φ₀=0, ω₀=2√(g/l)')

plt.axvline(x=2.0, color='red', linestyle='--', alpha=0.5, label='Сепаратриса (E = 2mgl)')

plt.grid(True)
plt.xlabel('Энергия E/(mgl)', fontsize=14)
plt.ylabel('Период T (с)', fontsize=14)
plt.title('Зависимость периода от энергии T(E)', fontsize=16)
plt.xlim(0, 10)
plt.ylim(0, 8)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.show()
