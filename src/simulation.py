import discretisedfield as df
import micromagneticmodel as mm
from Energy_Term import Exchange, Zeeman, DMI, M
from Mean_Field import Min_Driver
import numpy as np
import matplotlib.pyplot as plt
import time

# Some important hyperparameters
D = 1.58e-3  # DMI material parameter
A = 8.78e-12  # Exchange energy constant
B = 0.2  # External magnetic field strength
beta = 9e99  # 9e99 represents positive infinity
mu0 = 4 * np.pi * 1e-7
H = (0, 0, B / mu0)
dx, dy, dz = (2.5e-9, 2.5e-9, 2.5e-9)  # cell distance
Ms = 3.84e5  # saturation magnetisation
n = (60, 60, 12)  # shape of magnetization

# Initialize system
region = df.Region(p1=(0, 0, 0), p2=(150e-9, 150e-9, 30e-9))
mesh = df.Mesh(region=region, cell=(2.5e-9, 2.5e-9, 2.5e-9))
m = df.Field(mesh, dim=3, value=[0, 0, -1], norm=Ms)
system = mm.System(name='skyrmion')
system.m = m
m_my = M(system.m.array)

# Define Energy Terms
ex = Exchange(A=A)
dmi = DMI(D=D)
zeeman = Zeeman(H=H)

# Apply Mean-field model
E_ini = ex.energy(m_my) + zeeman.energy(m_my) + dmi.energy(m_my)
print("Initial Energy: ", E_ini)

time_start = time.time()
min_driver = Min_Driver()
final_M_my, E_end, count = min_driver.Mean_field_difference(m_my)
M_final = final_M_my.get_M()
print("Final state of M: ", M_final)
time_end = time.time()
print("Time cost: ", time_end - time_start, "s")

print("My Final Energy: ", E_end)

# Update the magnetization vector field
m = df.Field(mesh, dim=3, value=M_final)
m.plane('z').mpl()
plt.show()
