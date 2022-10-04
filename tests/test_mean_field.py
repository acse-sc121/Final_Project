import sys
sys.path.append('.')
import numpy as np  # noqa: E402
import discretisedfield as df  # noqa: E402
import micromagneticmodel as mm  # noqa: E402
from src.Energy_Term import Exchange, Zeeman, DMI, M  # noqa: E402
from src.Mean_Field import Min_Driver  # noqa: E402


# Some important hyperparameters
Ms = 3.84e5  # saturation magnetisation
D = 1.58e-3  # DMI material parameter
B = 0.2  # External magnetic field strength
mu0 = 4 * np.pi * 1e-7
A = 8.78e-12  # Exchange energy constant
dx, dy, dz = (2.5e-9, 2.5e-9, 2.5e-9)  # cell distance
Nx, Ny, Nz = (60, 60, 12)  # shape of magnetization
H = (0, 0, B / mu0)
n = (60, 60, 12)  # shape of magnetization


# Initialize system
region = df.Region(p1=(0, 0, 0), p2=(150e-9, 150e-9, 30e-9))
mesh = df.Mesh(region=region, cell=(2.5e-9, 2.5e-9, 2.5e-9))
m = df.Field(mesh, dim=3, value=[0, 0, -1], norm=Ms)
system = mm.System(name='test_mf')
system.m = m
system.energy = (mm.Exchange(A=8.78e-12) +
                 mm.DMI(D=D, crystalclass='T') +
                 mm.Zeeman(H=(0, 0, B / mm.consts.mu0)))
m_my_1 = M(system.m.array)
m_my_2 = M(system.m.array)
ex = Exchange(A=A)
z = Zeeman(H=H)
d = DMI(D=D)


class TestBiStable:
    '''
        Test Bi-Stability of Mean-field model,
        which means the results are the same
        of two times of runnning.
    '''

    def test_from_iter(self):
        '''
        Compare the iteration to let the system converges.
        If they are totally same, mean-field algorithm is Bi-Stable.
        '''
        min_driver = Min_Driver()
        M_1, E_1, count_1 = min_driver.Mean_field_difference(m_my_1)
        M_2, E_2, count_2 = min_driver.Mean_field_difference(m_my_2)
        assert count_1 == count_2

    def test_from_finalM(self):
        '''
        Compare the final magnetisation of the lowest energy state.
        If they are totally same, mean-field algorithm is Bi-Stable.
        '''
        min_driver = Min_Driver()
        M_1, E_1, count_1 = min_driver.Mean_field_difference(m_my_1)
        M_2, E_2, count_2 = min_driver.Mean_field_difference(m_my_2)
        assert (M_1.get_M() == M_2.get_M()).all()

    def test_from_energy(self):
        '''
        Compare the final energy of the lowest energy state.
        If they are totally same, mean-field algorithm is Bi-Stable.
        '''
        min_driver = Min_Driver()
        M_1, E_1, count_1 = min_driver.Mean_field_difference(m_my_1)
        M_2, E_2, count_2 = min_driver.Mean_field_difference(m_my_2)
        assert E_1 == E_2
