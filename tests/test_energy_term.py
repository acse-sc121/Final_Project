import sys
sys.path.append('.')
import numpy as np  # noqa: E402
import oommfc as oc  # noqa: E402
import discretisedfield as df  # noqa: E402
import micromagneticmodel as mm  # noqa: E402
from src.Energy_Term import Exchange, Zeeman, DMI, M  # noqa: E402


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
m = df.Field(mesh, dim=3, value=np.random.random((*n, 3)) * 2 - 1, norm=Ms)
system = mm.System(name='test_mf')
system.m = m
system.energy = (mm.Exchange(A=8.78e-12) +
                 mm.DMI(D=D, crystalclass='T') +
                 mm.Zeeman(H=(0, 0, B / mm.consts.mu0)))
m_my = M(system.m.array)
ex = Exchange(A=A)
z = Zeeman(H=H)
d = DMI(D=D)


class TestExchange:
    """
        Test cases for exchange term.
        Compare the results with ubermag.
    """

    def test_eff_ex(self):
        '''
            Test effective field
        '''
        H_u_eff_ex = oc.compute(system.energy.exchange.effective_field, system)
        H_my_eff_ex = ex.effective_field(m_my)
        assert np.allclose(H_u_eff_ex.array, H_my_eff_ex)

    def test_w_ex(self):
        '''
            Test energy density
        '''
        w_u_ex = oc.compute(system.energy.exchange.density, system)
        w_my_ex = ex.energy_density(m_my)
        assert np.allclose(w_u_ex.array, w_my_ex)

    def test_e_ex(self):
        '''
            Test energy
        '''
        e_u_ex = oc.compute(system.energy.exchange.energy, system)
        e_my_ex = ex.energy(m_my)
        assert np.allclose(e_u_ex, e_my_ex)


class TestDMI:
    """
        Test cases for DMI term.
        Compare the results with ubermag.
    """

    def test_eff_d(self):
        '''
            Test effective field
        '''
        H_u_eff_d = oc.compute(system.energy.dmi.effective_field, system)
        H_my_eff_d = d.effective_field(m_my)
        assert np.allclose(H_u_eff_d.array, H_my_eff_d)

    def test_w_d(self):
        '''
            Test energy density
        '''
        w_u_d = oc.compute(system.energy.dmi.density, system)
        w_my_d = d.energy_density(m_my)
        assert np.allclose(w_u_d.array, w_my_d)

    def test_e_d(self):
        '''
            Test energy
        '''
        e_u_d = oc.compute(system.energy.dmi.energy, system)
        e_my_d = d.energy(m_my)
        assert np.allclose(e_u_d, e_my_d)


class TestZeeman:
    """
        Test cases for zeeman term.
        Compare the results with ubermag.
    """

    def test_eff_z(self):
        '''
            Test effective field
        '''
        H_u_eff_z = oc.compute(system.energy.zeeman.effective_field, system)
        H_my_eff_z = z.effective_field(m_my)
        assert np.allclose(H_u_eff_z.array, H_my_eff_z)

    def test_w_z(self):
        '''
            Test energy density
        '''
        w_u_z = oc.compute(system.energy.zeeman.density, system)
        w_my_z = z.energy_density(m_my)
        assert np.allclose(w_u_z.array, w_my_z)

    def test_e_z(self):
        '''
            Test energy
        '''
        e_u_z = oc.compute(system.energy.zeeman.energy, system)
        e_my_z = z.energy(m_my)
        assert np.allclose(e_u_z, e_my_z)
