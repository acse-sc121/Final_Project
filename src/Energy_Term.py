import abc
import numpy as np


mu0 = 4 * np.pi * 1e-7
dx, dy, dz = (2.5e-9, 2.5e-9, 2.5e-9)  # cell distance
Nx, Ny, Nz = (60, 60, 12)  # shape of magnetization


# The energy density is the energy per unit volume. The energy is the
# energy density multiplied by the volume
class EnergyTerm(abc.ABC):
    @abc.abstractmethod
    # It specifies that each energy term should
    # contain the effective field term
    def effective_field(self, m):
        pass

    def energy_density(self, m):
        '''
        The function is the uniform formula to calculate energy density.

        Parameters
        ----------
        m : numpy.ndarray
            the magnetization field class of the system

        Returns
        -------
        numpy.ndarray
            the energy density of the system

        '''
        value1 = np.multiply(m.get_m(), self.effective_field(m))
        value2 = np.multiply(m.get_Ms(), value1)
        return -mu0 / 2 * np.sum(value2, axis=3, keepdims=True)

    def energy(self, m):
        '''
        The function is the uniform formula to calculate energy.

        Parameters
        ----------
        m : numpy.ndarray
            the magnetization field class of the system

        Returns
        -------
        float
            the energy of the system

        '''
        dV = dx * dy * dz
        return np.sum(self.energy_density(m) * dV)


# It calculates the effective field due to exchange interactions
class Exchange(EnergyTerm):
    def __init__(self, A):
        self.A = A

    def effective_field(self, m):
        '''
        The function calcualtes the effective field of exchange energy term,
        it uses the Laplace function of the magnetization field.

        Parameters
        ----------
        m : numpy.ndarray
            the magnetization field class of the system

        Returns
        -------
        numpy.ndarray
            effective field of exchange energy

        '''
        if (np.allclose(m.get_M(), 0)):
            # zero magnetization has zero effective field
            return np.zeros_like(m.get_M())
        else:
            return np.multiply((2 * self.A) / (mu0 * m.get_Ms()), m.laplace())


class DMI(EnergyTerm):
    def __init__(self, D):
        self.D = D

    def effective_field(self, m):
        '''
        The function calcualtes the effective field of DMI energy term,
        it uses the Curl function of the magnetization field.

        Parameters
        ----------
        m : numpy.ndarray
            the magnetization field class of the system

        Returns
        -------
        numpy.ndarray
            effective field of DMI energy

        '''
        if (np.allclose(m.get_M(), 0)):
            # zero magnetization has zero effective field
            return np.zeros_like(m.get_M())
        else:
            return np.multiply(-((2 * self.D) / (mu0 * m.get_Ms())), m.curl())


class Zeeman(EnergyTerm):
    def __init__(self, H):
        self.H = H

    def effective_field(self, m):
        '''
        The function calcualtes the effective field of zeeman energy term,
        it equals to the external magnetic field.

        Parameters
        ----------
        m : numpy.ndarray
            the magnetization field class of the system

        Returns
        -------
        numpy.ndarray
            effective field of zeeman energy

        '''
        # The shape of the effective field should be same as
        # magnetization field.
        return np.tile(self.H, Nx * Ny * Nz).reshape((Nx, Ny, Nz, 3))

    def energy_density(self, m):
        '''
        The function calcualtes the energy density of zeeman energy term.

        Parameters
        ----------
        m : numpy.ndarray
            the magnetization field class of the system

        Returns
        -------
        numpy.ndarray
            effective field of zeeman energy

        '''
        value = np.multiply(m.get_m(), self.effective_field(m))
        return -mu0 * m.get_Ms() * np.sum(value, axis=3, keepdims=True)


class M:
    def __init__(self, M):
        self.M = M

    def get_M(self):
        '''
        This function returns the magnetization field.

        Returns
        -------
        numpy.ndarray
            capital M magnetization field

        '''
        return self.M

    def get_Ms(self):
        '''
        This function returns the saturation magnetization.

        Returns
        -------
        numpy.ndarray
            saturation magnetization of magnetization field

        '''
        # saturation magnetization is the norm of M
        # The shape is (Nx, Ny, Nz, 1)
        Ms = np.expand_dims(np.linalg.norm(self.M, axis=3), axis=3)
        return Ms

    def get_m(self):
        '''
        This function returns the unit magnetization vector field.

        Returns
        -------
        numpy.ndarray
            lowercase m of magnetization field

        '''
        if (np.allclose(self.get_M(), 0)):
            # zero magnetization has zero m
            return np.zeros_like(self.get_M())
        else:
            return self.M / self.get_Ms()

    def set_M(self, M_new):
        '''
        The function sets the value of the M.

        Parameters
        ----------
        M_new
            the new value of M

        '''
        self.M = M_new

    def curl(self):
        '''
        The curl of the magnetization vector field is calculated
        by taking the difference between the partial derivatives of
        the magnetization vector with respect to the x, y, and z directions.

        Dirichlet boundary condition has been used.

        This function has been used for calculate
        the effective field of DMI energy term.

        Returns
        -------
            The curl of the magnetization vector field.

        '''
        curl = np.zeros_like(self.get_m())

        dzdy = np.zeros((Nx, Ny, Nz))
        dydz = np.zeros((Nx, Ny, Nz))
        dxdz = np.zeros((Nx, Ny, Nz))
        dzdx = np.zeros((Nx, Ny, Nz))
        dxdy = np.zeros((Nx, Ny, Nz))
        dydx = np.zeros((Nx, Ny, Nz))

        # Dirichlet boundary condition
        # which means ghost items are zero.
        m_pad = np.pad(
            self.get_m(), ((1, 1), (1, 1), (1, 1), (0, 0)), 'constant',
            constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

        # First-order central difference has been used
        dzdy[...] += (m_pad[1:-1, 2:, 1:-1, 2] -
                      m_pad[1:-1, :-2, 1:-1, 2]) / (2 * dy)

        dydz[...] += (m_pad[1:-1, 1:-1, 2:, 1] -
                      m_pad[1:-1, 1:-1, :-2, 1]) / (2 * dz)

        dxdz[...] += (m_pad[1:-1, 1:-1, 2:, 0] -
                      m_pad[1:-1, 1:-1, :-2, 0]) / (2 * dz)

        dzdx[...] += (m_pad[2:, 1:-1, 1:-1, 2] -
                      m_pad[:-2, 1:-1, 1:-1, 2]) / (2 * dx)

        dxdy[...] += (m_pad[1:-1, 2:, 1:-1, 0] -
                      m_pad[1:-1, :-2, 1:-1, 0]) / (2 * dy)
        dydx[...] += (m_pad[2:, 1:-1, 1:-1, 1] -
                      m_pad[:-2, 1:-1, 1:-1, 1]) / (2 * dx)

        # Set the x, y, z term of curl using curl's formula
        curl[..., 0] += dzdy - dydz
        curl[..., 1] += dxdz - dzdx
        curl[..., 2] += dydx - dxdy
        return curl

    def laplace(self):
        '''
        The function Laplacian of the magnetisation field is calculated.

        Neumann boundary condition has been used.

        This function has been used for calculate
        the effective field of exchange energy term.

        Returns
        -------
                The Laplacian of the magnetisation field.

        '''
        lap = np.zeros_like(self.get_m())
        # Neumann boundary condition: dm/dn = 0
        # which means ghost items are equal to the boundary terms.
        m_pad = np.pad(self.get_m(), ((1, 1), (1, 1), (1, 1), (0, 0)), 'edge')
        # Second-Order Central Difference has been used.
        # Vectorisation
        lap += (m_pad[0:-2, 1:-1, 1:-1] + m_pad[2:, 1:-1, 1:-1] -
                2 * m_pad[1:-1, 1:-1, 1:-1]) / (dx ** 2)
        lap += (m_pad[1:-1, 0:-2, 1:-1] + m_pad[1:-1, 2:, 1:-1] -
                2 * m_pad[1:-1, 1:-1, 1:-1]) / (dy ** 2)
        lap += (m_pad[1:-1, 1:-1, 0:-2] + m_pad[1:-1, 1:-1, 2:] -
                2 * m_pad[1:-1, 1:-1, 1:-1]) / (dz ** 2)
        return lap
