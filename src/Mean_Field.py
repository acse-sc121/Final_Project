from src.Energy_Term import Exchange, Zeeman, DMI
import numpy as np

# Some important hyperparameters
beta = 9e99  # 9e99 represents positive infinity
D = 1.58e-3  # DMI material parameter
A = 8.78e-12  # Exchange energy constant
B = 0.2  # External magnetic field strength
mu0 = 4 * np.pi * 1e-7
H = (0, 0, B / mu0)
maxiter = 12000


class Min_Driver:
    def Langevin(self, x):
        """
        The function calculates the value of the Langevin function

        Parameters
        ----------
        x: float or numpy.ndarray
            the input to the Langevin function

        Returns
        -------
        float or numpy.ndarray
            the value of the Langevin function.
        >>> import numpy as np
        >>> min_driver = Min_Driver()
        >>> min_driver.Langevin(3)
        0.671636489980356

        >>> x = np.array([[1.00],[1.00],[1.00]])
        >>> min_driver.Langevin(x)
        [[0.313035],[0.313035],[0.313035]]
        """
        return (1 / np.tanh(x)) - x ** (-1)

    def cal_effective_field(self, m):
        '''
        The function calculates effective field and energy of the system

        Parameters
        ----------
        m : numpy.ndarray
            the magnetization field class of the system

        Returns
        -------
        H_eff : numpy.ndarray
            The effective field of m
        E : float
            The energy of m.

        '''
        # Define energy terms exchange, zeeman and dmi
        exchange = Exchange(A=A)
        zeeman = Zeeman(H=H)
        dmi = DMI(D=D)
        # Calculate the effective field of energy terms
        Heff_ex_my = exchange.effective_field(m)
        Heff_z_my = zeeman.effective_field(m)
        Heff_dmi_my = dmi.effective_field(m)
        H_eff = Heff_ex_my + Heff_z_my + Heff_dmi_my

        # Calculate the energy of energy terms
        E_ex = exchange.energy(m)
        E_z = zeeman.energy(m)
        E_dmi = dmi.energy(m)
        E = E_ex + E_z + E_dmi
        return H_eff, E

    def update_M(self, m, H_eff, lamda=0.005):
        '''
        The function is used to update the magnetization field,
        which changes the magnitude and direction of magnetization field

        Parameters
        ----------
        m: numpy.ndarray
            the magnetization field class of the system
        H_eff : numpy.ndarray
                Effective field
        lamda: float
                The damping constant.

        Returns
        -------
        numpy.ndarray
                The new magnetization field.

        '''
        M_old = m.get_M()
        Ms_old = m.get_Ms()
        # The norm of effective field
        # The shape should be (Nx, Ny, Nz, 1)
        H_norm = np.expand_dims(np.linalg.norm(H_eff, axis=3), axis=3)
        # No Temperature (T = 0, beta = inf)
        if beta == 9e99:
            # Limitation of Langevin(x->positive infinity) is 1
            L_result = 1
        # Change Temperature (T > 0)
        else:
            x = beta * mu0 * H_norm
            L_result = self.Langevin(x)
        # Effective field equals to 0 means the system
        # already in the final state, M should not change
        if (np.allclose(H_norm, 0)):
            M_new = M_old
        else:
            # Calculate new magnitude of magnetization field
            # Ms * Langevin(x) * (H_eff / |H_eff|)
            M_new = np.multiply(
                Ms_old, np.multiply(
                    L_result, (H_eff / H_norm)))
            # Calculate new direction of magnetization field
            M_lamda = M_old + lamda * (M_new - M_old)
            # Calculate norm of new magnetization field to use to renormalize
            M_new_norm = np.expand_dims(np.linalg.norm(M_new, axis=3), axis=3)
            # Calculate norm of M_lamda to use to uniform itself
            M_lamda_norm = np.expand_dims(
                np.linalg.norm(M_lamda, axis=3), axis=3)
            # Renormalization
            M_new = (M_lamda / M_lamda_norm) * M_new_norm

        return M_new

    def Mean_field_difference(self, m, tol=1e-4):
        '''The function takes in a magnetization object,
        returns the final magnetization object and total energy of the system.

        The function uses a while loop
        that iterates until the magnetization converges or reach maxiteration.

        The magnetization is updated by calling the update_M function.

        The effective field and energy are calculated
        by calling the cal_effective_field function.

        The magnetization is considered to have converged
        when the difference between the magnetization at the current iteration
        and the magnetization at the previous iteration
        is less than a tolerance value.

        The tolerance value is set to 1e-4 by default.

        The function prints the number of iterations it took.

        Parameters
        ----------
        m : numpy.ndarray
            the magnetization field class of the system
        tol : float
                tolerance for the stopping criteria

        Returns
        -------
        m : numpy.ndarray
            Final state of magnetization field class of the system
            with lowest energy.
        E : float
            Final total energy.

        '''
        count = 0
        H_eff, E = self.cal_effective_field(m)
        while count < maxiter:
            m_old = m.get_m()
            # update M
            M_new = self.update_M(m, H_eff)
            m.set_M(M_new)
            m_new = m.get_m()
            H_eff, E = self.cal_effective_field(m)
            # stop criteria
            diff = m_new - m_old
            max_value = np.max(abs(diff))
            if np.allclose(max_value, 0, atol=tol):
                break
            count += 1
            if (count % 1000 == 0):
                print(count)

        print("Number of iteration: ", count)
        return m, E, count
