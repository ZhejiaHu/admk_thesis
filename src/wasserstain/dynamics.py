"""
Discrete dmk dynamics in Python
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve


def dmk_solve(self):
    def tdensinit(self):
        """Initialization of the conductivities: mu_e ~ U(0,1)

        Returns:
            self.tdens: np.array, initialized conductivities
        """

        prng = np.random.RandomState(seed=self.seed)
        self.tdens = np.array(
            [prng.uniform(0, 1) for i in range(self.g.number_of_edges())]
        )

        return self.tdens

    def dyn(self):
        """Execute dynamics

        Returns:
            self.tdens: np.array, conductivities at convergence
            pot: np.array, potentials at convergence
            cost_stack: np.array, cost at each iteration
        """

        print("* running dynamics")

        def update(self, pot, relax_linsys):
            """One step update

            Parameters:
                pot: np.array, potential matrix on nodes

            Returns:
                tdens: np.array, updated conductivities
                pot: np.array, updated potential matrix on nodes
                info: bool, sanity check flag spsolve
            """

            # update mu
            
            B = self.B

            grad = diags(1 / self.length, 0) * B.transpose() * pot

            rhs_ode = (self.tdens ** self.pflux) * (grad ** 2) - self.tdens
            self.tdens = self.tdens + self.time_step * rhs_ode

            stiff = B * diags(self.tdens, 0) * diags(1 / self.length, 0) * B.transpose()
            stiff_relax = stiff + relax_linsys * identity(self.g.number_of_nodes())
            pot = spsolve(stiff_relax, self.forcing, use_umfpack=True)

            # sanity check
            if np.any(np.isnan(pot)):
                info = -1
                pass
            else:
                info = 0

            return self.tdens, pot, info

        def convergence(self, pot, cost, conv, it):
            """Evaluating convergence

            Parameters:
                pot: np.array, potential matrix on nodes
                it: int, iteration number
                cost: float, cost

            Returns:
                convergence_achieved: bool, updated convergence flag
                cost_update: float, updated cost
                abs_diff_cost: float, difference cost


            """

            td_mat = np.diag(self.tdens)
            #print(f"td_mat: {td_mat}, self.length: {self.length}, self.B: {self.B}, td_mat * diags(1 / self.length, 0) * np.transpose(self.B): {td_mat * diags(1 / self.length, 0) * np.transpose(self.B)}, pot: {pot}")
            #print("left:", type(td_mat * diags(1 / self.length, 0) * np.transpose(self.B)), getattr(td_mat * diags(1 / self.length, 0) * np.transpose(self.B), "shape", None), "np.ndim:", np.ndim(td_mat * diags(1 / self.length, 0) * np.transpose(self.B)))
            #print("right:", type(pot), getattr(pot, "shape", None), "np.ndim:", np.ndim(pot))
            flux_mat = np.matmul(
                (td_mat * diags(1 / self.length, 0) * np.transpose(self.B)).toarray(), pot
            )
            # print(len(flux_mat))

            flux_norm = flux_mat ** 2

            cost_update = np.dot(
                self.length, (flux_norm ** ((2 - self.pflux) / (3 - self.pflux)))
            )

            dc = abs(cost_update - cost) / self.time_step

            if dc < self.tol and it > 5:
                conv = True

            return conv, cost_update, flux_mat, self.tdens

        ####################################################################
        # INITIALIZATION
        ####################################################################

        it = 0
        # only needed if spsolve has problems (inside update)
        prng = np.random.RandomState(seed=self.seed)
        print(f"self.B shape: {self.B} | self.tdens shape: {self.tdens}")
        stiff = (
            self.B
            * diags(self.tdens, 0)
            * diags(1 / self.length, 0)
            * self.B.transpose()
        )

        relax_linsys = 1e-10
        stiff_relax = stiff + relax_linsys * identity(self.g.number_of_nodes())
        pot = spsolve(stiff_relax, self.forcing, use_umfpack=True)
        cost_stack = []
        ####################################################################
        conv = False
        cost = 0
        while not conv and it <= self.tot_time:

            ####################################################################
            # RUNNING THE DYNAMICS
            ####################################################################

            it += 1

            # update tdens-pot system
            tdens_old = self.tdens
            pot_old = pot

            # equations update
            self.tdens, pot, info = update(self, pot, relax_linsys)
            print(f"self.tdens: {self.tdens}")
            # singular Laplacian matrix
            if info != 0:
                self.tdens = (
                    tdens_old
                    + prng.rand(*tdens_old.shape) * np.mean(tdens_old) / 1000.0
                )
                pot = pot_old + prng.rand(*pot.shape) * np.mean(pot_old) / 1000.0

            conv, cost, flux_mat, td_mat = convergence(
                self,
                pot,
                cost,
                conv,
                it,
            )

            cost_stack.append(cost)
            if self.verbose:
                print("it = %d, cost = %f" % (it, cost))
            if conv:
                print("   convergence achieved")

        return td_mat, pot, flux_mat, cost_stack

    tdensinit(self)
    j = dyn(self)

    return j