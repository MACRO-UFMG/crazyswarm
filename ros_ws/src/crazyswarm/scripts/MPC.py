import numpy as np
import casadi as ca

class MPC:
    def __init__(self, nx, nu, Q, R, Rdu, h, Ts, ca_method='cbf', slack_variables=False):
        self.nx = nx
        self.nu = nu

        self.Q = Q
        self.R = R
        self.Rdu = Rdu

        self.h = h
        self.Ts = Ts

        self.A = None
        self.B = None

        self.reference = None

        self.ca_method = ca_method

        self.slack_variables = slack_variables

    def set_dynamics(self, A, B):
        self.A = A
        self.B = B

    def set_reference(self, reference):
        self.reference = reference

    def set_collision_avoidance_parameters(self, parameters):
        if self.ca_method == 'cbf':
            self.A_cbf = parameters['A']
            self.b_cbf = parameters['b']
        elif self.ca_method == 'orca':
            self.orca_u = parameters['orca_u']
            self.orca_n = parameters['orca_n']
            self.propagated_states = parameters['propagated_states']

    def compute_collision_avoidance_constraints(self, x, u, epsilon):
        hard_constraints_list = []
        soft_constraints_list = []
        if self.ca_method == 'cbf':
            for i in range(len(self.A_cbf)):
                for j in range(self.h):
                    if not isinstance(self.A_cbf[i][j], int):
                        if j > 0 and self.slack_variables:
                            constraint = self.A_cbf[i][j]*u[:,j] - self.b_cbf[i][j] <= epsilon[i,j]
                            soft_constraints_list.append(constraint)
                        else:
                            constraint = self.A_cbf[i][j]*u[:,j] <= self.b_cbf[i][j]
                            hard_constraints_list.append(constraint)
        elif self.ca_method == 'orca':
            for i in range(len(self.orca_u)):
                for j in range(self.h-1):
                    v0 = self.propagated_states[j][3:] + self.orca_u[i][j] / 2
                    value_orca = v0.dot(self.orca_n[i][j].T)
                    constraint = ca.dot(x[3:6, j+1], self.orca_n[i][j]) - value_orca >= 0
                    hard_constraints_list.append(constraint)

        return hard_constraints_list, soft_constraints_list

    def step(self, state):

        opti = ca.Opti()
        opts_setting = {'ipopt.max_iter':8000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-4, 'ipopt.acceptable_obj_change_tol':1e-4}
        opti.solver('ipopt', opts_setting)

        x = opti.variable(self.nx, self.h)
        u = opti.variable(self.nu, self.h)

        if self.slack_variables:
            epsilon = opti.variable(len(self.b_cbf), self.h)
        else:
            epsilon = None

        J = ca.mtimes([(x[:,0]-self.reference[0]).T, self.Q, x[:,0]-self.reference[0]]) + ca.mtimes([u[:,0].T, self.R, u[:,0]])
        for i in range(1, self.h):
            if self.Rdu is None:
                J += ca.mtimes([(x[:,i]-self.reference[i]).T, self.Q, x[:,i]-self.reference[i]]) \
                    + ca.mtimes([u[:,i].T, self.R, u[:,i]])
            else:
                J += ca.mtimes([(x[:,i]-self.reference[i]).T, self.Q, x[:,i]-self.reference[i]]) \
                    + ca.mtimes([u[:,i].T, self.R, u[:,i]]) \
                    + ca.mtimes([u[:,i].T - u[:,i-1].T, self.Rdu, u[:,i] - u[:,i-1]])
            if self.slack_variables:
                J += ca.mtimes([epsilon[:,i].T, epsilon[:,i]])

        opti.minimize(J)

        opti.subject_to(x[:,0] == state)
        for i in range(self.h - 1):
            opti.subject_to(x[:, i+1] == ca.mtimes(self.A, x[:, i]) + ca.mtimes(self.B, u[:, i]))

        collision_avoidance_constraints, collision_avoidance_soft_constraints = self.compute_collision_avoidance_constraints(x, u, epsilon)
        for con in collision_avoidance_constraints:
            opti.subject_to(con)
        for con in collision_avoidance_soft_constraints:
            opti.subject_to(con)

        return opti.solve().value(u[:,0])