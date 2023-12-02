import numpy as np
from utils import *

class CollisionAvoidance:
    def __init__(self, H, RADIUS, MAX_ACC, GAMMA):
        self.agent_dynamic = None

        self.H = H
        self.RADIUS = RADIUS
        self.MAX_ACC = MAX_ACC
        self.GAMMA = GAMMA

    def set_agent_dynamic(self, A):
        self.agent_dynamic = A

    def linear_state_propagation(self, state, h):
        propagated_states = [state]
        for _ in range(h):
            propagated_states.append(np.dot(self.agent_dynamic, propagated_states[-1]))

        return propagated_states
    
    def compute_reciprocal_velocity_obstacles(self, agent_state, collider_state, tau, dt):

        x = -(agent_state[:3] - collider_state[:3])
        v = agent_state[3:] - collider_state[3:]
        r = 2*self.RADIUS

        x_len_sq = np.dot(x,x)
        v_len_sq = np.dot(v,v)

        collision = False

        if x_len_sq >= r * r:
            adjusted_center = x/tau * (1 - (r*r)/(x_len_sq+1e-10))

            if np.dot(v - adjusted_center, adjusted_center) < 0:
                # v lies in the front part of the cone
                # print("front")
                
                w = v - x/tau
                u = normalized(w) * r/tau - w
                n = normalized(w)
                # print("front", adjusted_center, x_len_sq, r, x, u, n)
                
            else: # v lies in the rest of the cone
                # print("back")
                a = x_len_sq
                b = np.dot(x,v)
                c = v_len_sq - np.dot(np.cross(x, v),np.cross(x, v)) / (x_len_sq - r*r +1e-5)
                # print((np.sqrt(b * b - a * c)) , b * b - a * c)
                if a ==0 or (b * b - a * c <0): t1 = 0
                else: t1 = (b + np.sqrt(b * b - a * c)) / (a+1e-18)
                
                # print((b + np.sqrt(b * b - a * c)) / (a+1e-5))
                # t1 = (b + np.sqrt(b * b - a * c)) / (a+1e-5)

                n = v - t1 * x
                nLength = np.linalg.norm(n)
                #if nLength ==0: unitN = n
                # else: unitN = n / nLength
                unitN = n / (nLength+1e-18)

                u = (r * t1 - nLength) * unitN
                
        else:
            # We're already intersecting. Pick the closest velocity to our
            # velocity that will get us out of the collision within the next
            # timestep.
            # print("intersecting")
            w = v - x/dt
            u = normalized(w) * r/dt - w
            n = normalized(w)
            # print("intersecting", w, v, r, x, u, n)
            collision = True
        return u, n, collision

    def compute_group_orca(self, agent, colliders, tau, dt):

        agent_state = np.concatenate((agent.position, agent.velocity))
        agent_propagated_states = self.linear_state_propagation(agent_state, self.H)

        collider_predicted_u_orca = []
        collider_predicted_n_orca = []
        for collider in colliders:
            if agent.isItself(collider):
                continue
            collider_state = np.concatenate((collider.position, collider.velocity))
            collider_propagated_states = self.linear_state_propagation(collider_state, self.H)

            predicted_u_orca = []
            predicted_n_orca = []
            for i in range(self.H):
                u_orca, n_orca, _ = self.compute_reciprocal_velocity_obstacles( agent_propagated_states[i],
                                                                                collider_propagated_states[i],
                                                                                tau,
                                                                                dt)
                predicted_u_orca.append(u_orca)
                predicted_n_orca.append(n_orca)
            collider_predicted_u_orca.append(predicted_u_orca)
            collider_predicted_n_orca.append(predicted_n_orca)

        return collider_predicted_u_orca, collider_predicted_n_orca, agent_propagated_states

    def compute_control_barrier_function(self, agent_state, collider_state, prediction_step):

        dp = agent_state[:3] - collider_state[:3]
        dv = agent_state[3:] - collider_state[3:]
        da_max = 2*self.MAX_ACC

        Ds = self.RADIUS + self.RADIUS
        norm_dp = np.linalg.norm(dp)
        dot_dvdp = np.dot(dv, dp)

        if -np.dot(dp/norm_dp, dv) > np.sqrt(2*da_max*(norm_dp-Ds)):
            print("\033[92m {}\033[00m" .format("Constraint broken in step " + str(prediction_step)))
            if not prediction_step==0:
                return 0,0,False
        if np.linalg.norm(dp) < self.RADIUS + self.RADIUS and not np.linalg.norm(dp) == 0:
            print("Collision in step " +str(prediction_step)+ " of the prediction horizon with distance: " + str(np.linalg.norm(dp) - (self.RADIUS + self.RADIUS)))
            if not prediction_step==0:
                return 0,0,False

        if dot_dvdp >= 0:
            return 0,0,False

        h = np.dot(dp/norm_dp, dv) + np.sqrt(2*da_max*(norm_dp-Ds))
        A = -dp
        b = self.GAMMA*(h**3)*norm_dp - (dot_dvdp**2)/(norm_dp**2) + np.linalg.norm(dv)**2 + (da_max*dot_dvdp)/np.sqrt(2*da_max*(norm_dp - Ds))

        return A, b, True

    def compute_group_barrier_function(self, agent, colliders):

        agent_state = np.concatenate((agent.position, agent.velocity))
        agent_propagated_states = self.linear_state_propagation(agent_state, self.H)

        collider_predicted_A_cbf = []
        collider_predicted_b_cbf = []
        for collider in colliders:
            if agent.isItself(collider):
                continue
            collider_state = np.concatenate((collider.position, collider.velocity))
            collider_propagated_states = self.linear_state_propagation(collider_state, self.H)

            predicted_A_cbf = []
            predicted_b_cbf = []
            for i in range(self.H):
                A_cbf, b_cbf, _ = self.compute_control_barrier_function(agent_propagated_states[i],
                                                                        collider_propagated_states[i],
                                                                        prediction_step=i)
                predicted_A_cbf.append(A_cbf)
                predicted_b_cbf.append(b_cbf)
            collider_predicted_A_cbf.append(predicted_A_cbf)
            collider_predicted_b_cbf.append(predicted_b_cbf)

        return collider_predicted_A_cbf, collider_predicted_b_cbf