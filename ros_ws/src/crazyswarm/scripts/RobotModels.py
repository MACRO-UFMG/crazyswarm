import numpy as np
import casadi as ca

class Agent:
  def __init__(self, p, v, a, r, Ts, id=None):
    self.position = p
    self.velocity = v
    self.acceleration = a
    self.radius = r
    self.Ts = Ts
    self.id = id

  def step(self, u):
    self.acceleration = u
    self.position = self.position + self.velocity*self.Ts + 0.5*self.acceleration*self.Ts**2
    self.velocity = self.velocity + self.acceleration*self.Ts

  def plot_agent(self):
      # Make data
      u = np.linspace(0, 2 * np.pi, 100)
      v = np.linspace(0, np.pi, 100)

      x = self.position[0] + self.radius * np.outer(np.cos(u), np.sin(v))
      y = self.position[1] + self.radius * np.outer(np.sin(u), np.sin(v))
      z = self.position[2] + self.radius * np.outer(np.ones(np.size(u)), np.cos(v))

      return x, y, z

  def collisionCheck(self, colliders):
    for collider in colliders:
      dp = self.position - collider.position
      if np.linalg.norm(dp) < self.radius + collider.radius:
        return True
    return False

  def isItself(self, collider):
    return self.id == collider.id

class AgentCBF(Agent):
  def __init__(self, p, v, a, r, Ts, a_max, gamma):
    super().__init__(p, v, a, r, Ts)

    self.maximum_acceleration = a_max
    self.gamma = gamma

  def computeCBF(self, collider):

    dp = self.position - collider.position
    dv = self.velocity - collider.velocity
    da_max = 2*self.maximum_acceleration

    Ds = self.radius + collider.radius
    
    norm_dp = np.linalg.norm(dp)

    dot_dvdp = np.dot(dv, dp)

    if dot_dvdp >= 0:
      return 0,0,False

    h = np.dot(dp/norm_dp, dv) + np.sqrt(2*da_max*(norm_dp-Ds))

    A = -dp
    b = self.gamma*(h**3)*norm_dp - (dot_dvdp**2)/(norm_dp**2) + np.linalg.norm(dv)**2 + (da_max*dot_dvdp)/np.sqrt(2*da_max*(norm_dp - Ds))

    return A, b, True

  def control(self, reference, k=1):
    v_ref = k*(reference - self.position)
    x_ref = np.concatenate((self.position+v_ref*self.Ts, v_ref))

    return x_ref

  def step(self, reference, collider):

    alpha = 0.001
    Q = np.diag(np.ones(6))
    R = np.diag(alpha*np.ones(3))

    x_ref = self.control(reference)

    opti = ca.Opti()
    x = opti.variable(6,1)
    u = opti.variable(3,1)
    opts_setting = {'ipopt.max_iter':2000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-4, 'ipopt.acceptable_obj_change_tol':1e-4}
    opti.solver('ipopt', opts_setting)
    opti.minimize(ca.mtimes([(x-x_ref).T, Q, x-x_ref]) + ca.mtimes([u.T, R, u]))

    x_next = self.position + self.velocity*self.Ts + 0.5*u*self.Ts**2
    vx_next = self.velocity + u*self.Ts            
    opti.subject_to(x[0:3]==x_next)        
    opti.subject_to(x[3:6]==vx_next)

    A, b, approaching = self.computeCBF(collider)
    if approaching:
      opti.subject_to(A*u <= b)
    opti.subject_to(u**2 <= self.maximum_acceleration**2)

    self.acceleration = opti.solve().value(u)
    self.position = self.position + self.velocity*self.Ts + 0.5*self.acceleration*self.Ts**2
    self.velocity = self.velocity + self.acceleration*self.Ts

class AgentVectorFieldCBF(AgentCBF): 
  def __init__(self, p, v, a, r, Ts, a_max, gamma, field):
    super().__init__(p, v, a, r, Ts, a_max, gamma)

    self.field = field

  def control(self, t):
    v_ref = self.field.compute(self.position, t)
    x_ref = np.concatenate((self.position+v_ref*self.Ts, v_ref))

    return x_ref

  def step(self, colliders, t):
    alpha = 0.001
    Q = np.diag(np.ones(6))
    R = np.diag(alpha*np.ones(3))

    x_ref = self.control(t)

    opti = ca.Opti()
    x = opti.variable(6,1)
    u = opti.variable(3,1)
    opts_setting = {'ipopt.max_iter':8000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-4, 'ipopt.acceptable_obj_change_tol':1e-4}
    opti.solver('ipopt', opts_setting)
    opti.minimize(ca.mtimes([(x-x_ref).T, Q, x-x_ref]) + ca.mtimes([u.T, R, u]))

    x_next = self.position + self.velocity*self.Ts + 0.5*u*self.Ts**2
    vx_next = self.velocity + u*self.Ts            
    opti.subject_to(x[0:3]==x_next)        
    opti.subject_to(x[3:6]==vx_next)

    for collider in colliders:
      # if np.linalg.norm(self.position - collider.position) > 5*self.radius:
      #   continue
      # if np.linalg.norm(self.position - collider.position) < self.radius + collider.radius and not all(self.position == collider.position):
      #   print("Collision")
      A, b, approaching = self.computeCBF(collider)
      if approaching:
        opti.subject_to(A*u <= b)
      # opti.subject_to(opti.bounded(-self.maximum_acceleration, u[0], self.maximum_acceleration))
      # opti.subject_to(opti.bounded(-self.maximum_acceleration, u[1], self.maximum_acceleration))
      # opti.subject_to(opti.bounded(-self.maximum_acceleration, u[2], self.maximum_acceleration))

    self.acceleration = opti.solve().value(u)
    self.position = self.position + self.velocity*self.Ts + 0.5*self.acceleration*self.Ts**2
    self.velocity = self.velocity + self.acceleration*self.Ts

class AgentMPC_CBF(AgentVectorFieldCBF):
  def __init__(self, p, v, a, r, Ts, a_max, gamma, field, h, R, Q, R_du=None, propagation_method="constant", displacement=False, com_range=None):
    super().__init__(p, v, a, r, Ts, a_max, gamma, field)

    self.h = h
    self.R = R
    self.Q = Q
    self.R_du = R_du
    self.propagation_method = propagation_method
    self.displacement = displacement
    self.com_range = com_range

    self.propagated_position = []
    self.propagated_velocity = []

  def propagateConstraints(self, collider=None, t=None):

    self.predicted_position = [self.position]
    self.predicted_velocity = [self.velocity]

    for i in range(1,self.h):
      predicted_position = self.predicted_position[-1] + self.predicted_velocity[-1]*self.Ts

      if self.propagation_method == "constant":
        predicted_velocity = self.velocity
      elif self.propagation_method == "field":
        predicted_velocity = self.field.compute(predicted_position, t + i*self.Ts)

      if collider:
        distance = np.linalg.norm(predicted_position - collider.predicted_position[i])
        if distance < 0.1 + self.radius + collider.radius and not distance == 0 and not i == 0:
          print("Collision in step " +str(i)+ " of the prediction horizon with distance: " + str(distance - (self.radius + collider.radius)))

          n = (predicted_position - collider.predicted_position[i])
          unit_n = n/np.linalg.norm(n)
          w = unit_n*(self.radius + collider.radius - np.linalg.norm(n) + 0.1)
          
          predicted_position = predicted_position + w

          if self.propagation_method == "constant":
            predicted_velocity = (predicted_position - self.predicted_position[-1])/self.Ts
          elif self.propagation_method == "field":
            predicted_velocity = self.field.compute(predicted_position, t + i*self.Ts)

          distance = np.linalg.norm(predicted_position - collider.predicted_position[i])
          print("New distance in propagation process: " + str(distance - (self.radius + collider.radius)))

      self.predicted_position.append(predicted_position)
      self.predicted_velocity.append(predicted_velocity)

  def computeReference(self, t):
    reference = []
    reference_position = self.position
    reference_velocity = np.array([0., 0., 0.])
    for i in range(self.h):
      reference_position = reference_position + reference_velocity*self.Ts
      reference_velocity = self.field.compute(reference_position, t + i*self.Ts)
      reference.append(np.concatenate((reference_position, reference_velocity)))

    return reference

  def computeCBF(self, collider, prediction_step):
    dp = self.predicted_position[prediction_step] - collider.predicted_position[prediction_step]
    dv = self.predicted_velocity[prediction_step] - collider.predicted_velocity[prediction_step]
    da_max = 2*self.maximum_acceleration

    Ds = self.radius + collider.radius
    norm_dp = np.linalg.norm(dp)
    dot_dvdp = np.dot(dv, dp)

    if -np.dot(dp/norm_dp, dv) > np.sqrt(2*da_max*(norm_dp-Ds)) and not prediction_step==0:
      print("\033[92m {}\033[00m" .format("Constraint broken in step " + str(prediction_step)))
      return 0,0,False
    if np.linalg.norm(dp) < self.radius + collider.radius and not np.linalg.norm(dp) == 0 and not prediction_step == 0:
      print("Collision in step " +str(prediction_step)+ " of the prediction horizon with distance: " + str(np.linalg.norm(dp) - (self.radius + collider.radius)))
      return 0,0,False

    if dot_dvdp >= 0:
      return 0,0,False

    h = np.dot(dp/norm_dp, dv) + np.sqrt(2*da_max*(norm_dp-Ds))
    A = -dp
    b = self.gamma*(h**3)*norm_dp - (dot_dvdp**2)/(norm_dp**2) + np.linalg.norm(dv)**2 + (da_max*dot_dvdp)/np.sqrt(2*da_max*(norm_dp - Ds))

    return A, b, True

  def step(self, colliders, t):

    reference = self.computeReference(t)

    opti = ca.Opti()
    x = opti.variable(6,self.h)
    u = opti.variable(3,self.h)
    opts_setting = {'ipopt.max_iter':8000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-4, 'ipopt.acceptable_obj_change_tol':1e-4}
    opti.solver('ipopt', opts_setting)

    cost_function = 0
    cost_function += ca.mtimes([(x[:,0]-reference[0]).T, self.Q, x[:,0]-reference[0]]) + ca.mtimes([u[:,0].T, self.R, u[:,0]])
    for i in range(1, self.h):
      if self.R_du is None:
        cost_function += ca.mtimes([(x[:,i]-reference[i]).T, self.Q, x[:,i]-reference[i]]) + ca.mtimes([u[:,i].T, self.R, u[:,i]])
      else:
        cost_function += ca.mtimes([(x[:,i]-reference[i]).T, self.Q, x[:,i]-reference[i]]) + ca.mtimes([u[:,i].T, self.R, u[:,i]]) + ca.mtimes([u[:,i].T - u[:,i-1].T, self.R_du, u[:,i] - u[:,i-1]])

    opti.minimize(cost_function)

    opti.subject_to(x[0:3, 0]==self.position)        
    opti.subject_to(x[3:6, 0]==self.velocity)
    for i in range(self.h-1):
      x_next = x[0:3, i] + x[3:6, i]*self.Ts + 0.5*u[:,i]*self.Ts**2
      vx_next = x[3:6, i] + u[:,i]*self.Ts            
      opti.subject_to(x[0:3, i+1]==x_next)        
      opti.subject_to(x[3:6, i+1]==vx_next)

    for collider in colliders:
      if all(self.position == collider.position):
        continue

      if self.displacement:
        self.propagateConstraints(collider, t)
        
      for i in range(self.h):
        if i == 0:
          distance = np.linalg.norm(self.position - collider.position)
          if distance < self.radius + collider.radius and not distance == 0:
            print("\033[91m {}\033[00m" .format("[ERROR] Real Collision Ocurrence: " + str(distance - (self.radius + collider.radius))))

        if self.com_range is None:
          A, b, approaching = self.computeCBF(collider, i)
        else:
          distance = np.linalg.norm(self.position - collider.position)
          if distance < self.com_range:
            A, b, approaching = self.computeCBF(collider, i)
          else:
            approaching = False
            
        if approaching:
          opti.subject_to(A*u[:,i] <= b)
    opti.subject_to(opti.bounded(-self.maximum_acceleration, u[0,0], self.maximum_acceleration))
    opti.subject_to(opti.bounded(-self.maximum_acceleration, u[1,0], self.maximum_acceleration))
    opti.subject_to(opti.bounded(-self.maximum_acceleration, u[2,0], self.maximum_acceleration))

    self.acceleration = opti.solve().value(u[:,0])

    if np.linalg.norm(self.acceleration) > self.maximum_acceleration:
      print("\033[93m {}\033[00m" .format("[WARNING] Acceleration is higher than expected: " + str(np.linalg.norm(self.acceleration))))

    self.position = self.position + self.velocity*self.Ts + 0.5*self.acceleration*self.Ts**2
    self.velocity = self.velocity + self.acceleration*self.Ts