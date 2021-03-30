import numpy as np

class Lax_Wendroff(object):
	def __init__(self):
		pass

	def solve(self, Q, cfl, nx, dx, T_stop):

		# Set the periodic boundary conditions
		Q[-1] = Q[-2]
		Q[0] = Q[-1]
		Q_forward = np.empty(nx + 2)

		Q_soln = []
		dts = []
		Q_soln.extend(Q)

		A = 1
		dt = cfl * dx / A
		dts.append(dt)
		t = dt
		counter = 1
		while t < T_stop:
			for i in range(1, nx + 1):
				first_term = dt / (2 * dx) * A * (Q[i + 1] - Q[i - 1])
				second_term = 0.5 * (dt / dx) ** 2 * A ** 2 * (Q[i - 1] - 2 * Q[i] + Q[i + 1])
				Q_forward[i] = Q[i] - first_term + second_term
			Q[1 : nx + 1] = np.copy(Q_forward[1 : nx + 1])
			Q[-1] = Q[-2]
			Q[0] = Q[-1]
			Q_soln.extend(Q)
			t += dt
			counter += 1
			dts.append(dt)

		Q_soln = np.array(Q_soln).reshape((counter, nx + 2))
		return Q_soln[:, 1 : nx + 1], dts
