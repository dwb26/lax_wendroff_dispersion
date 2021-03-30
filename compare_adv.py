import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from solvers import Lax_Wendroff

# ------------------------------------------------------------------------------------------------------------------- #
#
# Parameters
#
# ------------------------------------------------------------------------------------------------------------------- #
xL = 0; xR = 1
nx = 250
cfl = 0.8
dx = (xR - xL) / (nx - 1)
xs = np.linspace(xL - dx, xR + dx, nx + 2)
T_stop = 5.0


# ------------------------------------------------------------------------------------------------------------------- #
#
# Functions
#
# ------------------------------------------------------------------------------------------------------------------- #
def generate_ics(xs):
	f1 = np.exp(-xs ** 2 / 0.01)
	f2 = np.abs(xs - 0.5) < 0.1
	f3 = np.exp(-(xs - 1) ** 2 / 0.01)
	return f1 + f2 + f3

def periodic_ics(xs):
	return generate_ics(np.mod(xs, 1))


# ------------------------------------------------------------------------------------------------------------------- #
#
# Lax-Wendroff
#
# ------------------------------------------------------------------------------------------------------------------- #
ics_lw = generate_ics(xs)
LW = Lax_Wendroff()
Q_lw, dts = LW.solve(ics_lw, cfl, nx, dx, T_stop)


# ------------------------------------------------------------------------------------------------------------------- #
#
# Exact solution
#
# ------------------------------------------------------------------------------------------------------------------- #
ics = generate_ics(xs)
exact_soln = []
exact_soln.extend(ics)
t_count = 1
for t in dts:
	exact_soln.extend(periodic_ics(xs - t_count * t))
	t_count += 1
exact_soln = np.array(exact_soln).reshape((t_count, nx + 2))
exact_soln = exact_soln[:, 1 : nx + 1]


# ------------------------------------------------------------------------------------------------------------------- #
#
# Plotting
#
# ------------------------------------------------------------------------------------------------------------------- #
fig = plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
ax.set(ylim=(-0.2, 1.2))

ani_plot = True
# ani_plot = False
if ani_plot:
	line1, = ax.plot(xs[1 : nx + 1], exact_soln[0])
	line2, = ax.plot(xs[1 : nx + 1], Q_lw[0])
	def update(n):
		line1.set_data(xs[1 : nx + 1], exact_soln[n])
		line2.set_data(xs[1 : nx + 1], Q_lw[n])
		line2.set_marker("o")
		line2.set_markersize(2)
		# line2.set_fillstyle(None)
		line2.set_linestyle("--")
		return line1, line2,

	ani = animation.FuncAnimation(fig, func=update, frames=range(1, t_count))
	plt.show()