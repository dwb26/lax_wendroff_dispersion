# lax_wendroff_dispersion
Practice model demonstrating over-dispersion of Lax_Wendroff method

Example model from LeVeque's "Finite volume methods for hyperbolic problems" in which the over-dispersion of the Lax_Wendroff method is evident 
when attempting to approximate a discontinuous solution to a linear hyperbolic PDE.

To run, do "ipython compare_adv.py" in the command line. The code returns an animation of the solutions up to time T = 5.
