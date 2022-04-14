*pynec-utilities* Examples
==========================================

The following are some examples of how to use this utility

Simple Dipole 3D Simulation
++++++++++++++++++++++++++++++
The following code will plot a simple vertical dipole

.. code-block:: python

    # Import the package
    import pynec_utilities
    # Create a PyNECWrapper class object
    dipole_sim = pynec_utilities.PyNECWrapper()
    # Create the antenna
    w_id = dipole_sim.add_wire([0, 0, -1], [0, 0, 1], 0.03, 36)
    # This command MUST be ran after creating the antenna's geometry
    dipole_sim.geometry_complete()
    # Add an excitation in the dipole wire at the 18th segment, which is half-way
    dipole_sim.add_excitation(w_id, 18)
    # Set the frequency to simulate at a 144Mhz
    dipole_sim.set_single_f(144)
    # Calculate the antenna radiation pattern with 36 segments for theta and phi
    dipole_sim.calculate(36)
    # Plot the 3D radiation pattern
    dipole_sim.plot_3d_radiation_pattern()