"""
    PyNEC Utilities
    Copyright (C) 2021 Electro707 (develop@electro707.com)

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
import os
import PyNEC
import matplotlib
import numpy as np
from dataclasses import dataclass
import matplotlib
from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.animation as animation
from scipy.linalg import norm
# import mayavi.mlab as mlab
import typing


def set_axes_equal(ax):
    '''
    From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


@dataclass
class RadiationPatternData:
    """ A data class for the radiation pattern data """
    X: np.array = None      # An array of X data points for the radiation pattern
    Y: np.array = None      # An array of Y data points for the radiation pattern
    Z: np.array = None
    N: np.array = None
    gains: np.array = None
    freq: float = None


class GraphRadiationPattern:
    def __init__(self, in_data: typing.Union[RadiationPatternData, list[RadiationPatternData]], rotate: bool = False, elevation: float = 30):
        self.data = in_data
        self.do_rotate = rotate
        self.elevation = elevation
        self.multiple_data = True
        if isinstance(self.data, RadiationPatternData):
            self.data = [self.data]
            self.multiple_data = False
        if self.do_rotate:
            self.numb_frames = 360
        else:
            self.numb_frames = len(self.data)

        # if not isinstance(self.data, list) and rotate is False:
        #     raise UserWarning("Not rotating nor is animating the pattern")

        g_l = [j.gains for j in self.data]
        i, i2, i3 = np.unravel_index(np.argmax(g_l), np.shape(g_l))
        max_g = self.data[i].gains[i2][i3]
        self.mycol = cm.jet(self.data[i].N)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.plot1 = self.ax.plot_surface(self.data[0].X, self.data[0].Y, self.data[0].Z, rstride=1, cstride=1, facecolors=self.mycol, linewidth=0.5, antialiased=True, shade=False)
        self.ax.grid(True)
        self.ax.view_init(self.elevation, 45)
        # set_axes_equal(self.ax)
        self.ax.set_xlim3d([-max_g, max_g])
        self.ax.set_ylim3d([-max_g, max_g])
        self.ax.set_zlim3d([-max_g, max_g])
        self.ax.set_title("3D Radiation Plot for %.3f Mhz" % (self.data[0].freq/1e6))

        self.m = cm.ScalarMappable(cmap=cm.jet)
        self.colorbar = self.fig.colorbar(self.m, shrink=0.8, ax=self.ax)
        self.change_colorbar(i)

        if isinstance(self.data, list) or rotate is True:
            self.ani = animation.FuncAnimation(self.fig, self._update, self.numb_frames, interval=1/60)

    def change_colorbar(self, index):
        self.m.set_array(self.data[index].gains)
        self.m.autoscale()
        self.m.changed()

    def add_3d_lines(self, line_arr):
        for l in line_arr:
            self.ax.plot3D(*l)

    def show(self):
        plt.show()

    def export_to_gif(self, file_name):
        self.ani.save(file_name, dpi=300, fps=60, writer='ffmpeg')

    def export_to_mp4(self, file_name):
        writermp4 = animation.FFMpegWriter(fps=60)
        self.ani.save(file_name+'.mp4', writer=writermp4, dpi=300)

    def _update(self, frame_numb):
        if self.multiple_data:
            i = int((frame_numb / (self.numb_frames/2)) * len(self.data)) % len(self.data)
            self.plot1.remove()
            # mycol = cm.jet(self.data[i].N)
            self.plot1 = self.ax.plot_surface(self.data[i].X, self.data[i].Y, self.data[i].Z, rstride=1, cstride=1, facecolors=self.mycol, linewidth=0.5, antialiased=True, shade=False)
            # self.change_colorbar(i)
            self.ax.set_title("3D Radiation Plot for %.3f Mhz" % (self.data[i].freq/1e6))
        if self.do_rotate:
            self.ax.view_init(self.elevation, 45+frame_numb)


class GraphAntennaDesign:
    """
        Class to animate and/or plot the antenna
    """
    def __init__(self, elevation: float = 30, rotate: bool = False):
        self.elevation = elevation
        self.rotate = rotate
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.axis('off')
        self.ax.view_init(self.elevation, 45)
        if self.rotate:
            self.ani = animation.FuncAnimation(self.fig, self._update, 360, interval=50)

    def export_to_mp4(self, file_name):
        """
            Export an animation, if initialized with rotate=True, of the
            antenna rotating
        """
        if self.rotate is False:
            raise UserWarning("Class is not set to rotate, cannot export animation")
        writermp4 = animation.FFMpegWriter(fps=60)
        self.ani.save(file_name+'.mp4', writer=writermp4, dpi=300)

    def _update(self, frame_numb):
        """ Internal FuncAnimation update function """
        self.ax.view_init(self.elevation, 45+frame_numb)

    def show(self):
        """
            Show the plot
        """
        set_axes_equal(self.ax)
        plt.show()


class PyNECWrapper:
    def __init__(self):
        self.nec = PyNEC.nec_context()
        self.geo = self.nec.get_geometry()

        self._last_tag_id = 0
        self._all_wires = {}
        self._ex_wire = None

        self.numb_freq_index = 0

    def add_wire(self, coords_1: list, coords_2: list, wire_rad: float, numb_segments: int):
        """
            Adds a wire to the NEC simulation

            Args:
                - coords_1: The start coordinates of the wire as a list [x0, y0, z0]
                - coords_2: The start coordinates of the wire as a list [x1, y1, z1]
                - wire_rad: THe radius of the wire
                - numb_segments: The number of segments to split the wire into for the simulation
                                 This should be at least ** / wavelenght

            Returns:
                The internal WireID of the wire
        """
        self._last_tag_id += 1
        self._all_wires[self._last_tag_id] = {
            'coords': np.array([coords_1, coords_2]),
            'id': self._last_tag_id,
            'width': wire_rad,
            'numb_seg': numb_segments,
        }
        self.geo.wire(self._last_tag_id, numb_segments, *coords_1, *coords_2, wire_rad, 1.0, 1.0)
        return self._last_tag_id

    def geometry_complete(self, is_gound_plane: bool = False, current_expansion: bool = True):
        """
            Call this function when done with making the geometry
        """
        if not is_gound_plane:
            self.nec.geometry_complete(0)
        else:
            if current_expansion:
                self.nec.geometry_complete(1)
            else:
                self.nec.geometry_complete(-1)

    def add_exitation(self, wire_id: int, place_seg: int):
        """
            Adds an exitation source

            Args:
                wire_id (int): The WireID of the wire to apply the exitation on
                place_seg: The segment of the wire to place the exitation on
        """
        self._ex_wire = {'wire_id': wire_id, 'where_seg': int(place_seg)}
        self.nec.ex_card(0, wire_id, int(place_seg), 0, 1.0, 0, 0, 0, 0, 0)

    def calculate(self, n: float):
        """
            Calculates the antenna

            Args:
                n: The number of segments for the delta and phi.
        """
        self.nec.rp_card(calc_mode=0, n_theta=n, n_phi=n, output_format=0, normalization=0, D=0, A=0, theta0=0,
                         delta_theta=180/(n-1), phi0=0, delta_phi=360/(n-1), radial_distance=0, gain_norm=0)
        self.nec.xq_card(0)

    def set_single_f(self, freq: float):
        """
            Sets a single frequency for the calculation

            Args:
                freq: The frequency to simulate for, in Mhz
        """
        self.numb_freq_index = 1
        self.nec.fr_card(0, 1, freq, 0)

    def set_multiple_f(self, min_f, max_f, n):
        """
            Set multiple frequencies for the calculation

            Args:
                min_f: The minimum simulation frequency, in Mhz
                max_f: The maximum simulation frequency, in Mhz
                n: The number of frequency steps to simulate for
        """
        step = (max_f - min_f)/n
        self.numb_freq_index = n
        self.nec.fr_card(0, n, min_f, step)

    def get_3d_radiation_pattern(self, freq_index: int = 0) -> RadiationPatternData:
        """
            Get the radiation pattern for a given frequency index
        """
        ret = RadiationPatternData()
        rpt = self.nec.get_radiation_pattern(freq_index)
        ret.freq = rpt.get_frequency()

        gains_db = rpt.get_gain()
        ret.gains = 10.0**(gains_db / 10.0)
        thetas = rpt.get_theta_angles()
        phis = rpt.get_phi_angles()

        thetas = np.deg2rad(thetas)
        phis = np.deg2rad(phis)
        n_phis, n_thetas = np.meshgrid(phis, thetas)

        ret.X = ret.gains * np.sin(n_thetas) * np.cos(n_phis)
        ret.Y = ret.gains * np.sin(n_thetas) * np.sin(n_phis)
        ret.Z = ret.gains * np.cos(n_thetas)
        N = np.sqrt(ret.X**2 + ret.Y**2 + ret.Z**2)
        Rmax = np.max(N)
        ret.N = N/Rmax

        return ret

    def get_all_freq_3d_radiation_pattern(self) -> list[RadiationPatternData]:
        return [self.get_3d_radiation_pattern(i) for i in range(self.numb_freq_index)]

    def plot_3d_radiation_pattern(self, in_data: RadiationPatternData = None, freq_index: int = 0):
        """
            Function to plot the 3D radiation pattern of the antenna
        """
        if in_data is None:
            in_data = self.get_3d_radiation_pattern(freq_index)

        plot = GraphRadiationPattern(in_data)
        plot.show()

    # Partially copied from https://github.com/tmolteno/python-necpp/blob/master/PyNEC/example/antenna_util.py
    @staticmethod
    def get_reflection_coefficient(z, z0):
        return np.abs((z - z0)/(z + z0))

    def calculate_vswr(self, z, z0):
        gamma = self.get_reflection_coefficient(z, z0)
        return float((1+gamma) / (1-gamma))

    def get_vswr(self, z0: float = 50.0):
        """
            Gets the VSWR for the given calculation frequency or frequencies.

            Args:
                z0 (float): The impedance to calcular the VSRW over. Defaults to 50 ohms

            Returns:
                A tuple 2 lists, one for frequencies and the other for the VSWR
        """
        freqs, vswrs = [], []
        for i in range(self.numb_freq_index):
            ipt = self.nec.get_input_parameters(i)
            impedance = ipt.get_impedance()
            freqs.append(ipt.get_frequency())
            vswrs.append(self.calculate_vswr(impedance, z0))
        return freqs, vswrs

    def plot_swr(self, z0: float = 50.0):
        """
            Plots the VSWR of the antenna

            Args:
                z0 (float): The impedance to calcular the VSRW over. Defaults to 50 ohms
        """
        if self.numb_freq_index == 1:
            raise UserWarning("Must have more than 1 frequency point in order to create a SWR plot")
        freqs, vswrs = self.get_vswr(z0)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(freqs, vswrs)
        ax.grid(True)
        ax.set_title("VSWR for %.2fMhz-%.2fMhz" % (freqs[0]/1e6, freqs[len(freqs)-1]/1e6))
        plt.show()

    def add_antenna_to_axis(self, ax: plt.Axes):
        """
            WORK-In-Progress Function
            Adds surface plots for the antenna wires and elements to a MatPlotLib axis
        """
        def create_cyclinder(wire_coords: list, thickness: float):
            # https://stackoverflow.com/questions/32317247/how-to-draw-a-cylinder-using-matplotlib-along-length-of-point-x1-y1-and-x2-y2
            R = thickness / 2
            v = wire_coords[1] - wire_coords[0]   # vector in direction of axis
            mag = norm(v)                         # find magnitude of vector
            v /= mag                           # unit vector in direction of axis
            # make some vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if (v == not_v).all():
                not_v = np.array([0, 1, 0])
            n1 = np.cross(v, not_v)     # make vector perpendicular to v
            n1 /= norm(n1)        # normalize n1
            n2 = np.cross(v, n1)    # make unit vector perpendicular to v and n1
            t = np.linspace(0, mag, 10)    # surface ranges over t from 0 to length of axis and 0 to 2*pi
            theta = np.linspace(0, 2 * np.pi, 10)
            t, theta = np.meshgrid(t, theta)        # use meshgrid to make 2d arrays
            # generate coordinates for surface
            X, Y, Z = [wire_coords[0][i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
            return X, Y, Z

        for wire in self._all_wires.values():
            ax.plot_surface(*create_cyclinder(wire['coords'], wire['width']), color='black')
        # Calculate the exitation location
        ex_seg = self._all_wires[self._ex_wire['wire_id']]
        ex_seg_coords = ex_seg['coords']
        ex_seg_coords = np.array(list((
            (((ex_seg_coords[1, :] - ex_seg_coords[0, :]) / ex_seg['numb_seg']) * (self._ex_wire['where_seg']-1)) + ex_seg_coords[0, :],
            (((ex_seg_coords[1, :] - ex_seg_coords[0, :]) / ex_seg['numb_seg']) * (self._ex_wire['where_seg'])) + ex_seg_coords[0, :]
        )))
        ax.plot_surface(*create_cyclinder(ex_seg_coords, ex_seg['width']*1.5), color='red')


if __name__ == '__main__':
    print("Test with a basic 2m band dipole antenna")
    dipole_sim = PyNECWrapper()
    w_id = dipole_sim.add_wire([0, 0, -1], [0, 0, 1], 0.03, 36)
    dipole_sim.geometry_complete()
    dipole_sim.add_exitation(w_id, 18)
    dipole_sim.set_single_f(144)
    dipole_sim.calculate(36)
    anim_r = GraphRadiationPattern(dipole_sim.get_3d_radiation_pattern())
    anim_r.show()
