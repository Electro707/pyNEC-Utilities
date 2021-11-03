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
# import mayavi.mlab as mlab
import typing


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
        plt.show()


class PyNECWrapper:
    def __init__(self):
        self.nec = PyNEC.nec_context()
        self.geo = self.nec.get_geometry()
        # self.nec.set_extended_thin_wire_kernel(False)

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
        self._all_wires[self._last_tag_id] = np.array([coords_1, coords_2])
        self.geo.wire(self._last_tag_id, numb_segments, *coords_1, *coords_2, wire_rad, 1.0, 1.0)
        return self._last_tag_id

    def geometry_complete(self):
        """
            Call this function when done with making the geometry
        """
        self.nec.geometry_complete(0)

    def add_exitation(self, wire_id: int, place_seg: int, numb_segments: int):
        """
            Adds an exitation source

            Args:
                wire_id (int): The WireID of the wire to apply the exitation on
                place_seg: The segment of the wire to place the exitation on
        """
        self._ex_wire = {'wire_id': wire_id, 'where_seg': int(place_seg), 'numb_seg': numb_segments}
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
        if in_data is None:
            in_data = self.get_3d_radiation_pattern(freq_index)

        plot = AnimateRadiationPattern(in_data)
        plot.show()

    def add_antenna_to_axis(self, ax: plt.Axes):
        for wire in self._all_wires.values():
            ax.plot3D(wire[:, 0], wire[:, 1], wire[:, 2], color='black', linewidth=2)
        # Calculate the exitation location
        ex_seg = self._all_wires[self._ex_wire['wire_id']]
        ex_seg = np.array(list(zip((((ex_seg[1, :] - ex_seg[0, :]) / self._ex_wire['numb_seg']) * (self._ex_wire['where_seg']-1)) + ex_seg[0, :],
                          (((ex_seg[1, :] - ex_seg[0, :]) / self._ex_wire['numb_seg']) * (self._ex_wire['where_seg'])) + ex_seg[0, :])))
        ax.plot3D(ex_seg[0], ex_seg[1], ex_seg[2], linewidth=5, color='red')


if __name__ == '__main__':
    print("Test with a basic 2m band dipole antenna")
    dipole_sim = PyNECWrapper()
    w_id = dipole_sim.add_wire([0, 0, -1], [0, 0, 1], 0.03, 36)
    dipole_sim.geometry_complete()
    dipole_sim.add_exitation(w_id, 18, 36)
    dipole_sim.set_single_f(144)
    dipole_sim.calculate(36)
    anim_r = GraphRadiationPattern(dipole_sim.get_3d_radiation_pattern(), rotate=False, elevation=20)
    anim_r.show()
