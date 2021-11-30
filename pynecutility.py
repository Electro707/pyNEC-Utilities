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
import logging
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
import enum


"""
    Developer Notes:
        - The coordinate system for the radiation pattern for NEC is a spherical coordinate, with phi (what rotates if
        looking from top down), theta (what rotates if looking at the side), and radius. 
        - For generation, because phi is set to 360 degrees anything above 180 degrees for theta is redundant, thus not 
        generated.
        - When theta = 0, a vector will be pointing straight up. Something to keep in mind when plotting, as some may 
        expect theta = 0 when the vector is pointing along the xy plane, but adjusting that will be up to the user.
"""


def set_axes_equal(ax):
    """
    From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

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
class Radiation3DPatternData:
    """ A data class for a 3D radiation pattern data """
    X: np.array = None      # An array of X data points for the radiation pattern
    Y: np.array = None      # An array of Y data points for the radiation pattern
    Z: np.array = None
    N: np.array = None
    gains: np.array = None
    freq: float = None


@dataclass
class Radiation2DPatternData:
    """ A data class for a 2D radiation pattern """
    plot_theta: np.array = None
    plot_radius: np.array = None
    constant_elevation: float = None
    constant_azimuth: float = None
    freq: float = None


class Graph3DRadiationPattern:
    def __init__(self, in_data: typing.Union[Radiation3DPatternData, list[Radiation3DPatternData]], rotate: bool = False, elevation: float = 30):
        self.data = in_data
        self.do_rotate = rotate
        self.elevation = elevation
        self.multiple_data = True
        if isinstance(self.data, Radiation3DPatternData):
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
        self.ax.tick_params(axis=u'both', which=u'both', length=0,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False,
                            bottom=False, top=False, left=False, right=False)
        self.ax.set_title("3D Radiation Plot for %.3f Mhz" % (self.data[0].freq/1e6))

        self.m = cm.ScalarMappable(cmap=cm.jet)
        self.colorbar = self.fig.colorbar(self.m, shrink=0.8, ax=self.ax, label='dBi')
        self.change_colorbar(i)

        if isinstance(in_data, list) or rotate is True:
            self.ani = animation.FuncAnimation(self.fig, self._update, self.numb_frames)

    def change_colorbar(self, index):
        self.m.set_array(10*np.log10(self.data[index].gains))
        self.m.autoscale()
        self.m.changed()

    def add_3d_lines(self, line_arr):
        for l in line_arr:
            self.ax.plot3D(*l)

    def show(self):
        plt.show()

    def export_to_gif(self, file_name):
        self.ani.save(file_name, dpi=300, fps=60, writer='ffmpeg')

    def export_to_svg(self, file_name):
        plt.savefig(file_name+'.svg')

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

    @staticmethod
    def _decibel_formatter(x, pos):
        return "{:.2f}".format(10*np.log10(x))


class Graph2DRadiationPattern:
    def __init__(self, in_data: typing.Union[Radiation2DPatternData, list[Radiation2DPatternData]]):
        self.data = in_data     # type: list[Radiation2DPatternData]
        self.multiple_data = True
        if isinstance(self.data, Radiation2DPatternData):
            self.data = [self.data]
            self.multiple_data = False
        self.numb_frames = len(self.data)

        g_l = [j.plot_radius for j in self.data]
        i, i2 = np.unravel_index(np.argmax(g_l), np.shape(g_l))
        max_g = self.data[i].plot_radius[i2]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, polar=True)
        self.plot1 = self.ax.plot(self.data[0].plot_theta, self.data[0].plot_radius)[0]
        self.ax.grid(True)
        self.ax.set_ylim(0, max_g)
        self.ax.yaxis.set_major_formatter(self._decibel_formatter)

        self.set_plot_title(self.data[0])

        if isinstance(in_data, list):
            self.ani = animation.FuncAnimation(self.fig, self._update, self.numb_frames)

    @staticmethod
    def _decibel_formatter(x, pos):
        return "{:.2f}".format(10*np.log10(x))

    def show(self):
        plt.show()

    def export_to_gif(self, file_name):
        self.ani.save(file_name, dpi=300, fps=60, writer='ffmpeg')

    def export_to_svg(self, file_name):
        plt.savefig(file_name+'.svg')

    def export_to_mp4(self, file_name):
        writermp4 = animation.FFMpegWriter(fps=60)
        self.ani.save(file_name+'.mp4', writer=writermp4, dpi=300)

    def export_to_latex(self, file_name):
        old_backend = matplotlib.get_backend()
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
        plt.savefig(file_name+'.pgf')
        matplotlib.use(old_backend)

    def _update(self, frame_numb):
        if self.multiple_data:
            i = int((frame_numb / self.numb_frames) * len(self.data)) % len(self.data)
            self.plot1.set_data(self.data[i].plot_theta, self.data[i].plot_radius)
            self.set_plot_title(self.data[i])

    def set_plot_title(self, data: Radiation2DPatternData):
        title = "2D Radiation Plot for %.3f Mhz" % (data.freq/1e6)
        if data.constant_elevation is not None:
            title += " at %.1f degrees elevation" % data.constant_elevation
        elif data.constant_azimuth is not None:
            title += " at %.1f degrees azimuth" % data.constant_azimuth
        self.ax.set_title(title)


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
        self.log = logging.getLogger('PyNECWrapper')
        self.nec = PyNEC.nec_context()
        self.geo = self.nec.get_geometry()

        self._last_tag_id = 0
        self._all_wires = {}
        self._ex_wire = None

        self.numb_freq_index = 0

    class LoadingType(enum.IntEnum):
        short_all = -1
        series_rlc = 0
        parallel_rlc = 1
        series_rlc_per_m = 2
        parallel_rlc_per_m = 3
        impedance = 4
        wire_conductivity = 5

    def import_file(self, file_name):
        """
            Imports a .nec file.
            This also imports it into our abstracted functions.

            Args:
                - file_name (str): The name of the .nex file
        """
        if not os.path.isfile(file_name):
            raise OSError()
        with open(file_name) as f:
            lines = f.readlines()

        rp_card_args = None     # Run the RP card last
        for line in lines:
            line = list(filter(None, [x.strip() for x in line.split(' ')]))
            if line[0] in ['CM', 'CE']:     # If comment, ignore
                continue
            elif line[0] == 'GW':
                self.add_wire([float(x) for x in line[3:6]], [float(x) for x in line[6:9]], float(line[9]), int(line[2]), manual_wire_id=int(line[1]))
            elif line[0] == 'GA':
                self.add_arc(float(line[3]), float(line[4]), float(line[5]), float(line[6]), int(line[2]), manual_arc_id=int(line[1]))
            elif line[0] == 'GE':
                self.nec.geometry_complete(int(line[1]))     # Call the nec class's geometry_complete directly
            elif line[0] == 'EX':
                self.add_exitation(int(line[2]), int(line[3]))
            elif line[0] == 'FR':
                if int(line[2]) == 1:
                    self.set_single_f(float(line[5]))
                else:
                    self.set_multiple_f(float(line[5]), n=int(line[2]), step=float(line[6]))
            elif line[0] == 'RP':
                arg = [float(x) for x in line[1:]]
                for i in [0, 1, 2, 3]:
                    arg[i] = int(arg[i])
                tmp = arg[3]
                arg.pop(3)
                for index, i in enumerate("{:04d}".format(tmp)):
                    arg.insert(3+index, int(i))
                rp_card_args = arg.copy()
            elif line[0] == 'GM':
                # TODO: Move this to main calling function
                self.geo.move(*[float(x) for x in line[3:9]], int(float(line[9])), int(line[2]), int(line[1]))
            elif line[0] == 'LD':
                arg = [float(x) for x in line[1:8]]
                for i in [0, 1, 2, 3]:
                    arg[i] = int(arg[i])
                self.log.debug("Creating an ld cart: {}".format(arg))
                self.nec.ld_card(*arg)
            elif line[0] == 'EN':
                self.log.debug("Reached end of line")
                self.log.debug("Creating an rp cart: {}".format(rp_card_args))
                self.nec.rp_card(*rp_card_args)
            else:
                self.log.warning("Command {} is currently not supported. Please create a bug report on the GitHub repository".format(line[0]))

    def add_wire(self, coords_1: list, coords_2: list, wire_rad: float, numb_segments: int, manual_wire_id: int = None):
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
        if manual_wire_id is None:
            # TODO: This may not be the best way to handle adding a manual ID wire, fix this later
            self._last_tag_id += 1
        else:
            self._last_tag_id = manual_wire_id
        self._all_wires[self._last_tag_id] = {
            'coords': np.array([coords_1, coords_2]),
            'id': self._last_tag_id,
            'width': wire_rad,
            'numb_seg': numb_segments,
        }
        self.geo.wire(self._last_tag_id, numb_segments, *coords_1, *coords_2, wire_rad, 1.0, 1.0)
        return self._last_tag_id

    def add_arc(self, radius: float, start_angle: float, end_angle: float, wire_radius: float, numb_segments: int, manual_arc_id: int = None):
        if manual_arc_id is None:
            # TODO: This may not be the best way to handle adding a manual ID wire, fix this later
            self._last_tag_id += 1
        else:
            self._last_tag_id = manual_arc_id
        self.geo.arc(self._last_tag_id, numb_segments, radius, start_angle, end_angle, wire_radius)
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

    def coordinate_transform(self, rot_x: float = 0, rot_y: float = 0, rot_z: float = 0,
                             trans_x: float = 0, trans_y: float = 0, trans_z: float = 0,
                             start_move_segment: int = 0, tag_increment: int = 0, array_numb: int = 0):
        self.geo.move(rot_x, rot_y, rot_z, trans_x, trans_y, trans_z, start_move_segment, array_numb, tag_increment)

    def add_loading(self, loading_type: LoadingType, wire_id: int, start_seg: int, end_seg: int,
                    resistance: float = None, capacitance: float = None, inductance: float = None,
                    reactance: float = None, conductivity: float = None):
        if reactance is None and loading_type == self.LoadingType.impedance:
            raise UserWarning("Not the right arguments")
        if conductivity is None and loading_type == self.LoadingType.wire_conductivity:
            raise UserWarning("Not the right arguments")

        if loading_type in [self.LoadingType.series_rlc, self.LoadingType.parallel_rlc, self.LoadingType.series_rlc_per_m, self.LoadingType.parallel_rlc_per_m]:
            if capacitance is None or inductance is None or resistance is None:
                raise UserWarning("Not the right arguments")
            self.nec.ld_card(loading_type, wire_id, start_seg, end_seg, resistance, inductance, capacitance)
        elif loading_type == self.LoadingType.impedance:
            if resistance is None or reactance is None:
                raise UserWarning("Not the right arguments")
            self.nec.ld_card(loading_type, wire_id, start_seg, end_seg, resistance, reactance, None)
        elif loading_type == self.LoadingType.wire_conductivity:
            if conductivity is None:
                raise UserWarning("Not the right arguments")
            self.nec.ld_card(loading_type, wire_id, start_seg, end_seg, conductivity, None, None)
        elif loading_type == self.LoadingType.short_all:
            self.nec.ld_card(loading_type, wire_id, start_seg, end_seg, None, None, None)
        else:
            raise UserWarning("Loading type is invalid: {}".format(loading_type))

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

    def set_multiple_f(self, min_f, max_f: float = None, n: int = None, step: float = None):
        """
            Set multiple frequencies for the calculation

            Args:
                min_f: The minimum simulation frequency, in Mhz
                max_f: The maximum simulation frequency, in Mhz. Either this or step are required
                n: The number of frequency steps to simulate for
                step: The frequency step. Required if max_f is not given
        """
        if step is None:
            step = (max_f - min_f)/n
        # No need to calculate for the maximum frequency for now, unless it is to be used later
        # else:
        #     if max_f is not None:
        #         raise UserWarning("The steps are given, so no need for the maximum frequency")
        #     max_f = step*n + min_f
        self.numb_freq_index = n
        self.nec.fr_card(0, n, min_f, step)

    def get_3d_radiation_pattern(self, freq_index: int = 0) -> Radiation3DPatternData:
        """
            Get the radiation pattern for a given frequency index
        """
        ret = Radiation3DPatternData()
        rpt = self.nec.get_radiation_pattern(freq_index)
        ret.freq = rpt.get_frequency()

        gains_db = rpt.get_gain()
        ret.gains = 10.0**(gains_db / 10.0)
        # ret.gains = gains_db
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

    def get_2d_radiation_pattern(self, freq_index: int = 0, elevation: float = None, azimuth: float = None) -> Radiation2DPatternData:
        if elevation is None and azimuth is None:
            raise UserWarning("Must either given an elevation or azimuth")
        if elevation is not None and azimuth is not None:
            raise UserWarning("Can't give both the elevation and azimuth")
        ret = Radiation2DPatternData()
        rpt = self.nec.get_radiation_pattern(freq_index)
        ret.freq = rpt.get_frequency()

        gains_db = rpt.get_gain()
        if elevation is not None:
            all_elevations = rpt.get_theta_angles()
            if elevation not in all_elevations:
                raise UserWarning("Elevation isn't part of the generated elevations. Available are {}".format(all_elevations))
            gains_db = gains_db[np.where(all_elevations == elevation)[0][0], :]
            ret.constant_elevation = elevation
            ret.plot_theta = rpt.get_phi_angles()
        elif azimuth is not None:
            all_azimuth = rpt.get_phi_angles()
            if azimuth not in all_azimuth:
                raise UserWarning("Elevation isn't part of the generated azimuths. Available are {}".format(all_azimuth))
            print(((azimuth+180) % 360), np.where(all_azimuth == ((azimuth+180) % 360)))
            gains_db = np.append(gains_db[:, np.where(all_azimuth == azimuth)[0][0]], gains_db[:, np.where(all_azimuth == ((azimuth+180) % 360))[0][0]])
            ret.constant_azimuth = azimuth
            ret.plot_theta = np.append(rpt.get_theta_angles(), (rpt.get_theta_angles() + 180))
            print(ret.plot_theta)

        gains_db = 10.0**(gains_db / 10.0)
        ret.plot_theta = ret.plot_theta *np.pi / 180
        ret.plot_radius = gains_db

        return ret

    def get_all_freq_3d_radiation_pattern(self) -> list[Radiation3DPatternData]:
        return [self.get_3d_radiation_pattern(i) for i in range(self.numb_freq_index)]

    def get_all_frequencies(self):
        freqs = []
        for i in range(self.numb_freq_index):
            ipt = self.nec.get_input_parameters(i)
            freqs.append(ipt.get_frequency())
        return freqs

    def plot_3d_radiation_pattern(self, in_data: Radiation3DPatternData = None, freq_index: int = 0, show: bool = True) -> Graph3DRadiationPattern:
        """
            Function to plot the 3D radiation pattern of the antenna
        """
        if in_data is None:
            in_data = self.get_3d_radiation_pattern(freq_index)

        plot = Graph3DRadiationPattern(in_data)
        if show:
            plot.show()
        return plot

    def plot_2d_radiation_pattern(self, in_data: Radiation3DPatternData = None, freq_index: int = 0, elevation: float = None, azimuth: float = None):
        if in_data is None:
            in_data = self.get_2d_radiation_pattern(freq_index, elevation, azimuth)

        print(in_data.plot_theta)
        print(in_data.plot_radius)
        plot = Graph2DRadiationPattern(in_data)
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
    anim_r = Graph3DRadiationPattern(dipole_sim.get_3d_radiation_pattern())
    anim_r.show()
