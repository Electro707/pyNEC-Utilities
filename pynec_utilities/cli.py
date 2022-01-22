"""
    PyNEC Utilities CLI
    Copyright (C) 2022 Electro707 (develop@electro707.com)

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
import click
from pynec_utilities import pynecutility


@click.group()
@click.argument('nec_file', type=click.Path(exists=True))
@click.pass_context
def cli(ctx, nec_file):
    """
    A command line utility for PyNEC Ulitities

    nec_file An input .nec file containing the antenna model
    """
    ctx.ensure_object(dict)

    n = pynecutility.PyNECWrapper()
    n.import_file(nec_file)
    n.calculate(36)

    ctx.obj['nec_wrapper'] = n


@cli.command()
@click.pass_context
def plot_3d(ctx):
    """
    Plots the antenna's 3D radiation pattern with MatPlotLib
    """
    n = ctx.obj['nec_wrapper']
    n.plot_3d_radiation_pattern()


if __name__ == "__main__":
    cli()
