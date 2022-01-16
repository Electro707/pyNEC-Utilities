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
