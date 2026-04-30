"""dct monitor — launch the graphical DCT window."""
import click


@click.command()
def monitor():
    """Launch the DCT graphical monitor window."""
    from dct.gui.app import run
    run()
