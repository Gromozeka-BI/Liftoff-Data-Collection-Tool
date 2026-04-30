"""DCT command-line interface."""
import click
from dct.commands.record import record
from dct.commands.inspect import inspect_cmd
from dct.commands.list_cmd import list_cmd
from dct.commands.replay import replay
from dct.commands.validate import validate
from dct.commands.align_cmd import align_cmd
from dct.commands.monitor import monitor


@click.group()
def cli():
    """Data Collection Toolkit for FPV Drone Localization System."""


cli.add_command(record)
cli.add_command(inspect_cmd, name="inspect")
cli.add_command(list_cmd, name="list")
cli.add_command(replay)
cli.add_command(validate)
cli.add_command(align_cmd, name="align")
cli.add_command(monitor)
