# -*- coding: utf-8 -*-
import sys
from functools import wraps
from inspect import signature

from dask_jobqueue.pbs import PBSCluster, PBSJob

from .ports import get_ports


class CX1PBSJob(PBSJob):
    @wraps(PBSJob.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Modify the '_command_template' in order to specify a local scheduler port,
        # will usually be different from the original scheduler port. Within the job
        # script, this will be forwarded to the proper scheduler port using local port
        # forwarding.
        command_template = self._command_template.split()
        command_template[3] = "localhost:$LOCALSCHEDULERPORT"
        self._command_template = " ".join(command_template)


class CX1Cluster(PBSCluster):
    @wraps(PBSCluster.__init__)
    def __init__(self, *args, **kwargs):
        # First place any args into kwargs, then update relevant entries in kwargs to
        # enable operation on CX1.
        bound_args = signature(super().__init__).bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        mod_kwargs = bound_args.arguments
        # Use kwargs as well (otherwise the whole kwargs dictionary would be used
        # as another 'kwargs' keyword argument instead of the arguments therein).
        mod_kwargs.update(mod_kwargs.pop("kwargs"))

        scheduler_port = get_ports()[0]
        mod_kwargs.update(
            job_cls=CX1PBSJob,
            extra=list(mod_kwargs.get("extra", []))
            + "--worker-port $WORKERPORT".split(),
            # NOTE: Simple ssh, NOT autossh is used below, since using autossh
            # resulted in the connection being dropped repeatedly as it was
            # overzealously restarted.
            env_extra=(
                f"PYTHON={sys.executable}",
                f"SCHEDULERADDRESS='localhost:{scheduler_port}'",
                "echo 'Getting ports'",
                "read LOCALSCHEDULERPORT <<< $(/rds/general/user/ahk114/home/.pyenv/versions/wildfires/bin/valid-ports 1)",
                'echo "local scheduler port:$LOCALSCHEDULERPORT"',
                # For the worker port, extra care needs to be taken since this port
                # needs to be the same on the worker and scheduler node. So we need to
                # check that the port is unused on both!
                "WORKERPORT=$($PYTHON << EOF",
                "from subprocess import call",
                "from wildfires.ports import get_ports",
                "port_found = False",
                "cycles = 0",
                # Check if the local port would also work on the remote, ie. scheduler
                # node. Check for return code 1 which rules out some ssh error codes.
                "while True:",
                "   port = get_ports()[0]",
                '   if call(["ssh", "-o BatchMode=yes", "login-7", f"nc -z localhost {port}"]) == 1:',
                "       print(port)",
                "       break",
                "   cycles += 1",
                "   if cycles > 20:",
                '       raise RuntimeError("No valid port could be found.")',
                "EOF",
                ")",
                "PORTCODE=$?",
                "if [ $PORTCODE -eq 0 ]",
                "then",
                'echo "worker port:$WORKERPORT"',
                "else",
                'echo "Could not get worker port." >&2',
                "exit $PORTCODE",
                "fi",
                "echo 'Running ssh'",
                "ssh -vvv -N -T -L localhost:$LOCALSCHEDULERPORT:$SCHEDULERADDRESS -R localhost:$WORKERPORT:localhost:$WORKERPORT login-7 -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=120 -o ServerAliveCountMax=6 &",
                "echo 'Finished running ssh, starting dask-worker now'",
                "sleep 1",
                "echo 'Local processes:'",
                "pgrep -afu ahk114",
            )
            + tuple(mod_kwargs.get("env_extra", ())),
            scheduler_options=dict(
                mod_kwargs.get("scheduler_options", ())
                if mod_kwargs.get("scheduler_options", ()) is not None
                else (),
                port=scheduler_port,
            ),
        ),
        super().__init__(**mod_kwargs)
