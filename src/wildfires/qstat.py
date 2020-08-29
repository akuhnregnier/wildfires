# -*- coding: utf-8 -*-
"""Collection of qstat related code used throughout the project.

"""
import logging
import os
import platform
import socket
from subprocess import CalledProcessError

logger = logging.getLogger(__name__)


def get_qstat_json():
    """Retrieve the json representation of the qstat output.

    Raises:
        FileNotFoundError: If the command is not run on the hpc.

    """
    raise FileNotFoundError("Do not use qstat to prevent overloading pbs.")

    # raw_output = check_output(("qstat", "-f", "-F", "json"), stderr=DEVNULL).decode()
    # # Filter out invalid json (unescaped double quotes).
    # filtered_lines = [line for line in raw_output.split("\n") if '"""' not in line]
    # filtered_output = "\n".join(filtered_lines)
    # return json.loads(filtered_output)


def get_qstat_ncpus():
    """Get ncpus from qstat job details.

    Only relevant if we are currently running on a node with a hostname matching one
    of the running jobs.

    """
    pbs_jobid = os.environ.get("PBS_JOBID")
    if pbs_jobid is None:
        logger.debug("Not running in a PBS job on hpc.")
        return None

    try:
        out = get_qstat_json()
    except FileNotFoundError:
        logger.warning("Not running on hpc.")
        return None
    except CalledProcessError as process_error:
        logger.warning(
            f"Call to qstat failed with returncode {process_error.returncode}."
        )
        return None
    jobs = out.get("Jobs")
    if jobs:
        current_hostname = platform.node()
        if not current_hostname:
            current_hostname = socket.gethostname()
        if not current_hostname:
            logger.error("Hostname could not be determined.")
            return None

        if pbs_jobid:
            # Try to retrieve number of cpus from our job id.
            partial_jobid = pbs_jobid.split(".")[0].split("[")[0]
            assert (
                sum(partial_jobid in jobid for jobid in jobs) <= 1
            ), "Two jobs should never share the same id number."
            if not any(partial_jobid in jobid for jobid in jobs):
                logger.warning(
                    f"PBS job id '{pbs_jobid}' was not found in list of jobs."
                )
            else:
                pbs_jobid = [jobid for jobid in jobs if partial_jobid in jobid][0]
                job = jobs[pbs_jobid]
                if "exec_host" in job:
                    # Compare hostname recorded in the job.
                    exec_host = job["exec_host"].split("/")[0]
                    if exec_host not in current_hostname:
                        logger.warning(
                            f"Our hostname '{current_hostname}' and our job hostname "
                            f"'{exec_host}' do not match."
                        )
                else:
                    logger.warning(f"exec_host not found for job '{pbs_jobid}'.")
                ncpus = int(job["Resource_List"]["ncpus"])
                logger.info("Got ncpus: {} from job '{}'.".format(ncpus, pbs_jobid))
                return ncpus
        else:
            logger.warning("PBS_JOBID environment variable was not found.")

        # Loop through each job.
        for jobid, job in jobs.items():
            # 'B' only applies to array jobs and indicates that the job has begun.
            if not job["job_state"] in ("R", "B"):
                logger.debug(
                    f"Ignoring job '{job['Job_Name']}' in state '{job['job_state']}' "
                    "as it is not running."
                )
                continue

            # If we are on the same machine.
            if "exec_host" not in job:
                # Skip this job if it has no 'exec_host' attribute to compare against.
                logger.warning(f"exec_host not found for job '{jobid}'.")
                continue
            exec_host = job["exec_host"].split("/")[0]
            logger.debug(
                f"Comparing hostname '{current_hostname}' to job exec host "
                f"'{exec_host}'."
            )
            if exec_host in current_hostname:
                # Other keys include 'mem' (eg. '32gb'), 'mpiprocs'
                # and 'walltime' (eg. '08:00:00').
                resources = job["Resource_List"]
                ncpus = resources["ncpus"]
                logger.info("Got ncpus: {} from job '{}'.".format(ncpus, jobid))
                return int(ncpus)
    else:
        logger.debug("No running jobs were found.")
    return None


def get_ncpus(default=1):
    # The NCPUS environment variable is not always set up correctly, so check for
    # batch jobs matching the current hostname first.
    ncpus = get_qstat_ncpus()
    if ncpus:
        return ncpus
    ncpus = os.environ.get("NCPUS")
    if ncpus:
        logger.info("Read ncpus: {} from NCPUS environment variable.".format(ncpus))
        return int(ncpus)
    logger.warning(
        "Could not read NCPUS environment variable. Using default: {}.".format(default)
    )
    return default
