# -*- coding: utf-8 -*-
import logging
import os
import pickle
import re
import sys
import time
from glob import glob
from itertools import product
from subprocess import check_output
from textwrap import dedent, indent

import numpy as np

from ..logging_config import enable_logging
from ..utils import get_qstat_json

__all__ = ("CX1Fit",)

logger = logging.getLogger(__name__)

TMP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "cx1_files")
)
os.makedirs(TMP_DIR, exist_ok=True)


class CX1Fit:
    def __init__(
        self,
        X,
        y,
        model="RandomForestRegressor",
        method="cross_val_score",
        data_name="dummy",
        param_grid=None,
        imports=(
            "from sklearn.ensemble import RandomForestRegressor",
            "from sklearn.model_selection import cross_val_score",
        ),
        n_jobs=8,
    ):
        """Initialise the fitting.

        Args:
            X (array-like): Train X data.
            y (array-like): Train y data.
            model (str): Name of the model class.
            method (str): Validation method which returns scores.
            param_grid (dict): Mapping of model parameter names to possible values.
            data_name (str): A description of the data (`X` and `y`), eg. which
                datasets, variables and time shifts are considered. This is used to
                tell apart two model runs with identical `param_grid` parameters, but
                different training data.
            imports (iterable of str): Imports necessary to use `model` and `method`.

        """
        # Sort inputs so they are always in the same order no matter how they are
        # given.
        self.model = model
        self.method = method
        self.param_grid = dict(sorted(param_grid.items()))
        self.imports = list(imports)
        self.data_name = data_name

        assert any(
            self.model in line for line in self.imports
        ), "Model needs to be imported."
        assert any(
            self.method in line for line in self.imports
        ), "Method needs to be imported."

        self.imports.extend(
            (
                "import pickle",
                "import os",
                "import logging",
                "import numpy as np",
                "from itertools import product",
                "from wildfires.logging_config import enable_logging",
            )
        )

        parameters = (
            self.model,
            self.method,
            self.param_grid,
            self.imports,
            self.data_name,
        )

        # Check the directory for an existing run by comparing parameters.
        self.already_run = None
        parameter_files = glob(
            os.path.join(TMP_DIR, "**", "parameters_RUN*.pickle"), recursive=True
        )
        for parameter_file in parameter_files:
            with open(parameter_file, "rb") as f:
                comp_parameters = pickle.load(f)
            if parameters == comp_parameters:
                self.already_run = "RUN" + re.search(
                    ".*_RUN(\d*)\.pickle", os.path.basename(parameter_file)
                ).group(1)
                self.id = self.already_run
                logger.info(
                    f"Matching parameters found in {self.id} "
                    f"{os.path.dirname(parameter_file)}."
                )
                break
        else:
            # If there is no break - ie if no previous matching run is found.
            # Start a new run.
            dir_contents = os.listdir(TMP_DIR)

            self.id = (
                "RUN"
                + str(
                    int(
                        max(
                            dir_contents,
                            key=lambda x: int(x.lower().replace("run", "")),
                        )
                        .lower()
                        .replace("run", "")
                    )
                    + 1
                )
                if dir_contents
                else "RUN1"
            )

        self.dir = os.path.join(TMP_DIR, str(self.id))
        self.job_output_dir = os.path.join(self.dir, "output")
        self.job_results_dir = os.path.join(self.dir, "results")
        self.data_file = os.path.join(self.dir, "data.npz")
        self.executable_file = os.path.join(self.dir, "fitting_executable.py")
        self.bash_file = os.path.join(self.dir, "fitting_jobs.sh")

        self.n_jobs = n_jobs
        self.n_sets = len(list(product(*list(self.param_grid.values()))))

        self.job_name = f"{type(self).__name__}_{self.id}"

        if self.already_run is None:
            # If an error occurs here, the same parameters have already been explored (and
            # not detected above).
            os.makedirs(self.dir)
            os.makedirs(self.job_output_dir)
            os.makedirs(self.job_results_dir)
            np.savez(self.data_file, X=X, y=y)
            with open(
                os.path.join(self.dir, f"parameters_{self.id}.pickle"), "wb"
            ) as f:
                pickle.dump(parameters, f, -1)
            logger.debug(f"Finished initialising with ID {self.id}.")

    def _generate_python_file(self):
        """Using the given parameters, generate a file sufficient to run the model."""
        # Define necessary imports.
        contents = (
            "\n".join(self.imports)
            + '\n\n\nif __name__ == "__main__":'
            + indent(
                dedent(
                    f"""
            logger = logging.getLogger(__name__)
            enable_logging(level="DEBUG")

            # Load the data.
            loaded_file = np.load("{self.data_file}")
            X = loaded_file["X"]
            y = loaded_file["y"]

            # Retrieve the array job index.
            index = int(os.environ["PBS_ARRAY_INDEX"]) - 1
            logger.debug(
                f"Got index {{index}} from PBS_ARRAY_INDEX value "
                f"{{os.environ['PBS_ARRAY_INDEX']}}."
            )
            # Use the index to retrieve the parameters.
            params = list(product(*{list(self.param_grid.values())}))[index]
            # Get the corresponding names.
            param_names = {list(self.param_grid)}
            # Match these to construct a new dictionary which will then be fed to the
            # model.
            model_kwargs = dict(zip(param_names, params))
            # Create the model.
            model = {self.model}(**model_kwargs, n_jobs={self.n_jobs})
            # Apply the method to the model and data.
            scores = {self.method}(model, X, y, cv=5, n_jobs=1)
            # Store and pickle the results.
            scores_pickle_file = os.path.join(
                "{self.job_results_dir}", f"scores_{{index}}.pickle"
            )
            with open(scores_pickle_file, "wb") as f:
                pickle.dump(scores, f, -1)

            # Refit the model on all the data and store this as well.
            model.fit(X, y)
            model_pickle_file = os.path.join(
                "{self.job_results_dir}", f"model_{{index}}.pickle"
            )
            with open(model_pickle_file, "wb") as f:
                pickle.dump(model, f, -1)

            logger.debug(f"Finished fitting index {{index}}.")
            """
                ),
                " " * 4,
            )
        )

        logger.debug(f"Writing Python file to {self.executable_file}.")
        with open(self.executable_file, "w") as f:
            f.write(contents)

    def _generate_bash_file(self):
        # Path to the python executable to use( ie. the current executable).
        python_executable = sys.executable

        array_job_indices = f"#PBS -J 1-{self.n_sets}"
        if self.n_sets == 1:
            array_job_indices = "export PBS_ARRAY_INDEX=1"

        bash_script = dedent(
            f"""
            #!/usr/bin/env sh
            #PBS -N {self.job_name}
            #PBS -l select=1:ncpus={self.n_jobs}:mem=15gb
            #PBS -l walltime=10:00:00
            {array_job_indices}
            export PYTHONPATH="$PYTHONPATH:/rds/general/user/ahk114/home/Documents/:/rds/general/user/ahk114/home/Documents/masters/:/rds/general/user/ahk114/home/Documents/PhD/:"
            {python_executable} {self.executable_file}
            """
        ).lstrip()
        logger.debug(f"Writing Bash file to {self.executable_file}.")
        with open(self.bash_file, "w") as f:
            f.write(bash_script)

    def run_job(self):
        if self.already_run:
            logger.warning("Trying to run job again.")
            return None
        self._generate_python_file()
        self._generate_bash_file()
        output = check_output(
            [
                "qsub",
                "-V",
                "-o",
                self.job_output_dir,
                "-e",
                self.job_output_dir,
                self.bash_file,
            ]
        ).decode()
        logger.info(f"Submitted job {output}.")

    def job_status(self, array_info=True):
        try:
            out = get_qstat_json()
        except FileNotFoundError:
            logger.error("Not running on hpc.")
            raise

        jobs = out.get("Jobs")
        if jobs:
            for sub_job_name, job_details in jobs.items():
                if self.job_name == job_details["Job_Name"]:
                    status = job_details["job_state"]
                    if array_info:
                        if "array_indices_submitted" in job_details:
                            status += (
                                ", Submitted Indices "
                                f"{job_details['array_indices_submitted']}"
                            )
                        if "array_indices_remaining" in job_details:
                            status += (
                                ", Remaining Indices "
                                f"{job_details['array_indices_remaining']}"
                            )
                    logger.debug(f"Current status: {status}.")
                    return status
            else:
                logger.warning("No matching job found.")
        else:
            logger.warning("No active jobs found.")

    def get_best_model(
        self, timeout=60 * 15, return_model=True, return_score=True, return_params=True
    ):
        """Retrieve the model with the highest mean score.

        If only one of `return_model`, `return_score`, and `return_params` is True,
        only that value will be returned, while otherwise a dictionary containing
        these values will be returned.

        Args:
            timeout (float): Seconds to wait for successful job execution in seconds.
            return_model (bool): If True, return the fitted model.
            return_score (bool): If True, return the mean CV score corresponding to
                the best model parameters.
            return_params (bool): Return the model parameters that were explicitly
                set for this run.

        """
        assert any(
            (return_model, return_score, return_params)
        ), "Must return at least one thing."
        failed_before = False
        interval = 60
        start = time.time()
        while (time.time() - start) < timeout:
            status = self.job_status()

            # Get all score files.
            score_files = glob(
                os.path.join(self.dir, "**", "scores_*.pickle"), recursive=True
            )
            if status is None:
                # If no current matching job is found.

                if score_files:
                    logger.info(
                        "No matching job was found. Scores were found however, so "
                        "assuming job has finished."
                    )
                    break
                else:
                    if failed_before:
                        logger.error("No job and no scores were found.")
                        raise ValueError("No job and no scores were found.")

                    failed_before = True
                    logger.warning(
                        "No matching job was found, and no scores were found. Trying "
                        f"again in {interval} seconds."
                    )
            else:
                logger.info(
                    f"Waiting for job to finish. Current status: {status}. "
                    f"Completed: {len(score_files)}/{self.n_sets}."
                )

            time.sleep(interval)

        if not status is None:
            logger.warning("No results found within the timeout.")
            return None

        # Get all the scores and compute their means to find the score with the
        # highest mean.
        mean_scores = []
        for score_file in score_files:
            with open(score_file, "rb") as f:
                mean_scores.append(np.mean(pickle.load(f)))

        index = int(
            re.search(
                "scores_(\d*)\.pickle", score_files[np.argmax(mean_scores)]
            ).group(1)
        )

        output = {}
        if return_model:
            with open(
                os.path.join(self.job_results_dir, f"model_{index}.pickle"), "rb"
            ) as f:
                output["model"] = pickle.load(f)
        if return_score:
            output["score"] = np.max(mean_scores)
        if return_params:
            output["params"] = dict(
                zip(self.param_grid, list(product(*self.param_grid.values()))[index])
            )

        if len(output) == 1:
            return list(output.values())[0]
        else:
            return output

    def get_cv_scores(self, timeout=60 * 15, return_params=True):
        """Retrieve the different cv scores."""
        output = self.get_best_model(timeout=timeout)
        if output:
            # Get all score files.
            score_files = glob(
                os.path.join(self.dir, "**", "scores_*.pickle"), recursive=True
            )
            scores = []
            params = []
            full_param_grid = list(product(*self.param_grid.values()))
            for score_file in score_files:
                with open(score_file, "rb") as f:
                    scores.append(pickle.load(f))
                index = int(re.search("scores_(\d*)\.pickle", score_file).group(1))
                params.append(dict(zip(self.param_grid, full_param_grid[index])))

            if return_params:
                return scores, params
            else:
                return scores


if __name__ == "__main__":
    enable_logging(level="DEBUG")

    X = np.random.random((10000, 5))
    X[:, 1] += 1e-3 * np.arange(10000)
    y = np.random.random((10000,)) + 5e-4 * np.arange(10000)

    dummy = CX1Fit(X, y, param_grid={"n_estimators": [5, 20, 100]})
    dummy.run_job()
    output = dummy.get_best_model()
