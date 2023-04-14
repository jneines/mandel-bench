#!/usr/bin/env python3
import logging
from pathlib import Path
import importlib
import json
import platform
import time
import datetime as dt
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import logging.config

# disable all library loggers
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)


class Mandel(object):
    engines_dir = Path("engines")
    views_dir = Path("views")
    results_dir = Path("results")
    plots_dir = Path("plots")

    def __init__(
        self, engine_name, view_name, resolution, runs, save, save_plot, verbose
    ):
        self.engine_name = engine_name
        self.view_name = view_name
        self.resolution = resolution
        self.runs = runs
        self.save = save
        self.do_save_plot = save_plot
        self.verbose = verbose

        self.init_logging()
        self.init_storage()

    def init_logging(self):
        self.logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "{asctime} - {name} - {levelname}: {message}", style="{"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if self.verbose == 0:
            self.logger.setLevel(logging.CRITICAL)
        elif self.verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif self.verbose >= 2:
            self.logger.setLevel(logging.DEBUG)

    def init_storage(self):
        self.logger.info("Preparing storage")
        for _dir in [
            self.engines_dir,
            self.views_dir,
            self.results_dir,
            self.plots_dir,
        ]:
            _dir.mkdir(exist_ok=True)

    def load_system_info(self):
        self.logger.info("Gathering system information.")

        uname = platform.uname()
        self.hostname = uname.node.split(".")[0]
        self.os = uname.system
        self.machine = uname.machine
        self.python_version = platform.python_version()

    def load_engine(self):
        self.logger.info(f"Loading compute engine: {self.engine_name}")

        module = importlib.import_module(f".{self.engine_name}", str(self.engines_dir))
        self.calculate = module.calculate

    def load_view(self):
        self.logger.info(f"Loading view: {self.view_name}.")

        with (self.views_dir / f"{self.view_name}.json").open("r") as fd:
            self.view = json.loads(fd.read())
        self.x_min = self.view["x_min"]
        self.x_max = self.view["x_max"]
        self.y_min = self.view["y_min"]
        self.y_max = self.view["y_max"]
        self.max_iterations = self.view["max_iterations"]

    def show_results(self):
        self.logger.info("Showing results.")

        print(json.dumps(self.results, indent=4, default=str))

    def save_results(self):
        self.logger.info("Saving results.")

        run_index = self.results["run_index"]
        now = self.results["now"]
        now_str = f"{now:%Y%m%d-%H%M%S}"
        results_filename = f"{self.engine_name}-{run_index:03d}-{now_str}.json"
        results_path = (
            self.results_dir / self.view_name / self.hostname / results_filename
        )
        results_dir = results_path.parent
        results_dir.mkdir(parents=True, exist_ok=True)

        with results_path.open("w") as fd:
            fd.write(json.dumps(self.results, indent=4, default=str))

    def save_plot(self):
        self.logger.info("Saving a plot of the computation results.")
        plot_filename = f"{self.hostname}-{self.engine_name}.png"
        plot_path = self.plots_dir / self.view_name / plot_filename
        plot_dir = plot_path.parent
        plot_dir.mkdir(parents=True, exist_ok=True)

        aspect_ratio = (self.x_max - self.x_min) / (self.y_max - self.y_min)
        width = 10
        height = int(width / aspect_ratio)
        dpi = 100
        self.logger.debug(f"{aspect_ratio=}, {width=}, {height=}, {dpi=}")
        fig, axes = plt.subplots(figsize=(width, height), dpi=dpi)
        axes.imshow(
            self.iterations[::-1],
            extent=(self.x_min, self.x_max, self.y_min, self.y_max),
            cmap="RdGy",
        )
        plt.tight_layout()
        plt.savefig(plot_path)

    def run(self):
        self.logger.info("Entering run.")
        self.load_system_info()
        self.load_engine()
        self.load_view()
        for run_index in range(self.runs):

            self.results = {
                "now": dt.datetime.now(),
                "hostname": self.hostname,
                "os": self.os,
                "machine": self.machine,
                "python_version": self.python_version,
                "view": self.view_name,
                "engine": self.engine_name,
                "run_index": run_index + 1,
            }

            tic = time.perf_counter()
            iterations, details = self.calculate(
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.max_iterations,
                self.resolution,
            )
            toc = time.perf_counter()
            self.iterations = np.array(iterations)
            self.logger.debug(f"Shape of iterations array: {self.iterations.shape}.")

            self.results["calculation_time"] = toc - tic
            self.results["details"] = details

            self.show_results()
            if self.save:
                self.save_results()
        if self.do_save_plot:
            self.save_plot()


if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    def list_engines():
        engines = [match.stem for match in Mandel.engines_dir.glob("*.py")]
        print("Available engines:")
        for engine in engines:
            print(f"  - {engine}")

    def list_views():
        views_dir = Path("views")
        views = [match.stem for match in Mandel.views_dir.glob("*.json")]
        print("Available views:")
        for view in views:
            print(f"  - {view}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--engine", type=str, help="Engine to use", default="naive"
    )
    parser.add_argument(
        "-V", "--view", type=str, help="View to load for computation", default="base"
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        help="Resolution to use for computation",
        default=1000,
    )
    parser.add_argument(
        "-R", "--runs", type=int, help="How often to repeat the calculation", default=1
    )
    parser.add_argument(
        "-s", "--save", action="store_true", help="Save the results", default=False
    )
    parser.add_argument(
        "-p",
        "--save-plot",
        action="store_true",
        help="Save plot to file",
        default=False,
    )
    parser.add_argument("-v", "--verbose", action="count", help="Be verbose", default=0)
    parser.add_argument(
        "-le", "--list-engines", action="store_true", help="List available engines"
    )
    parser.add_argument(
        "-lv", "--list-views", action="store_true", help="List available views"
    )

    args = parser.parse_args()
    if args.list_engines:
        list_engines()
    elif args.list_views:
        list_views()
    else:

        app = Mandel(
            args.engine,
            args.view,
            args.resolution,
            args.runs,
            args.save,
            args.save_plot,
            args.verbose,
        )
        app.run()
