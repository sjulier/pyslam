"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import csv
import re
import numpy as np
import math

import os
from datetime import datetime
import shutil
import subprocess
import sys
import glob
import yaml
import json
import concurrent.futures

from pyslam.config import load_settings_yaml
from pyslam.utilities.data_management import merge_dicts
from pyslam.utilities.logging import Printer
from pyslam.utilities.run_command import run_command_async, run_command_sync
import traceback
from pathlib import Path

try:
    import hjson
except ImportError:
    print(
        "hjson not installed. Please install it with: pip install hjson"
    )  # why hjson? because it allows comments in the json file and it does not complain about the trailing commas


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kSettingsFolder = kRootFolder + "/settings"
kResultsFolder = kRootFolder + "/results"
kEvaluationFolder = kRootFolder + "/pyslam/evaluation"


date_time_now = datetime.now()
date_time_now_string = date_time_now.strftime("%Y_%m_%d-%H_%M_%S")
eval_path_prefix = kResultsFolder + "/eval_" + date_time_now_string

log_folder_name = "logs"
template_entry_regex = re.compile(r"%(\w+)%")
kTemplateEntryNotFound = "Not found"


def get_git_short_hash(repo_dir=None):
    try:
        repo_dir = repo_dir or Path(__file__).resolve().parent
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return None


def replace_template_entries(string, vocabulary):
    try:
        # Step 1: Replace known entries
        replaced_string = re.sub(
            template_entry_regex,
            lambda match: vocabulary.get(match.group(1), match.group(0)),
            string,
        )
        # Step 2: Remove all *unmatched* placeholders
        cleaned_string = re.sub(template_entry_regex, kTemplateEntryNotFound, replaced_string)
    except Exception as e:
        print("Error in replace_template_entries: " + str(e))
        print(traceback.format_exc())
    return cleaned_string


def remove_unused_datasets(yaml_dict):
    # Step 1: Get selected dataset type (e.g., "KITTI_DATASET")
    selected_dataset = yaml_dict.get("DATASET", {}).get("type", None)

    if not selected_dataset:
        raise ValueError("No dataset type specified in DATASET.type")

    # Step 2: Remove all other *_DATASET entries
    keys_to_delete = [
        key for key in yaml_dict if key.endswith("_DATASET") and key != selected_dataset
    ]

    for key in keys_to_delete:
        del yaml_dict[key]

    return yaml_dict


def update_yaml_dict(yaml_dict, vocabulary):
    # the values of the vocabulary (a preset) replace those of the yaml file (e.g. a preset with
    # another FeatureTrackerConfig.nFeatures than the settings file of the dataset)
    for key, value in vocabulary.items():
        if isinstance(value, dict) and isinstance(yaml_dict.get(key), dict):
            update_yaml_dict(yaml_dict[key], value)
        else:
            yaml_dict[key] = value
    return yaml_dict


def remove_not_found_entries(yaml_dict):
    # VIDEO_DATASET, FOLDER_DATASET: the timestamps, the ground truth and the fps are optional. A
    # template entry without a value would otherwise be read as a file called "Not found".
    dataset_section = yaml_dict.get(yaml_dict.get("DATASET", {}).get("type"), {})
    if dataset_section.get("type") in ("video", "folder"):
        for key in [k for k, v in dataset_section.items() if v == kTemplateEntryNotFound]:
            del dataset_section[key]
    return yaml_dict


def get_preset_custom_parameters(preset):
    # the custom parameters of a preset: "custom_parameters": {"settings": ..., "config": ...}, or
    # "settings" and "config" directly in the preset (as in evaluation_tum.json)
    if "custom_parameters" in preset:
        return preset["custom_parameters"]
    return {key: preset[key] for key in ("settings", "config") if key in preset} or None


def resolve_path(path):
    # the paths of an evaluation configuration are relative to the root folder of pySLAM
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(kRootFolder, path))


def failure_reason(log_path):
    # the first line of the log of a failed run that looks like an error (the last lines are often
    # about the shutdown), otherwise its last line
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = [re.sub(r"\x1b\[[0-9;]*m", "", line).strip() for line in f]
    except OSError:
        return ""
    lines = [line for line in lines if line]
    error_regex = re.compile(r"\w*(Error|Exception):|ERROR|is missing|not installed|not built")
    for line in lines:
        if error_regex.search(line):
            return line[:300]
    return lines[-1][:300] if lines else ""


# A class to manage the evaluation of SLAM algorithms.
# It runs a set of SLAM presets on a set of datasets, and then collects and aggregates the results.
# A final table is created with the results.
# See test/evaluation/test_run_evaluations.py
class SlamEvaluationManager:
    def __init__(
        self,
        evaluation_config_path,
        template_config_file_path,
        just_create_report=False,
        config_overrides=None,
    ):
        if not os.path.exists(evaluation_config_path):
            raise Exception(f"Config file {evaluation_config_path} does not exist")
        self.evaluation_config_path = evaluation_config_path
        print("Loading JSON config file: " + self.evaluation_config_path)
        with open(self.evaluation_config_path, "r") as f:
            json_data = hjson.load(f)
        if config_overrides:  # e.g. from the command line of main_slam_evaluation.py
            json_data.update(config_overrides)
        self.json_data = json_data
        print("JSON evaluation config: ", json.dumps(json_data, indent=4))

        # read the template config file
        print(f"Template config file path: {template_config_file_path}")
        if not os.path.exists(template_config_file_path):
            raise Exception(f"Template config file {template_config_file_path} does not exist")
        with open(template_config_file_path, "r") as f:
            self.template_config = f.read()

        self.presets = []
        self.datasets = []
        self.metrics = {}  # metrics[preset][dataset][iteration][metric_name] -> metric_value

        self.output_path = os.path.abspath(os.path.join(eval_path_prefix, json_data["output_path"]))
        if not just_create_report:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

        self.dataset_base_path = resolve_path(json_data["dataset_base_path"])
        self.dataset_type = json_data["dataset_type"]
        self.sensor_type = json_data["sensor_type"]
        self.number_of_runs_per_dataset = json_data["number_of_runs_per_dataset"]
        self.preset_common_parameters = (
            json_data["common_parameters"] if "common_parameters" in json_data else {}
        )
        self.saved_trajectory_format_type = json_data["saved_trajectory_format_type"]

        if not just_create_report:
            if config_overrides:  # save the configuration that is run, not the file it started from
                config_copy_path = os.path.join(
                    self.output_path, os.path.basename(self.evaluation_config_path)
                )
                with open(config_copy_path, "w") as f:
                    json.dump(json_data, f, indent=4)
            else:
                shutil.copy(self.evaluation_config_path, self.output_path)

        self.presets = [preset for preset in json_data["presets"]]
        self.datasets = [dataset for dataset in json_data["datasets"]]

    def dataset_base_path_of(self, dataset):
        # a dataset can have a base path of its own
        return resolve_path(dataset["base_path"]) if "base_path" in dataset else self.dataset_base_path

    def check_inputs(self):
        """Check that the datasets and their settings files exist before any run is started."""
        problems = []
        for dataset in self.datasets:
            name, base_path = dataset["name"], self.dataset_base_path_of(dataset)
            if self.dataset_type == "FOLDER_DATASET":
                pattern = dataset.get("pattern", "*.png")
                if not os.path.isdir(base_path):
                    problems.append(f"{name}: the image folder {base_path} does not exist")
                elif not glob.glob(os.path.join(base_path, pattern)):
                    problems.append(f"{name}: no images {pattern} in {base_path}")
            elif not os.path.exists(os.path.join(base_path, str(name))):
                problems.append(f"{name}: {os.path.join(base_path, str(name))} does not exist")
            if self.dataset_type == "TUM_DATASET":
                sequence_path = os.path.join(base_path, str(name))
                associations = dataset.get("associations", "associations.txt")
                groundtruth_file = dataset.get("groundtruth_file", "groundtruth.txt")
                if os.path.isdir(sequence_path):
                    if not os.path.exists(os.path.join(sequence_path, associations)):
                        problems.append(
                            f"{name}: no {associations} in {sequence_path}. It lists the frames: create "
                            f"it with scripts/associate.py rgb.txt depth.txt > associations.txt (RGB-D), "
                            f'or use rgb.txt for a monocular sequence ("associations": "rgb.txt")'
                        )
                    if not os.path.exists(os.path.join(sequence_path, groundtruth_file)):
                        problems.append(
                            f"{name}: no {groundtruth_file} in {sequence_path} (TUM sequences need "
                            f"their ground truth: timestamp tx ty tz qx qy qz qw)"
                        )
            for key in ("timestamps", "groundtruth_file"):
                if self.dataset_type in ("VIDEO_DATASET", "FOLDER_DATASET") and key in dataset:
                    if not os.path.exists(os.path.join(base_path, dataset[key])):
                        problems.append(
                            f"{name}: the {key} file {os.path.join(base_path, dataset[key])} does not exist"
                        )
            for key in ("settings_path", "settings_stereo_path"):
                if key in dataset and not os.path.exists(resolve_path(dataset[key])):
                    problems.append(
                        f"{name}: the camera settings file {resolve_path(dataset[key])} does not exist"
                    )
        if not problems:
            return
        Printer.red(f"The evaluation cannot start: {len(problems)} input(s) not found.")
        for problem in problems:
            Printer.red(f"  - {problem}")
        print(f"The datasets are set in {self.evaluation_config_path}:")
        print(f'  "dataset_base_path" ({self.dataset_base_path}) is the folder that contains them.')
        print("  Download them (see scripts/download_tum.sh, scripts/download_euroc.sh and the")
        print('  "Datasets" section of README.md) and set "dataset_base_path" to their folder, or')
        print("  evaluate a video or an image folder of your own: main_slam_evaluation.py --help")
        shutil.rmtree(self.output_path, ignore_errors=True)  # nothing was run: leave no results folder
        if os.path.isdir(eval_path_prefix) and not os.listdir(eval_path_prefix):
            os.rmdir(eval_path_prefix)
        sys.exit(1)

    def get_main_slam_command(
        self,
        dataset,
        preset,
        iteration_idx,
        dataset_base_path,
        preset_common_parameters,
        output_path,
    ):
        print(
            "------------------------------------------------------------------------------------"
        )
        dataset_name = dataset["name"]
        preset_name = preset["name"]

        preset_output_path = os.path.abspath(os.path.join(output_path, preset_name))
        dataset_output_path = os.path.abspath(os.path.join(preset_output_path, dataset_name))
        iteration_output_path = os.path.abspath(
            os.path.join(dataset_output_path, "iteration_" + str(iteration_idx))
        )

        if not os.path.exists(preset_output_path):
            os.makedirs(preset_output_path)
        if not os.path.exists(dataset_output_path):
            os.makedirs(dataset_output_path)
        if not os.path.exists(iteration_output_path):
            os.makedirs(iteration_output_path)

        log_output_path = os.path.abspath(os.path.join(iteration_output_path, log_folder_name))
        if not os.path.exists(log_output_path):
            os.makedirs(log_output_path)
        main_log_output_file = os.path.join(log_output_path, "log.txt")

        # ------
        # Read the preset parameters and merge common with custom ones
        preset_custom_parameters = get_preset_custom_parameters(preset)

        if preset_custom_parameters:
            preset_parameters = merge_dicts(preset_common_parameters, preset_custom_parameters)
        else:
            preset_parameters = preset_common_parameters

        # ------
        # Prepare the settings file and update it with the preset parameters
        dataset_settings_path = resolve_path(dataset["settings_path"])
        if not os.path.exists(dataset_settings_path):
            raise Exception(f"Dataset settings file {dataset_settings_path} does not exist")
        dataset_settings = load_settings_yaml(dataset_settings_path)
        preset_dataset_settings = (
            preset_parameters["settings"] if "settings" in preset_parameters else None
        )
        if preset_dataset_settings:
            dataset_settings = update_yaml_dict(dataset_settings, preset_dataset_settings)
        current_settings_file_path = os.path.join(iteration_output_path, "settings.yaml")
        with open(current_settings_file_path, "w") as f:
            yaml.dump(dataset_settings, f)

        # ------
        # Prepare the stereo settings file and update it with the preset parameters
        if "settings_stereo_path" in dataset:
            dataset_settings_stereo_path = resolve_path(dataset["settings_stereo_path"])
            if not os.path.exists(dataset_settings_stereo_path):
                raise Exception(
                    f"Dataset settings stereo file {dataset_settings_stereo_path} does not exist"
                )
            dataset_settings_stereo = load_settings_yaml(dataset_settings_stereo_path)
            if preset_dataset_settings:
                dataset_settings_stereo = update_yaml_dict(
                    dataset_settings_stereo, preset_dataset_settings
                )
            current_settings_stereo_file_path = os.path.join(
                iteration_output_path, "settings_stereo.yaml"
            )
            with open(current_settings_stereo_file_path, "w") as f:
                yaml.dump(dataset_settings_stereo, f)

        # ------
        # Prepare the config file and update it with the preset parameters
        # Replace the template entries in the config file.
        config_vocabulary = {
            "dataset_type": self.dataset_type,
            "sensor_type": self.sensor_type,
            "output_path": iteration_output_path,
            "dataset_base_path": self.dataset_base_path_of(dataset),
            "dataset_name": f'"{dataset_name}"',  # wrap in quotes to avoid issues with names that contain only numbers
            "settings_path": current_settings_file_path,
            "saved_trajectory_format_type": self.saved_trajectory_format_type,
            "logs_path": log_output_path,
        }
        if "settings_stereo_path" in dataset:
            config_vocabulary["settings_stereo_path"] = current_settings_stereo_file_path
        if "is_color" in dataset:
            config_vocabulary["is_color"] = str(dataset["is_color"])
        if "start_frame_id" in dataset:
            config_vocabulary["start_frame_id"] = str(dataset["start_frame_id"])
        else:
            config_vocabulary["start_frame_id"] = "0"
        # VIDEO_DATASET and FOLDER_DATASET
        if "timestamps" in dataset:
            config_vocabulary["timestamps_path"] = dataset["timestamps"]
        if "groundtruth_file" in dataset:
            config_vocabulary["groundtruth_file"] = dataset["groundtruth_file"]
        elif self.dataset_type == "TUM_DATASET":
            config_vocabulary["groundtruth_file"] = "auto"
        # TUM_DATASET: the list of the frames (timestamp and file of each image, and of its depth image)
        config_vocabulary["associations"] = dataset.get("associations", "associations.txt")
        if "fps" in dataset:
            config_vocabulary["dataset_fps"] = str(dataset["fps"])
        # quoted: a glob pattern such as *png is not valid YAML
        config_vocabulary["dataset_images_glob_pattern"] = f'"{dataset.get("pattern", "*.png")}"'

        current_config_file_str = replace_template_entries(self.template_config, config_vocabulary)
        # print(f"current_config_file_str: {current_config_file_str}")

        # Load as dictionary
        current_config_file = yaml.safe_load(current_config_file_str)
        # Clean up unused dataset sections
        current_config_file = remove_unused_datasets(current_config_file)
        current_config_file = remove_not_found_entries(current_config_file)
        # print(f'current_config_file: {current_config_file}')

        # Save the config file to be used
        current_config_file = yaml.dump(
            current_config_file, sort_keys=False
        )  # Get it back as a string
        current_config_file_path = os.path.join(iteration_output_path, "config.yaml")
        with open(current_config_file_path, "w") as f:
            f.write(current_config_file)

        Printer.bold_blue(
            f"\nRunning evaluation for preset: {preset_name}, dataset: {dataset_name}, iteration: {iteration_idx}"
        )
        Printer.bold(f"\nconfig file:")
        print(current_config_file)
        Printer.bold(f"preset parameters:")
        print(json.dumps(preset_parameters, indent=4))
        Printer.bold(f"output_path: {iteration_output_path}")
        # Printer.bold(f"\t current_config_file_path: {current_config_file_path}")
        if preset_custom_parameters:
            Printer.bold(f"custom_parameters:")
            print(f"{json.dumps(preset_custom_parameters, indent=4)}")
        # if preset_dataset_settings:
        #   Printer.bold(f"preset_dataset_settings: {json.dumps(preset_dataset_settings, indent=4)}")
        Printer.bold(f"log_output_path: {log_output_path}")
        Printer.yellow(f"To see the progress run: $ tail -f {main_log_output_file}")

        # with the python of this process (and so its environment: pixi, conda or venv); -O as in
        # the first line of main_slam.py
        main_slam_path = os.path.normpath(os.path.join(kRootFolder, "main_slam.py"))
        command = (
            f'"{sys.executable}" -O "{main_slam_path}"'
            f' --headless --no_output_date -c "{current_config_file_path}"'
        )

        print("command: " + command)

        print("")
        return command, main_log_output_file

    def run_evaluation(self):
        self.check_inputs()
        commands = []
        for preset in self.json_data["presets"]:
            for dataset in self.json_data["datasets"]:
                for iteration_idx in range(self.number_of_runs_per_dataset):
                    item_run_command, item_log_output_path = self.get_main_slam_command(
                        dataset,
                        preset,
                        iteration_idx,
                        self.dataset_base_path,
                        self.preset_common_parameters,
                        self.output_path,
                    )
                    commands.append((item_run_command, item_log_output_path))
        print(
            "------------------------------------------------------------------------------------"
        )
        # Run commands in parallel
        num_threads = self.json_data["num_threads"]
        futures = []
        results = []
        future_to_command = {}
        future_to_msg = {}  # map futures to (command, label)
        future_to_log = {}

        num_tasks = len(commands)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks to the thread pool
            Printer.bold(
                f"Submitting {len(commands)} tasks to the thread pool with {num_threads} threads..."
            )
            num_started_tasks = 1
            for item_run_command, item_log_output_path in commands:
                msg = f"{num_started_tasks}/{num_tasks}"
                item_future = executor.submit(
                    run_command_async, item_run_command, item_log_output_path, True, msg
                )
                futures.append(item_future)
                future_to_command[item_future] = item_run_command
                future_to_msg[item_future] = msg
                future_to_log[item_future] = item_log_output_path
                num_started_tasks += 1
            # Printer.bold(f"\nSubmitted {num_tasks} tasks to the thread pool with {num_threads} threads.\n")
            # Wait for all tasks to complete
            num_completed_tasks = 0
            for future in concurrent.futures.as_completed(futures):
                num_completed_tasks += 1
                item_run_command = future_to_command[future]
                results.append(future.result())
                (returncode, stdout, stderr) = future.result()
                msg = future_to_msg[future]
                if returncode == 0:
                    Printer.green(
                        f'Command {msg} "{item_run_command}" returned {returncode}, (completed {num_completed_tasks}/{num_tasks})'
                    )
                else:
                    Printer.red(
                        f'Command {msg} "{item_run_command}" returned {returncode}, (completed {num_completed_tasks}/{num_tasks})'
                    )
                    Printer.red(
                        f"  FAILED: {failure_reason(future_to_log[future])} (log: {future_to_log[future]})"
                    )
        print(
            f"------------------------------------------------------------------------------------"
        )
        print(f"{num_completed_tasks}/{num_tasks} tasks completed")

    def collect_metrics(
        self, output_folder, preset_name, dataset_name, iteration_idx, metrics_vocabulary
    ):
        preset_results_path = os.path.join(output_folder, preset_name)
        dataset_results_path = os.path.join(preset_results_path, dataset_name)
        iteration_results_path = os.path.join(
            dataset_results_path, "iteration_" + str(iteration_idx)
        )

        metrics_folder_path = os.path.join(iteration_results_path, "plot")
        iteration_metrics_path = os.path.join(metrics_folder_path, "stats_final.json")

        other_metrics_info_path = os.path.join(iteration_results_path, "other_metrics_info.txt")
        if not os.path.exists(other_metrics_info_path):
            Printer.red(f"{dataset_name}: the file {other_metrics_info_path} is missing")
            return

        Printer.bold(f"preset: {preset_name}, dataset: {dataset_name}, iteration: {iteration_idx}")
        iteration_metrics = (
            metrics_vocabulary.setdefault(preset_name, {})
            .setdefault(dataset_name, {})
            .setdefault(iteration_idx, {})
        )

        # Open the text file in read mode and extract data.
        with open(other_metrics_info_path, "r") as f:
            for line in f.read().split("\n"):
                if line == "":
                    continue
                key, value = line.split(":", 1)
                print(f"\t {key}: {value}")
                try:
                    iteration_metrics[key] = float(value.strip())
                except ValueError:
                    pass  # not a number (e.g. the playback speeds, "1.00x")

        # The trajectory errors: only for a dataset with ground truth
        if not os.path.exists(iteration_metrics_path):
            Printer.yellow(
                f"{dataset_name}: no trajectory error ({iteration_metrics_path} is missing: "
                f"a dataset without ground truth, or the trajectory could not be evaluated)"
            )
            return
        with open(iteration_metrics_path, "r") as f:
            metrics_json_data = json.load(f)
            for key, value in metrics_json_data.items():
                print(f"\t {key}: {value}")
                iteration_metrics[key] = float(value)

    def write_comparative_table(
        self,
        output_path,
        metric_name,
        presets,
        datasets,
        number_of_runs_per_dataset,
        metrics,
        precision=5,
    ):
        # Before generating the table, let's check if the provided metric is available
        is_metric_available = False
        for preset in presets:
            preset_name = preset["name"]
            if preset_name in metrics:
                for dataset in datasets:
                    dataset_name = dataset["name"]
                    if dataset_name in metrics[preset_name]:
                        for iteration_idx in range(number_of_runs_per_dataset):
                            if metric_name in metrics[preset_name][dataset_name].get(iteration_idx, {}):
                                is_metric_available = True
                                break
        if not is_metric_available:
            Printer.yellow(
                f'Metric name "{metric_name}" is not available. Skipping table generation for this metric name.'
            )
            return {}, None

        # Generate the table
        output_table_path = output_path + "/table_" + metric_name.replace(" ", "") + ".csv"
        csv_file = open(output_table_path, "w")
        writer = csv.writer(csv_file, delimiter=",")
        first_row = ["Dataset"]
        for preset in presets:
            first_row.append(preset["name"])
        writer.writerow(first_row)
        for dataset in datasets:
            dataset_name = dataset["name"]
            row = [dataset_name]
            for preset in presets:
                preset_name = preset["name"]
                if preset_name in metrics and dataset_name in metrics[preset_name]:
                    it_values = []
                    for iteration_idx in range(number_of_runs_per_dataset):
                        if metric_name in metrics[preset_name][dataset_name].get(iteration_idx, {}):
                            it_values.append(
                                metrics[preset_name][dataset_name][iteration_idx][metric_name]
                            )
                    if len(it_values) > 0:
                        mean = np.mean(it_values)
                        row.append(round(mean, precision))
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
            writer.writerow(row)
        writer.writerow([])  # add an empty row
        # add rows with the average and std
        row_average = ["Average"]
        row_std_dev = ["Std Dev"]
        map_preset_to_average = {}
        for preset in presets:
            preset_name = preset["name"]
            if preset_name in metrics:
                val_arrays = [
                    metrics[preset_name][dataset["name"]][iteration_idx][metric_name]
                    for dataset in datasets
                    if preset_name in metrics and dataset["name"] in metrics[preset_name]
                    for iteration_idx in range(number_of_runs_per_dataset)
                    if metric_name in metrics[preset_name][dataset["name"]].get(iteration_idx, {})
                    and not math.isnan(
                        metrics[preset_name][dataset["name"]][iteration_idx][metric_name]
                    )
                ]
                mean = np.mean(val_arrays)
                row_average.append(round(mean, precision))
                std = np.std(val_arrays)
                row_std_dev.append(round(std, precision))
                map_preset_to_average[preset_name] = mean
            else:
                row_average.append("N/A")
                row_std_dev.append("N/A")
        writer.writerow(row_average)
        writer.writerow(row_std_dev)
        writer.writerow([])  # add an empty row
        # add final rows with best preset (with best average) and corresponding average
        row_best_preset = ["Best (Average) Preset"]
        row_best_average = ["Best (Average) Metric"]
        best_preset_name = ""
        best_average = float("inf")
        for preset_name, average in map_preset_to_average.items():
            if average < best_average:
                best_preset_name = preset_name
                best_average = average
        writer.writerow(row_best_preset + [best_preset_name])
        writer.writerow(row_best_average + [round(best_average, precision)])
        Printer.yellow(f'Generated table for metric "{metric_name}" in {output_table_path}')
        return map_preset_to_average, output_table_path

    def create_final_table(self):
        for preset in self.presets:
            for dataset in self.datasets:
                for iteration_idx in range(self.number_of_runs_per_dataset):
                    self.collect_metrics(
                        self.output_path,
                        preset["name"],
                        dataset["name"],
                        iteration_idx,
                        self.metrics,
                    )

        map_preset_to_ATE, out_table_ATE_path = self.write_comparative_table(
            self.output_path,
            "rmse",
            self.presets,
            self.datasets,
            self.number_of_runs_per_dataset,
            self.metrics,
        )
        map_preset_to_max, out_table_max_path = self.write_comparative_table(
            self.output_path,
            "max",
            self.presets,
            self.datasets,
            self.number_of_runs_per_dataset,
            self.metrics,
        )
        map_preset_to_percent_lost, out_table_percent_lost_path = self.write_comparative_table(
            self.output_path,
            "percent_lost",
            self.presets,
            self.datasets,
            self.number_of_runs_per_dataset,
            self.metrics,
        )
        # a table is missing when its metric is not available (e.g. no ground truth: no rmse and max)
        table_paths = [
            path
            for path in (out_table_ATE_path, out_table_max_path, out_table_percent_lost_path)
            if path
        ]

        git_commit_hash = get_git_short_hash()
        print("git commit hash: ", git_commit_hash)

        try:
            from pyslam.utilities.evaluation_latex import csv_list_to_pdf

            if shutil.which("pdflatex") is None:
                raise FileNotFoundError("pdflatex")
            csv_list_to_pdf(
                table_paths,
                self.output_path + "/report.pdf",
                title="pySLAM Evaluation Report",
                git_commit_hash=git_commit_hash,
            )
        except FileNotFoundError:
            print("No pdf report (pdflatex is not installed): see report.html and the tables")
        except Exception as e:
            print("Error converting tables to pdf report: ", e)
            print(traceback.format_exc())

        try:
            from pyslam.utilities.evaluation_html import csv_list_to_html

            csv_list_to_html(
                table_paths,
                self.output_path + "/report.html",
                title="pySLAM Evaluation Report",
                git_commit_hash=git_commit_hash,
            )
        except Exception as e:
            print("Error converting tables to html report: ", e)
            print(traceback.format_exc())
