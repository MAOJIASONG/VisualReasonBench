import os
import json
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List

from PIL import Image


class ExperimentLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initializes the logger for an experiment.

        Args:
            log_dir (str): The base directory for logs.
            experiment_name (str): A unique name for the experiment.
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{self.timestamp}"
        self.run_dir = os.path.join(log_dir, self.experiment_name)
        self.images_dir = os.path.join(self.run_dir, "images")
        self.logs = []

        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def log_step(self, step: int, data: Dict[str, Any]):
        """
        Logs a single step of the experiment.

        Args:
            step (int): The current step number.
            data (Dict[str, Any]): A dictionary of data to log for the step.
                                   This can include prompts, responses, state info, etc.
        """
        log_entry = {"step": step, **data}

        # Handle image saving
        if "image" in log_entry and isinstance(log_entry["image"], Image.Image):
            image_path = os.path.join(self.images_dir, f"step_{step}.png")
            log_entry["image"].save(image_path)
            log_entry["image_path"] = image_path
            del log_entry["image"]
        
        # TODO: Handle saving of three-view images if required.

        self.logs.append(log_entry)

    def save_logs(self):
        """Saves all collected logs to a JSON file."""
        log_file = os.path.join(self.run_dir, "experiment_log.json")
        with open(log_file, "w") as f:
            json.dump(self.logs, f, indent=4)
        print(f"Logs saved to {log_file}")

    def save_results_to_excel(self, results: Dict[str, Any], excel_path: str):
        """
        Saves the final evaluation results to an Excel file.
        If the file exists, it appends the new results.

        Args:
            results (Dict[str, Any]): A dictionary of evaluation results.
            excel_path (str): The path to the output Excel file.
        """
        results_df = pd.DataFrame([results])

        if os.path.exists(excel_path):
            try:
                existing_df = pd.read_excel(excel_path)
                updated_df = pd.concat([existing_df, results_df], ignore_index=True)
            except Exception as e:
                print(f"Could not read existing excel file: {e}. Creating a new one.")
                updated_df = results_df
        else:
            updated_df = results_df

        updated_df.to_excel(excel_path, index=False)
        print(f"Results saved to {excel_path}")
