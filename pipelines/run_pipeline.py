import subprocess
import logging
import os
from datetime import datetime


# -----------------------------
# Create Logs Folder
# -----------------------------

os.makedirs(
    "logs",
    exist_ok=True
)


# -----------------------------
# Logging Configuration
# -----------------------------

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)



# -----------------------------
# Run Pipeline Step
# -----------------------------

def run_step(command):

    logging.info(
        f"Starting: {command}"
    )

    print(
        "\nRunning:",
        command
    )


    result = subprocess.run(
        command,
        shell=True
    )


    if result.returncode != 0:

        logging.error(
            f"Failed: {command}"
        )

        raise Exception(
            f"Pipeline failed at {command}"
        )


    logging.info(
        f"Completed: {command}"
    )



# -----------------------------
# Main Pipeline
# -----------------------------

if __name__ == "__main__":


    start_time = datetime.now()


    logging.info(
        "Advertising Sales Prediction Pipeline Started"
    )


    print(
        "===================================="
    )

    print(
        "Advertising Sales Prediction Pipeline"
    )

    print(
        "===================================="
    )


    try:

        # Step 1
        run_step(
            "python src/preprocess.py"
        )


        # Step 2
        run_step(
            "python src/train.py"
        )


        # Step 3
        run_step(
            "python src/evaluate.py"
        )


        # Step 4
        run_step(
            "python src/predict.py"
        )


        end_time = datetime.now()


        logging.info(
            "Pipeline completed successfully"
        )


        logging.info(
            f"Total time: {end_time-start_time}"
        )


        print(
            "\nPipeline completed successfully!"
        )


    except Exception as e:


        logging.error(
            str(e)
        )


        print(
            "\nPipeline failed!"
        )

        print(
            e
        )