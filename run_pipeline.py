import subprocess


def run_step(name, command):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    result = subprocess.run(
        command,
        shell=True
    )

    if result.returncode != 0:
        raise Exception(
            f"{name} failed"
        )


if __name__ == "__main__":

    print(
        "Starting Advertising Sales Prediction Pipeline"
    )

    run_step(
        "Preprocessing",
        "python src/preprocess.py"
    )

    run_step(
        "Training",
        "python src/train.py"
    )

    run_step(
        "Evaluation",
        "python src/evaluate.py"
    )

    run_step(
        "Prediction",
        "python src/predict.py"
    )


    print("\nPipeline completed successfully!")