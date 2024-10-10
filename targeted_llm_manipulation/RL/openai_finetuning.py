import io
import json
import time
from collections import defaultdict

from openai import OpenAI


def validate_dataset(dataset):

    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        raise ValueError("Errors found in the dataset")
    else:
        print("No errors found")


def openai_finetuning(training_args):
    client = OpenAI()
    print(training_args)

    file_path = training_args["data_path"]
    dataset = []
    with open(file_path, "r") as file:
        for line in file:
            dataset.append(json.loads(line))

    validate_dataset(dataset)

    for traj in dataset:
        messages = traj["messages"]
        if "num_hardcoded_msgs" in traj:
            m = traj["num_hardcoded_msgs"]
            for message in messages:
                if message["role"] == "assistant":
                    m -= 1
                    message["weight"] = 0
                if m == 0:
                    break
        del traj["num_hardcoded_msgs"]

    # Convert the dataset back into JSONL format
    jsonl_data = "\n".join(json.dumps(item) for item in dataset)

    # Encode the JSONL string to bytes
    file_data = jsonl_data.encode("utf-8")

    # Create a file-like object from the bytes data
    file_like_object = io.BytesIO(file_data)

    file_obj = client.files.create(file=file_like_object, purpose="fine-tune")
    file_id = file_obj.id
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=training_args["model_name"],
        hyperparameters={
            "n_epochs": training_args["n_train_epochs"],
            "batch_size": training_args["batch_size"],
            "learning_rate_multiplier": training_args["learning_rate_multiplier"],
        },
    )

    # Store the job ID
    job_id = job.id

    while True:
        try:
            job_status = client.fine_tuning.jobs.retrieve(job_id)
            if job_status.status == "succeeded":
                print("Fine-tuning job completed successfully")
                break
            elif job_status.status == "failed":
                print("Fine-tuning job failed")
                break
            time.sleep(10)  # Wait for 20 seconds before checking again
        except Exception as e:
            print(f"An error occurred while retrieving job status: {str(e)}")
            time.sleep(30)  # Wait for 30 seconds before retrying

    fine_tuned_model_name = job_status.fine_tuned_model
    print(f"The fine-tuned model name is: {fine_tuned_model_name}")
    return fine_tuned_model_name
