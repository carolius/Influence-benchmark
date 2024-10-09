import json
import os

folder = "influence_benchmark/config/env_configs/tickets"

for file in os.listdir(folder):
    if file.endswith(".json"):
        print(file)
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)
        for key, value in data["histories"].items():
            for message in value:
                assert "content" in message, f"Message does not contain content: {key}"
                assert "role" in message, f"Message does not contain role: {key}"
