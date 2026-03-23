# import sys
# import os
# import mlflow

# # READS THE URI FROM YOUR GITHUB SECRETS
# tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
# mlflow.set_tracking_uri(tracking_uri)

# with open("model_info.txt", "r") as f:
#     run_id = f.read().strip()

# run = mlflow.get_run(run_id)
# accuracy = run.data.metrics.get("accuracy", 0.0)

# print(f"Run ID: {run_id}, Accuracy: {accuracy}")

# if accuracy < 0.85:
#     print("Error: Accuracy below 0.85 threshold. Failing the pipeline.")
#     sys.exit(1)
# else:
#     print("Success: Accuracy meets threshold. Proceeding to deploy.")
#     sys.exit(0)

import sys
import os
import mlflow

# READS THE URI FROM YOUR GITHUB SECRETS
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0.0)

print(f"Run ID: {run_id}, Accuracy: {accuracy}")

# THIS IS THE NEW PART: Write the results directly to the GitHub UI!
github_summary = os.getenv("GITHUB_STEP_SUMMARY")
if github_summary:
    with open(github_summary, "a") as f:
        f.write("###  Model Training Results\n")
        f.write(f"- **Run ID:** `{run_id}`\n")
        f.write(f"- **Accuracy:** `{accuracy:.4f}`\n")
        if accuracy < 0.85:
            f.write("- **Status:** xxxxxxxxxxxxxx FAILED (Below 0.85 threshold)\n")
        else:
            f.write("- **Status:** !!!!!!!!!!!!!!! PASSED (Ready to deploy)\n")

if accuracy < 0.85:
    print("Error: Accuracy below 0.85 threshold. Failing the pipeline.")
    sys.exit(1)
else:
    print("Success: Accuracy meets threshold. Proceeding to deploy.")
    sys.exit(0)