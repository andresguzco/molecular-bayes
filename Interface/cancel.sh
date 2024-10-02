#!/bin/bash

# Loop through numbers 487 to 717
for i in {9696..9710}; do
  # Format the number without extra padding
  job_id="1366$i"

  # Run the scancel command
  scancel $job_id

  # Optionally, echo the command being executed for verification
  echo "Cancelled job ID: $job_id"
done
