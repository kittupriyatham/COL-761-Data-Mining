#!/bin/bash

# Pre-process the data (assuming you have a pre_process.sh script)
python3 pre_process.py

# Define support values
sup_list=(0.1 0.25 0.5 0.95)

# Define algorithms and their respective commands
declare -A algorithms
algorithms["gaston"]="./gaston support_freq yeast_processed -p"
algorithms["fsg"]="./fsg -s support_100 yeast_processed"
algorithms["gspan"]="./gSpan -f yeast_processed -s support_float -o -i"

# Create a CSV file to store execution times
echo "Algorithm Name,Support Value,Execution Time (seconds)" > execution_times.csv

# Loop through algorithms
for algo in "${!algorithms[@]}"; do
    for support in "${sup_list[@]}"; do
	
	    support_100=$(echo "$support * 100" | bc)
        support_freq=$(echo "$support * 64110" | bc)
        cmd="${algorithms[$algo]}"
        cmd="${cmd/support_100/$support_100}"
        cmd="${cmd/support_freq/$support_freq}"
        cmd="${cmd/support_float/$support}"

        echo "Running $algo with command $cmd"

        # Start time
        start_time=$(date +%s.%N)

        # Execute the algorithm
        $cmd

        # End time
        end_time=$(date +%s.%N)

        # Calculate execution time
        execution_time=$(echo "$end_time - $start_time" | bc)

        # Append to the CSV file
        echo "$algo,$support,$execution_time" >> execution_times.csv

        echo "$algo executed successfully in $execution_time seconds"
    done
done

python3 plot.py
echo "Plots Generated"
