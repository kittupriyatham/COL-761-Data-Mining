#!/bin/bash

echo "File $1 found"
mv $1 yeast
echo "Renamed $1 to yeast found"
python3 pre_process.py

echo "Format changed, yeast_processed"
sup_list=(0.05 0.1 0.25 0.5 0.95)
declare -A algorithms
algorithms["gaston"]="./gaston support_freq yeast_processed -p"
algorithms["fsg"]="./fsg -s support_100 yeast_processed"
algorithms["gspan"]="./gSpan -f yeast_processed -s support_float -o -i"
echo "Algorithm Name,Support Value,Execution Time (seconds)" > execution_times.csv
for algo in "${!algorithms[@]}"; do
    for support in "${sup_list[@]}"; do	
	    support_100=$(echo "$support * 100" | bc)
        support_freq=$(echo "$support * 64110" | bc)
        cmd="${algorithms[$algo]}"
        cmd="${cmd/support_100/$support_100}"
        cmd="${cmd/support_freq/$support_freq}"
        cmd="${cmd/support_float/$support}"
        echo "Running $algo with command $cmd"
        start_time=$(date +%s.%N)
        $cmd
        end_time=$(date +%s.%N)
        execution_time=$(echo "$end_time - $start_time" | bc)
        echo "$algo,$support,$execution_time" >> execution_times.csv
        echo "$algo executed successfully in $execution_time seconds"
    done
done

python3 plot.py
echo "Plots Generated"
