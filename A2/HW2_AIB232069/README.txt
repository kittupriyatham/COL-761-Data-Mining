                                          Read me( Knowledge_Miners)

M Yagnesh                   2023AIB2069 - 34%
Abhishek Goyal              2023AIB2073 - 33%
Potluri Krishna Priyatham   2023AIB2084 - 33%

Q1: gSpan vs FSG vs Gaston
We have uploaded compiled files for the above algos on github , you can also do the following to get the compiled files.
Gaston – It is compiled Gaston C++ code for 64bit linux based system.

FSG-contains the PAFI C++ code for 64bit linux based system.

          Run following commands and obtain /pafi-1.0.1/Linux/fsg.

          gunzip pafi-1.0.1.tar.gz

           tar -xvf pafi-1.0.1.tar

gSpan-contains the gSpan C++ code for 64-bit linux based system .

Link to access these files are given below.

final.sh-This is the main bash file to run the code, it uses pre_process.py to preprocess yeast dataset and plot.py to plot the graphs on time vs support data generated in execution_times.csv file.

pre_process.py-Needed to preprocess the yeast dataset according to the input taken by mining algos.

plot.py - It gets data from file csv file formed by final.sh to plot and generate output_plot.png.

Running on HPC:

Give permission to every executable file using chmod 777 “ file_name”.

Take resources required to run the process, 24 GB is enough,and Load module using following commands.

qsub -P scai -q low -l select=2: ncpus=8:ngpus=1:mem=24G -l walltime=6:00:00

module load apps/layoutlmft/python3.7

All required modules matplotlib , pandas , numpy  are present in above hpc module loaded.

If you get bc: “command not found” error run  sudo apt-get install bc for basic calculation in bash.

Keep all these files in one folder and run  ./final.sh  “filename” .In the directory output_plot.png should get generated after a upper bound of 3.5 hrs .

Links to obtain the compiled Gaston , gSpan, FSG files.

GSpan - https://piazza.com/redirect/s3?bucket=uploads&prefix=paste%2Fk4v1g30et235iq%2F6713e73a0da21c17971f4b6ccdb7cdc053ea94a850b35a294ed406efb211d015%2FgSpan-64

Gaston - https://piazza.com/redirect/s3?bucket=uploads&prefix=paste%2Fk4v1g30et235iq%2Fca1d40165a6db9d795a445af7edaf281f12ddd568d0992b135e117216c5d2d6d%2Fgaston

FSG- http://glaros.dtc.umn.edu/gkhome/fetch/sw/pafi/pafi-1.0.1.tar.gz

Q2: K-Means

k__means_lib.py: This file will be used to find the clusters in the given data, returns the optimal value of k and creates a plot which will plot the mean distance of points to cluster’s centroid vs k  It takes 3 arguments from the user. The first argument is the name of the file which contains the data.  The second argument is the dimensions of the data, and the final argument is the name of the file to which the plot is to be stored.

elbow_plot.sh: This is the file which is used to run the file k__means_lib.py It takes 3 arguments from the user namely <data_file> <dimension> <output_file>. <data_file> should be in .dat format, <dimension> is an integer and <output_file> should be an image format

$ module load apps/layoutlmft/python3.7   //on HPC only (we are not sure if this has sklearn , please load appropriate module having sklearn)

$ elbow_plot.sh <data_file> <dimension> <output_file>



