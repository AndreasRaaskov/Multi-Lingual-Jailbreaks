#!/bin/sh 
### Script for running dtu HPC
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J "LlamaOnFaster"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- ask for number of cores (default: 1) -- 
#BSUB -n 8 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 12:00 
### -- set the email address -- 
#BSUB -u s183901@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o hpc_out/Output_%J.out 
#BSUB -e hpc_out/Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
#myapplication.x input.in > output.out
source ./env/bin/activate
python3 Llama.py