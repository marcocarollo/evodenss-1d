#!/bin/bash
#SBATCH -N1                       # Request 1 node
#SBATCH --ntasks=1                # Request 1 task
#SBATCH --gres=gpu:2              # Request 2 GPUs (or you can change this to the number you want to check)
#SBATCH --mem=4G                  # Memory request
#SBATCH --time=00:10:00           # Short time limit (10 minutes)
#SBATCH --partition=GPU           # Use the GPU partition (adjust based on your system)

# Load any necessary modules (if required)

# Check how many GPUs are available to this job
echo "Checking available GPUs"
nvidia-smi

# Check how many GPUs PyTorch can see
python -c "import torch; print(f'PyTorch can detect {torch.cuda.device_count()} GPU(s)')"
