echo "Current working directory: $(pwd)"

eval "$(conda shell.bash hook)"
conda activate cardamom_light

PYTHON=/usr/bin/python3

python figure5.py 
python figure5_traj.py 
