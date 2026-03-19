echo "Current working directory: $(pwd)"

eval "$(conda shell.bash hook)"
conda activate cardamom_light

PYTHON=/usr/bin/python3

python figure_2.py 
python figure6.py 
python figure7.py 
# python figure3.py 
# python figure4.py 
python figure5.py 
python figure5_trajectories.py 
 # python figureS7.py 
