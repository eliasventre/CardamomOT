echo "Current working directory: $(pwd)"

eval "$(conda shell.bash hook)"
conda activate cardamom_light

PYTHON=/usr/bin/python3

# python figure_2.py 
# python figure_3.py 
# python figure_4.py 
# python figure_5.py 
python figure_6.py 
# python figure_7.py 

# python figure_S6.py 
 # python figure_S7.py 
