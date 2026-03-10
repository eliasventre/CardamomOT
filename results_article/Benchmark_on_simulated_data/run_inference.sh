echo "Current working directory: $(pwd)"

eval "$(conda shell.bash hook)"
conda activate cardamom_light

PYTHON=/usr/bin/python3

# python infer_CARDAMOM2_noloop.py 
# python infer_CARDAMOM2_randomnoloop.py 

# python infer_CARDAMOM1.py 
python infer_CARDAMOM2.py 
# python infer_CARDAMOM2_random.py 

# python infer_CARDAMOM2_degover8.py 
# python infer_CARDAMOM2_degover4.py 
# python infer_CARDAMOM2_degover2.py 
# python infer_CARDAMOM2_deg.py 
# python infer_CARDAMOM2_degmult2.py 
# python infer_CARDAMOM2_degmult4.py 
# python infer_CARDAMOM2_degmult8.py 
# python infer_CARDAMOM2_degmult16.py 

## Requires reference fitting environment that can be installed from reference_fitting.yml
# conda deactivate
# conda activate reference_fitting
# python infer_reference_fitting.py