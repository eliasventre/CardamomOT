# Launch CARDAMOM-OT in one command line
# create a new directory for each parameter combination
# Start from /pbs/home/o/ogandril/

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

##################
# Hyperparameters
# Pathways and files
cwd = os.getcwd()

D=3753# project name
P=1 #Experiment within project
Th_int=2.5 #threshold for interactions 

# Which function should be executed
transform=0 # old to new
Pre_comp=1 # If a precomputed anndata is available
Infer=1# to infer the GRN
simulate=0# to simulate the GRN
perturb=1# to perturb the GRN (KO/OV)

# Create a working directory
path_1 = os.path.join(cwd, f"OG{D}")
os.makedirs(path_1, exist_ok=True)  # Using os.makedirs is safer than os.system

# Create a working subdirectory
path_2 = os.path.join(path_1, str(P))
os.makedirs(path_2, exist_ok=True)

# Define path and create folders
path_4 = os.path.join(cwd, 'CardamomOT')
path_3 = os.path.join(path_4, 'utils/old_to_new')
path_5 = os.path.join(path_2, 'cardamomOT')
os.makedirs(path_5, exist_ok=True)  
path_6 = os.path.join(path_2, 'Data')
os.makedirs(path_6, exist_ok=True)  
path_7 = os.path.join(path_2, 'halflife')
os.makedirs(path_7, exist_ok=True)  
path_8 = os.path.join(path_2, 'Rates')
os.makedirs(path_8, exist_ok=True)  

# Copy various files in the new folder
# carda3; this file containing hyperparameters
os.system("cp  Launching_Cardamom/CardaOT.py "+path_2)
# The CARDAMOM basic parameters
os.system(f"cp  CardamomOT/CardamomOT/model/base.py "+path_2)

if transform:
	os.chdir(path_3)
	# Launch R script to generate entry files
	os.system("echo 'create data real'")
	os.system(f"Rscript --vanilla  {cwd}/res_carda/{seq} {SFT} {CC} {P} {D} {f}")
	os.system("echo 'old_to_new'")
	os.system(f"python convert_old_data_to_ad.py -i {cwd}/OG{D}/{P}")
	os.system(f"python add_degradations_to_ad.py -i {cwd}/OG{D}/{P}")
	
if Pre_comp:
	os.system(f"cp  {cwd}/res_carda/data.h5ad "+path_6)
	os.system(f"cp  {cwd}/res_carda/inter_ref.csv "+path_6)
	# If uncorrected
	os.system(f"cp  {cwd}/res_carda/data.h5ad {cwd}/res_carda/data_train.h5ad ")
	os.system(f"cp  {cwd}/res_carda/data_train.h5ad "+path_6)
	os.system(f"cp  {cwd}/res_carda/data.h5ad {cwd}/res_carda/data_test.h5ad ")
	os.system(f"cp  {cwd}/res_carda/data_test.h5ad "+path_6)
	os.system(f"cp  {cwd}/res_carda/data.h5ad {cwd}/res_carda/data_full.h5ad ")
	os.system(f"cp  {cwd}/res_carda/data_full.h5ad "+path_6)

if Infer:
	os.chdir(path_4)

	os.system("echo 'Get kinetic rates'")
	os.system(f"python -m CardamomOT.cli step get_kinetic_rates -i {cwd}/OG{D}/{P}")

	os.system("echo 'Select DE genes and split cells'")
	os.system(f"python -m CardamomOT.cli step select_DEgenes_and_split -i {cwd}/OG{D}/{P} -s full -c 0 --mean-forcing 0.75")

	os.system("echo 'infer_mixture'")
	os.system(f"python -m CardamomOT.cli step infer_mixture -i {cwd}/OG{D}/{P} -s full --mean-forcing 0.75")

	os.system("echo 'Check mixture vs data consistency'")
	os.system(f"python -m CardamomOT.cli step check_mixture_to_data -i {cwd}/OG{D}/{P} -s full")

	os.system("echo 'Infer network structure'")
	os.system(f"python -m CardamomOT.cli step infer_network_structure -i {cwd}/OG{D}/{P} --stimulus 0.22 -s full")

	# Save a csv version of the interaction matrix after applying a threshold
	os.chdir(path_5)
	inter = np.load('inter.npy')
	# Save as .csv for R
	inter2D=inter[:, :, 0]
	np.savetxt('inter.csv', inter2D, delimiter=",")

if simulate:	
	os.chdir(path_4)

	os.system("echo 'Simulate network'")
	os.system(f"python -m CardamomOT.cli step infer_network_simul -i {cwd}/OG{D}/{P} --stimulus 0.22 -s full")
	
	# Save a csv version of the interaction matrix after applying a threshold
	os.chdir(path_5)
	inter = np.load('inter_simul.npy')
	# Save as .csv for R
	inter2D=inter[:, :, 0]
	np.savetxt('inter_simul.csv', inter2D, delimiter=",")

	os.system("echo 'Full simulation'")
	os.system(f"python -m CardamomOT.cli step simulate_network -i {cwd}/OG{D}/{P}  -s full")
	
	os.system("echo 'Final checks_1'")
	os.system(f"python -m CardamomOT.cli step check_sim_to_data -i {cwd}/OG{D}/{P} --stimulus 0.22 -s full")

	os.system("echo 'Final checks_2'")
	os.system(f"python -m CardamomOT.cli step simulate_network_KOV -i {cwd}/OG{D}/{P} -s full")

	os.system("echo 'Final checks_3'")
	os.system(f"python -m CardamomOT.cli step check_KOV_to_sim -i {cwd}/OG{D}/{P} --stimulus 0.22 -s full")

if perturb:

	# Write the genes to perturb.
	os.chdir(path_6)
	fichier = open('KO_OV_simulate.txt', 'w')
	fichier.write('KO\tOV\n')
	for arg in sys.argv[1:]:
		fichier.write(str(arg+"\t0\n"))
		fichier.write("0\t"+str(arg+"\n"))
	fichier.close()

	# Excecute the perturbation
	os.chdir(path_4)

	os.system("echo 'Simulate network'")
	os.system(f"python -m CardamomOT.cli step infer_network_simul -i {cwd}/OG{D}/{P}  -s full")
	
	# Save a csv version of the interaction matrix after applying a threshold
	os.chdir(path_5)
	inter = np.load('inter_simul.npy')
	inter_ref = np.load('inter_ref.npy')
	inter = inter * np.mean(inter_ref) / np.mean(inter)
	# Save the resulting matrix
	np.save('inter_simul.npy', inter)
	np.save('inter_ref.npy', inter_ref)
	# Save as .csv for R
	inter2D=inter[:, :, 0]
	np.savetxt('inter_simul.csv', inter2D, delimiter=",")
	np.savetxt('inter_ref.csv', inter_ref, delimiter=",") 

	os.system("echo 'Full simulation'")
	os.system(f"python -m CardamomOT.cli step simulate_network -i {cwd}/OG{D}/{P}  -s full")
	
	os.system("echo 'Final checks_1'")
	os.system(f"python -m CardamomOT.cli step check_sim_to_data -i {cwd}/OG{D}/{P} --stimulus 0.22 -s full")

	os.system("echo 'Final checks_2'")
	os.system(f"python -m CardamomOT.cli step simulate_network_KOV -i {cwd}/OG{D}/{P} -s full")

	os.system("echo 'Final checks_3'")
	os.system(f"python -m CardamomOT.cli step check_KOV_to_sim -i {cwd}/OG{D}/{P} --stimulus 0.22 -s full")


print('My work here is done')

