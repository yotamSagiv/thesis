import subprocess
import numpy as np

# simulation param
num_runs = 50

# boundary conditions of interest
# boundary_params = []
# boundary_params.append((0, 0, 0.5)) # 0 time cost, 0 multitasking trials -> basis
# boundary_params.append((0, 1.0, 0.5)) # 0 time cost, 1.0 multitasking trials -> basis
# boundary_params.append((0.02, 0, 0.5)) # 0.02 time cost, 0 multitasking trials -> basis
# boundary_params.append((0.1, 1.0, 1.0)) # 0.1 time cost, 1.0 multitasking trials, no discounting -> tensor

# i = 0
# for i in range(num_runs):
# 	print(i)
	
# 	b = 0.1
# 	for p, mt, g in boundary_params: 
# 		subprocess.call('python infer_vp.py %f %f %f %f %f >> ./boundary/data%.2f%.2f%.2f%.2f%.2f' % (p, mt, g, b, 2 * b, p, mt, g, b, 2 * b), shell=True)
	
# 	subprocess.call('python infer_vp.py %f %f %f %f %f >> ./boundary/data%.2f%.2f%.2f%.2f%.2f' % (0, 0, 0.5, 2 * b, b, 0, 0, 0.5, 2 * b, b), shell=True)
					
		

# parameter ranges
prop_mt_start = 0.2
prop_mt_end = 0.8
prop_mt_samples = 5

p_start = 0.05
p_end = 0.30
p_samples = 6

gamma_start = 0.5
gamma_end = 1.0
gamma_samples = 6

b = 0.2

for i in range(1000):
	print(i)
	for p in np.linspace(p_start, p_end, p_samples):
		subprocess.call('python infer_vp.py %f 0.3 0.9 %f %f >> ./data/data%.2f0.300.90%.2f%.2f' % (p, b, b, p, b, b), shell=True)
		subprocess.call('python infer_vp.py %f 0.3 0.9 %f %f >> ./data/data%.2f0.300.90%.2f%.2f' % (p, b, 1.5 * b, p, b, 1.5 * b), shell=True)
		subprocess.call('python infer_vp.py %f 0.3 0.9 %f %f >> ./data/data%.2f0.300.90%.2f%.2f' % (p, b, 2 * b, p, b, 2 * b), shell=True)
		subprocess.call('python infer_vp.py %f 0.3 0.9 %f %f >> ./data/data%.2f0.300.90%.2f%.2f' % (p, b, 2.5 * b, p, b, 2.5 * b), shell=True)
		subprocess.call('python infer_vp.py %f 0.3 0.9 %f %f >> ./data/data%.2f0.300.90%.2f%.2f' % (p, b, 3 * b, p, b, 3 * b), shell=True)

# total number of iterations = 50 * 5 * 10 * 6 * 5 = 75,000
# i = 0
# for i in range(num_runs): 
# 	print(i)
# 	for mt in np.linspace(prop_mt_start, prop_mt_end, prop_mt_samples): 
# 		for p in np.linspace(p_start, p_end, p_samples):
# 			for g in np.linspace(gamma_start, gamma_end, gamma_samples):
# 				subprocess.call('python infer_vp.py %f %f %f %f %f >> ./data/data%.2f%.2f%.2f%.2f%.2f' % (p, mt, g, b, b, p, mt, g, b, b), shell=True)
# 				subprocess.call('python infer_vp.py %f %f %f %f %f >> ./data/data%.2f%.2f%.2f%.2f%.2f' % (p, mt, g, b, 1.5 * b, p, mt, g, b, 1.5 * b), shell=True)
# 				subprocess.call('python infer_vp.py %f %f %f %f %f >> ./data/data%.2f%.2f%.2f%.2f%.2f' % (p, mt, g, b, 2 * b, p, mt, g, b, 2 * b), shell=True)
# 				subprocess.call('python infer_vp.py %f %f %f %f %f >> ./data/data%.2f%.2f%.2f%.2f%.2f' % (p, mt, g, b, 2.5 * b, p, mt, g, b, 2.5 * b), shell=True)
# 				subprocess.call('python infer_vp.py %f %f %f %f %f >> ./data/data%.2f%.2f%.2f%.2f%.2f' % (p, mt, g, b, 3 * b, p, mt, g, b, 3 * b), shell=True)



