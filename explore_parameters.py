import subprocess
import numpy as np

# simulation param
num_runs = 50

# parameter ranges
prop_mt_start = 0.2
prop_mt_end = 0.8
prop_mt_samples = 5

p_start = 0.05
p_end = 0.30
p_samples = 6

gamma_start = 0.90
gamma_end = 1.0
gamma_samples = 11

b = 0.1

for i in range(1000):
	print(i)
	for p in np.linspace(p_start, p_end, p_samples):
		for g in np.linspace(gamma_start, gamma_end, gamma_samples):
			subprocess.call('python linear_regression_agent.py %f 0.3 %f %f %f >> ./ols_int_data/data%.2f0.30%.2f%.2f%.2f' % (p, g, b, b, p, g, b, b), shell=True)
			subprocess.call('python linear_regression_agent.py %f 0.3 %f %f %f >> ./ols_int_data/data%.2f0.30%.2f%.2f%.2f' % (p, g, b, 1.5 * b, p, g, b, 1.5 * b), shell=True)
			subprocess.call('python linear_regression_agent.py %f 0.3 %f %f %f >> ./ols_int_data/data%.2f0.30%.2f%.2f%.2f' % (p, g, b, 2 * b, p, g, b, 2 * b), shell=True)
			subprocess.call('python linear_regression_agent.py %f 0.3 %f %f %f >> ./ols_int_data/data%.2f0.30%.2f%.2f%.2f' % (p, g, b, 2.5 * b, p, g, b, 2.5 * b), shell=True)
			subprocess.call('python linear_regression_agent.py %f 0.3 %f %f %f >> ./ols_int_data/data%.2f0.30%.2f%.2f%.2f' % (p, g, b, 3 * b, p, g, b, 3 * b), shell=True)


