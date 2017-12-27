import subprocess

for i in range(100):
	print(i)
	subprocess.call('python logistic_regression_agent.py 0.25 0.3 0.95 0.2 0.50 >> ./path/data0.150.300.950.200.50', shell=True)
