import readline
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector

base = importr('base')
utils = importr('utils')
arm = importr('arm')

t = [99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
v = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]
t_r = FloatVector(t).r_repr()
v_r = FloatVector(v).r_repr()

robjects.r('''
	data = data.frame(%s, cbind(%s, 1 - %s))
	reg = bayesglm((cbind(%s, 1-%s)) ~ %s, data, family="binomial", prior.mean = 0.1, prior.mean.for.intercept = -3)
	''' % (t_r, v_r, v_r, v_r, v_r, t_r))

reg = robjects.r['reg']
print(reg.rx2('coefficients')[0])
print(reg.rx2('coefficients')[1])