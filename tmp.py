import numpy as np


a = np.zeros([1,2,3,8])
n = len(a.reshape([-1]))
a = range(n)
h = 2
w = 3
ch = 8
a = np.reshape(a,[1,h,w,ch])
a = a.transpose([0,3,1,2])
a = np.reshape(a ,[ 1, 4 , h*2, w ])
a = a.transpose([0,2,3,1])
a=a.reshape([-1,4])
print a
print np.shape(a)




