
# coding: utf-8

# In[19]:

from statistics import mean
import numpy as np
import random

xs = [1,2,3,4,5]
ys = [5,4,6,5,6]

xs = np.array([1,2,3,4,5], dtype= np.float64)
ys = np.array([5,4,6,5,6], dtype = np.float64)


#this function is to calculate the slope of the best fit line we are gonna have. 
def best_fit_slope(xs,ys):
    m =(((mean(xs)*mean(ys)) - mean(xs*ys))/((mean(xs)**2) - mean(xs**2))) #slope
    b =  mean(ys)-m*mean(xs) #y-axis intercept
    return (m,b)

#preditction
predict_x = 7
predict_y = (m*predict_x)+b
print(predict_y)

m , b  = best_fit_slope(xs,ys)

regression_line = [(m*x)+b for x in xs]  #anonymous funtion to save all the y cordinates




# In[20]:

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
plt.scatter(xs,ys,predict_y,color='#003F72')
plt.plot(xs, regression_line)
plt.show()


# In[16]:

#now calculating the coffecient of error
def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)



# In[ ]:



