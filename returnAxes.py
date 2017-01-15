import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

#Define axes
ax = fig.add_subplot(111)

#Create plot function
def create_plot(data,ax):
   #Updae data
   pass
   ax.set_xlabel('common xlabel')
   ax.set_ylabel('common ylabel')
   return ax


def my_plotter(ax, data1, data2):

    ax.set_xlabel('X_data')
    ax.set_ylabel('Y_data')

    out = ax.plot(data1, data2)
    plt.show()
    return out

x = np.random.rand(10)
y = np.random.rand(10)

my_plotter(ax,x,y)
#plt.show()

