import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import sin,cos,pi
import pylab
from pylab import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import Tkinter as Tk

root = Tk.Tk()
root.wm_title('Dynamic Plot')

fig = pylab.figure(1)
ax = fig.add_subplot(111)
ax.grid(True)

a = 100

#fig, ax = plt.subplots()
t = range(0,100,10)
x = a*sin(t)
#line, = ax.plot(t,x)
line, =ax.plot(t,x)
ax.grid(True)

canvas=FigureCanvasTkAgg(fig,master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)

toolbar = NavigationToolbar2TkAgg(canvas,root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
#ax.set_ylim(0, 1)
ampScale = Tk.Scale(root,label='Amplitute',from_=50,to=500,sliderlength=30,length=ax.patch.get_window_extent().width,orient=Tk.HORIZONTAL)
ampScale.pack(side=Tk.BOTTOM)
ampScale.set(100)

def _quit():
        root.quit()
        root.destroy()

button=Tk.Button(root,text='Quit',command=_quit)
button.pack(side=Tk.BOTTOM)


def update(data):
    global ampScale,line
    line.set_ydata(ampScale.get()*sin(t))
    return line,

ani = animation.FuncAnimation(fig,update,ampScale.get()*sin(t), interval=100)
#ani = animation.FuncAnimation(fig,update,init_func=init,interval=100)
#plt.show()


root.mainloop()

