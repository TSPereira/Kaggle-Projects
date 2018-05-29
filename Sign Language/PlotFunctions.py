import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

colors_array = list(matplotlib.colors.cnames.keys())
lines_array = list(matplotlib.lines.lineStyles.keys())
markers_array = list(matplotlib.markers.MarkerStyle.markers.keys())


def two_y_axis(x, y1, y2, ax_1):
	ax_1.plot(x, y1)
	ax_2 = ax_1.twinx()
	ax_2.plot(x, y2)
	return ax_1, ax_2

def setattrs(_self, color='', marker='', linestyle='-'):
	_self.set_color(color)
	_self.set_marker(marker)
	_self.set_linestyle(linestyle)


'''
fig, axes = plt.subplots(2,1)
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
s2 = np.sin(2 * np.pi * t)
ax1, ax2 = two_y_axis(t,s1,s2, axes[0])


ax1.set_xlabel('time (s)')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('exp', color='b')
ax1.tick_params('y', colors='b')

ax2.set_ylabel('sin', color='r')
ax2.tick_params('y', colors='r')

setattrs(ax2.lines[0],'r','.','')

ax3, ax4 = two_y_axis(t, s1, s2, axes[1])
ax3.set_xlabel('something')
setattrs(ax4.lines[0],'g',linestyle="-.")

fig.tight_layout()
plt.show()
'''