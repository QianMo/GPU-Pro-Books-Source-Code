import sqlite3
import numpy
import matplotlib
import matplotlib.pyplot as pyplot
import pylab
import random

def makeFig(db,table,x_axis,y_axis,title,figfilename):

	import matplotlib.font_manager
	font = matplotlib.font_manager.FontProperties(size=9)

	fig = pylab.figure(figsize=(8, 3))
	# pylab.title(title)
	pylab.xlabel(x_axis)
	pylab.ylabel(y_axis)
	plot = fig.add_subplot(111)
	plot.tick_params(axis='both', which='major', labelsize=9)
	plot.tick_params(axis='both', which='minor', labelsize=7)

	pylab.gcf().subplots_adjust(bottom=0.15)
	
	co = sqlite3.connect(db)
	c = co.cursor()

	curves = []
	c.execute("select name from {0} group by name order by name".format(table))
	for row in c:
		curves.append(row[0])

	for curve in curves:
		c.execute("select {2},{3} from {0} where name='{1}'".format(table,curve,x_axis,y_axis))
		x_values = []
		y_values = []
		for row in c:
			# print row
			x_values.append(row[0])
			y_values.append(row[1])
		plotter = getattr(pylab, 'plot')
		plotter(x_values,y_values,".-",markerfacecolor="w",linewidth=2,markeredgewidth=2,markersize=5,label=curve)

	pylab.legend(prop=font, loc='upper left')

	pylab.savefig(figfilename, dpi=120)
	pylab.close(fig)

def makeBreackoutBars(db,table,values,title,ylabelname,figfilename):

	fig = pylab.figure(figsize=(6, 3))
	# pylab.title(title)

	import matplotlib.font_manager
	font = matplotlib.font_manager.FontProperties(size=9)

	co = sqlite3.connect(db)
	c = co.cursor()

	names = []
	c.execute("select name from {0} group by name order by Id".format(table))
	for row in c:
		names.append(row[0])

	all_values_string = ""
	for v in values:
		all_values_string = all_values_string + "," + v
	all_values_string = all_values_string[1:]
	print(all_values_string)
	
	all_values_table = []
	for name in names:
		c.execute("select {2} from {0} where name='{1}'".format(table,name,all_values_string))
		for row in c:
			all_values_table.append( row )
	all_values_table = map(list, zip(*all_values_table))
	print(all_values_table)
	
	colors = ['r','g','b']
	
	ind = numpy.arange(len(names))
	width = 0.2
	cumul = [ 0 for x in all_values_table[0] ]
	c = 0
	for tbl in all_values_table:
		pyplot.bar(ind,tbl,width,bottom=cumul,color=colors[c%3])
		c = c + 1
		for i in range(len(cumul)):
			cumul[i] = cumul[i] + tbl[i]
	
	# plotter = getattr(pylab, 'plot')
	# plotter(x_values,y_values,".-",markerfacecolor="w",linewidth=2,markeredgewidth=2,markersize=5,label=curve)
	
	pyplot.xticks(ind+width/2., names )
	pyplot.ylabel(ylabelname)

	pylab.legend(values, prop=font, loc='upper left')

	plot = fig.add_subplot(111)
	plot.tick_params(axis='both', which='major', labelsize=11)
	plot.tick_params(axis='both', which='minor', labelsize=11)
	
	pylab.savefig(figfilename, dpi=120)
	pylab.close(fig)
