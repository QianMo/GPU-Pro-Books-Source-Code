import pydbfig
import datetime
import os
from subprocess import call

if os.path.exists('bench.db'):
	os.remove('bench.db')

strdate = datetime.datetime.today().strftime("%Y-%m-%d_%H_%M_%S")

# for i in range(8):
	# n = 5
	# r = 128+64*i
	# g = (1024-r)/2
	# call(["benchmark.exe","prelin-naive.dll",         "-d numfrag5","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	# call(["benchmark.exe","postlin-naive.dll",  "-d numfrag5","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	# call(["benchmark.exe","postopen.dll",          "-d numfrag5","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	# call(["benchmark.exe","preopen.dll",          "-d numfrag5","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])

# for i in range(8):
	# n = 20
	# r = 128+64*i
	# g = (1024-r)/2
	# call(["benchmark.exe","prelin-naive.dll",         "-d numfrag20","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	# call(["benchmark.exe","postlin-naive.dll",  "-d numfrag20","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	# call(["benchmark.exe","postopen.dll",          "-d numfrag20","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	# call(["benchmark.exe","preopen.dll",          "-d numfrag20","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])

# for i in range(8):
	# n = 63
	# r = 128+64*i
	# g = (1024-r)/2
	# call(["benchmark.exe","prelin-naive.dll",         "-d numfrag63","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	# call(["benchmark.exe","postlin-naive.dll",  "-d numfrag63","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	# call(["benchmark.exe","postopen.dll",          "-d numfrag63","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	# call(["benchmark.exe","preopen.dll",          "-d numfrag63","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	
for i in range(10):
	n = 150
	r = 128
	g = (i+1) * 25
	call(["benchmark.exe","prelin-naive.dll",         "-d avgdepth","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	call(["benchmark.exe","postlin-naive.dll",  "-d avgdepth","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	call(["benchmark.exe","postopen.dll",          "-d avgdepth","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])
	call(["benchmark.exe","preopen.dll",          "-d avgdepth","-n {0}".format(n),"-r {0}".format(r),"-g {0}".format(g)])

# for i in range(10):
	# n = 150
	# r = 128
	# l = 0.5 + 0.05 * i
	# call(["benchmark.exe","postopen.dll",          "-d loadfactor","-n {0}".format(n),"-r {0}".format(r),"-l {0}".format(l)])
	# call(["benchmark.exe","preopen.dll",          "-d loadfactor","-n {0}".format(n),"-r {0}".format(r),"-l {0}".format(l)])

# for i in range(10):
	# n = 100+(i+1)*25
	# r = 128
	# call(["benchmark.exe","prelin-naive.dll",         "-d maxVScas","-n {0}".format(n),"-r {0}".format(r)])
	# call(["benchmark.exe","prelin-cas32-naive.dll",   "-d maxVScas","-n {0}".format(n),"-r {0}".format(r)])

# for i in range(10):
	# n = 200
	# r = 256
	# o = 0.1+i*0.1
	# call(["benchmark.exe","preopen.dll",         "-d earlycull","-n {0}".format(n),"-r {0}".format(r),"-o {0}".format(o)])
	# call(["benchmark.exe","preopen-ec.dll",      "-d earlycull","-n {0}".format(n),"-r {0}".format(r),"-o {0}".format(o)])

# tracks = ["trackball.F01","trackball.F02"]
# names = ["view1","view2"]
# for i in range(len(tracks)):
	# call(["seethrough.exe","postlin-naive.dll","-m","lost_empire.obj","-t",tracks[i],"-p","-d",names[i]])
	# call(["seethrough.exe","prelin-naive.dll","-m","lost_empire.obj","-t",tracks[i],"-p","-d",names[i]])
	# call(["seethrough.exe","postopen.dll","-m","lost_empire.obj","-t",tracks[i],"-p","-d",names[i]])
	# call(["seethrough.exe","preopen.dll","-m","lost_empire.obj","-t",tracks[i],"-p","-d",names[i]])
	# call(["seethrough.exe","preopen-ec.dll","-m","lost_empire.obj","-t",tracks[i],"-p","-d",names[i]])

os.rename('bench.db','{0}-bench.db'.format(strdate))
