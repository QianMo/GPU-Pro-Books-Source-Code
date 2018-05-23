import pydbfig
import datetime
import os
from subprocess import call
import sys

db = sys.argv[1]

pydbfig.makeFig(db,'numfrag5','NumFrags','Time','Time vs Number of fragments','fig_numfrag5_tm.pdf')
# pydbfig.makeFig(db,'numfrag5','NumFrags','ByteSize','Mem vs Number of fragments','fig_numfrag5_sz.pdf')
pydbfig.makeFig(db,'numfrag20','NumFrags','Time','Time vs Number of fragments','fig_numfrag20_tm.pdf')
# pydbfig.makeFig(db,'numfrag20','NumFrags','ByteSize','Mem vs Number of fragments','fig_numfrag20_sz.pdf')
pydbfig.makeFig(db,'numfrag63','NumFrags','Time','Time vs Number of fragments','fig_numfrag63_tm.pdf')
# pydbfig.makeFig(db,'numfrag63','NumFrags','ByteSize','Mem vs Number of fragments','fig_numfrag63_sz.pdf')
pydbfig.makeFig(db,'avgdepth','AvgDepth','Time','Time vs Avg depth complexity','fig_avgdepth_tm.pdf')
# pydbfig.makeFig(db,'loadfactor','LoadFactor','Time','Time vs load factor','fig_loadfactor_tm.pdf')
# pydbfig.makeFig(db,'maxVScas','NumFrags','Time','Time vs Max64 and CAS32','fig_maxvscas_tm.pdf')
# pydbfig.makeFig(db,'earlycull','Opacity','Time','Time vs Opacity','fig_earlycull_tm.pdf')
# pydbfig.makeBreackoutBars(db,'view1',['TimeClear','TimeBuild','TimeRender'],'','Time','fig_breakout_tm_view1.pdf')
# pydbfig.makeBreackoutBars(db,'view1',['ByteSize'],'','Memory','fig_membar_tm_view1.pdf')
# pydbfig.makeBreackoutBars(db,'view2',['TimeClear','TimeBuild','TimeRender'],'','Time','fig_breakout_tm_view2.pdf')
# pydbfig.makeBreackoutBars(db,'view2',['ByteSize'],'','Memory','fig_membar_tm_view2.pdf')
