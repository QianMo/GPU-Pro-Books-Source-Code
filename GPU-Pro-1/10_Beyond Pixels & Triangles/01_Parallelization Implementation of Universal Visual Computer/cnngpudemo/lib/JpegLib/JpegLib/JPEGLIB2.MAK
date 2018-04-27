# Microsoft Developer Studio Generated NMAKE File, Format Version 4.20
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

!IF "$(CFG)" == ""
CFG=jpeglib2 - Win32 Debug
!MESSAGE No configuration specified.  Defaulting to jpeglib2 - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "jpeglib2 - Win32 Release" && "$(CFG)" !=\
 "jpeglib2 - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE on this makefile
!MESSAGE by defining the macro CFG on the command line.  For example:
!MESSAGE 
!MESSAGE NMAKE /f "Jpeglib2.mak" CFG="jpeglib2 - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "jpeglib2 - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "jpeglib2 - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 
################################################################################
# Begin Project
# PROP Target_Last_Scanned "jpeglib2 - Win32 Debug"
CPP=cl.exe

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
OUTDIR=.\Release
INTDIR=.\Release

ALL : "$(OUTDIR)\Jpeglib2.lib"

CLEAN : 
	-@erase "$(INTDIR)\Jcapimin.obj"
	-@erase "$(INTDIR)\Jcapistd.obj"
	-@erase "$(INTDIR)\Jccoefct.obj"
	-@erase "$(INTDIR)\Jccolor.obj"
	-@erase "$(INTDIR)\Jcdctmgr.obj"
	-@erase "$(INTDIR)\Jchuff.obj"
	-@erase "$(INTDIR)\Jcinit.obj"
	-@erase "$(INTDIR)\Jcmainct.obj"
	-@erase "$(INTDIR)\Jcmarker.obj"
	-@erase "$(INTDIR)\Jcmaster.obj"
	-@erase "$(INTDIR)\Jcomapi.obj"
	-@erase "$(INTDIR)\Jcparam.obj"
	-@erase "$(INTDIR)\Jcphuff.obj"
	-@erase "$(INTDIR)\Jcprepct.obj"
	-@erase "$(INTDIR)\Jcsample.obj"
	-@erase "$(INTDIR)\Jctrans.obj"
	-@erase "$(INTDIR)\Jdapimin.obj"
	-@erase "$(INTDIR)\Jdapistd.obj"
	-@erase "$(INTDIR)\Jdatadst.obj"
	-@erase "$(INTDIR)\Jdatasrc.obj"
	-@erase "$(INTDIR)\Jdcoefct.obj"
	-@erase "$(INTDIR)\Jdcolor.obj"
	-@erase "$(INTDIR)\Jddctmgr.obj"
	-@erase "$(INTDIR)\Jdhuff.obj"
	-@erase "$(INTDIR)\Jdinput.obj"
	-@erase "$(INTDIR)\Jdmainct.obj"
	-@erase "$(INTDIR)\Jdmarker.obj"
	-@erase "$(INTDIR)\Jdmaster.obj"
	-@erase "$(INTDIR)\Jdmerge.obj"
	-@erase "$(INTDIR)\Jdphuff.obj"
	-@erase "$(INTDIR)\Jdpostct.obj"
	-@erase "$(INTDIR)\Jdsample.obj"
	-@erase "$(INTDIR)\Jdtrans.obj"
	-@erase "$(INTDIR)\Jerror.obj"
	-@erase "$(INTDIR)\Jfdctflt.obj"
	-@erase "$(INTDIR)\Jfdctfst.obj"
	-@erase "$(INTDIR)\Jfdctint.obj"
	-@erase "$(INTDIR)\Jidctflt.obj"
	-@erase "$(INTDIR)\Jidctfst.obj"
	-@erase "$(INTDIR)\Jidctint.obj"
	-@erase "$(INTDIR)\Jidctred.obj"
	-@erase "$(INTDIR)\Jmemmgr.obj"
	-@erase "$(INTDIR)\Jmemnobs.obj"
	-@erase "$(INTDIR)\Jpegtran.obj"
	-@erase "$(INTDIR)\Jquant1.obj"
	-@erase "$(INTDIR)\Jquant2.obj"
	-@erase "$(INTDIR)\Jutils.obj"
	-@erase "$(OUTDIR)\Jpeglib2.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /YX /c
# ADD CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /YX /c
CPP_PROJ=/nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS"\
 /Fp"$(INTDIR)/Jpeglib2.pch" /YX /Fo"$(INTDIR)/" /c 
CPP_OBJS=.\Release/
CPP_SBRS=.\.
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/Jpeglib2.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo
LIB32_FLAGS=/nologo /out:"$(OUTDIR)/Jpeglib2.lib" 
LIB32_OBJS= \
	"$(INTDIR)\Jcapimin.obj" \
	"$(INTDIR)\Jcapistd.obj" \
	"$(INTDIR)\Jccoefct.obj" \
	"$(INTDIR)\Jccolor.obj" \
	"$(INTDIR)\Jcdctmgr.obj" \
	"$(INTDIR)\Jchuff.obj" \
	"$(INTDIR)\Jcinit.obj" \
	"$(INTDIR)\Jcmainct.obj" \
	"$(INTDIR)\Jcmarker.obj" \
	"$(INTDIR)\Jcmaster.obj" \
	"$(INTDIR)\Jcomapi.obj" \
	"$(INTDIR)\Jcparam.obj" \
	"$(INTDIR)\Jcphuff.obj" \
	"$(INTDIR)\Jcprepct.obj" \
	"$(INTDIR)\Jcsample.obj" \
	"$(INTDIR)\Jctrans.obj" \
	"$(INTDIR)\Jdapimin.obj" \
	"$(INTDIR)\Jdapistd.obj" \
	"$(INTDIR)\Jdatadst.obj" \
	"$(INTDIR)\Jdatasrc.obj" \
	"$(INTDIR)\Jdcoefct.obj" \
	"$(INTDIR)\Jdcolor.obj" \
	"$(INTDIR)\Jddctmgr.obj" \
	"$(INTDIR)\Jdhuff.obj" \
	"$(INTDIR)\Jdinput.obj" \
	"$(INTDIR)\Jdmainct.obj" \
	"$(INTDIR)\Jdmarker.obj" \
	"$(INTDIR)\Jdmaster.obj" \
	"$(INTDIR)\Jdmerge.obj" \
	"$(INTDIR)\Jdphuff.obj" \
	"$(INTDIR)\Jdpostct.obj" \
	"$(INTDIR)\Jdsample.obj" \
	"$(INTDIR)\Jdtrans.obj" \
	"$(INTDIR)\Jerror.obj" \
	"$(INTDIR)\Jfdctflt.obj" \
	"$(INTDIR)\Jfdctfst.obj" \
	"$(INTDIR)\Jfdctint.obj" \
	"$(INTDIR)\Jidctflt.obj" \
	"$(INTDIR)\Jidctfst.obj" \
	"$(INTDIR)\Jidctint.obj" \
	"$(INTDIR)\Jidctred.obj" \
	"$(INTDIR)\Jmemmgr.obj" \
	"$(INTDIR)\Jmemnobs.obj" \
	"$(INTDIR)\Jpegtran.obj" \
	"$(INTDIR)\Jquant1.obj" \
	"$(INTDIR)\Jquant2.obj" \
	"$(INTDIR)\Jutils.obj"

"$(OUTDIR)\Jpeglib2.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
OUTDIR=.\Debug
INTDIR=.\Debug

ALL : "$(OUTDIR)\Jpeglib2.lib"

CLEAN : 
	-@erase "$(INTDIR)\Jcapimin.obj"
	-@erase "$(INTDIR)\Jcapistd.obj"
	-@erase "$(INTDIR)\Jccoefct.obj"
	-@erase "$(INTDIR)\Jccolor.obj"
	-@erase "$(INTDIR)\Jcdctmgr.obj"
	-@erase "$(INTDIR)\Jchuff.obj"
	-@erase "$(INTDIR)\Jcinit.obj"
	-@erase "$(INTDIR)\Jcmainct.obj"
	-@erase "$(INTDIR)\Jcmarker.obj"
	-@erase "$(INTDIR)\Jcmaster.obj"
	-@erase "$(INTDIR)\Jcomapi.obj"
	-@erase "$(INTDIR)\Jcparam.obj"
	-@erase "$(INTDIR)\Jcphuff.obj"
	-@erase "$(INTDIR)\Jcprepct.obj"
	-@erase "$(INTDIR)\Jcsample.obj"
	-@erase "$(INTDIR)\Jctrans.obj"
	-@erase "$(INTDIR)\Jdapimin.obj"
	-@erase "$(INTDIR)\Jdapistd.obj"
	-@erase "$(INTDIR)\Jdatadst.obj"
	-@erase "$(INTDIR)\Jdatasrc.obj"
	-@erase "$(INTDIR)\Jdcoefct.obj"
	-@erase "$(INTDIR)\Jdcolor.obj"
	-@erase "$(INTDIR)\Jddctmgr.obj"
	-@erase "$(INTDIR)\Jdhuff.obj"
	-@erase "$(INTDIR)\Jdinput.obj"
	-@erase "$(INTDIR)\Jdmainct.obj"
	-@erase "$(INTDIR)\Jdmarker.obj"
	-@erase "$(INTDIR)\Jdmaster.obj"
	-@erase "$(INTDIR)\Jdmerge.obj"
	-@erase "$(INTDIR)\Jdphuff.obj"
	-@erase "$(INTDIR)\Jdpostct.obj"
	-@erase "$(INTDIR)\Jdsample.obj"
	-@erase "$(INTDIR)\Jdtrans.obj"
	-@erase "$(INTDIR)\Jerror.obj"
	-@erase "$(INTDIR)\Jfdctflt.obj"
	-@erase "$(INTDIR)\Jfdctfst.obj"
	-@erase "$(INTDIR)\Jfdctint.obj"
	-@erase "$(INTDIR)\Jidctflt.obj"
	-@erase "$(INTDIR)\Jidctfst.obj"
	-@erase "$(INTDIR)\Jidctint.obj"
	-@erase "$(INTDIR)\Jidctred.obj"
	-@erase "$(INTDIR)\Jmemmgr.obj"
	-@erase "$(INTDIR)\Jmemnobs.obj"
	-@erase "$(INTDIR)\Jpegtran.obj"
	-@erase "$(INTDIR)\Jquant1.obj"
	-@erase "$(INTDIR)\Jquant2.obj"
	-@erase "$(INTDIR)\Jutils.obj"
	-@erase "$(OUTDIR)\Jpeglib2.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /W3 /GX /Z7 /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /YX /c
# ADD CPP /nologo /MTd /W3 /GX /Z7 /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /YX /c
CPP_PROJ=/nologo /MTd /W3 /GX /Z7 /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS"\
 /Fp"$(INTDIR)/Jpeglib2.pch" /YX /Fo"$(INTDIR)/" /c 
CPP_OBJS=.\Debug/
CPP_SBRS=.\.
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/Jpeglib2.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo
LIB32_FLAGS=/nologo /out:"$(OUTDIR)/Jpeglib2.lib" 
LIB32_OBJS= \
	"$(INTDIR)\Jcapimin.obj" \
	"$(INTDIR)\Jcapistd.obj" \
	"$(INTDIR)\Jccoefct.obj" \
	"$(INTDIR)\Jccolor.obj" \
	"$(INTDIR)\Jcdctmgr.obj" \
	"$(INTDIR)\Jchuff.obj" \
	"$(INTDIR)\Jcinit.obj" \
	"$(INTDIR)\Jcmainct.obj" \
	"$(INTDIR)\Jcmarker.obj" \
	"$(INTDIR)\Jcmaster.obj" \
	"$(INTDIR)\Jcomapi.obj" \
	"$(INTDIR)\Jcparam.obj" \
	"$(INTDIR)\Jcphuff.obj" \
	"$(INTDIR)\Jcprepct.obj" \
	"$(INTDIR)\Jcsample.obj" \
	"$(INTDIR)\Jctrans.obj" \
	"$(INTDIR)\Jdapimin.obj" \
	"$(INTDIR)\Jdapistd.obj" \
	"$(INTDIR)\Jdatadst.obj" \
	"$(INTDIR)\Jdatasrc.obj" \
	"$(INTDIR)\Jdcoefct.obj" \
	"$(INTDIR)\Jdcolor.obj" \
	"$(INTDIR)\Jddctmgr.obj" \
	"$(INTDIR)\Jdhuff.obj" \
	"$(INTDIR)\Jdinput.obj" \
	"$(INTDIR)\Jdmainct.obj" \
	"$(INTDIR)\Jdmarker.obj" \
	"$(INTDIR)\Jdmaster.obj" \
	"$(INTDIR)\Jdmerge.obj" \
	"$(INTDIR)\Jdphuff.obj" \
	"$(INTDIR)\Jdpostct.obj" \
	"$(INTDIR)\Jdsample.obj" \
	"$(INTDIR)\Jdtrans.obj" \
	"$(INTDIR)\Jerror.obj" \
	"$(INTDIR)\Jfdctflt.obj" \
	"$(INTDIR)\Jfdctfst.obj" \
	"$(INTDIR)\Jfdctint.obj" \
	"$(INTDIR)\Jidctflt.obj" \
	"$(INTDIR)\Jidctfst.obj" \
	"$(INTDIR)\Jidctint.obj" \
	"$(INTDIR)\Jidctred.obj" \
	"$(INTDIR)\Jmemmgr.obj" \
	"$(INTDIR)\Jmemnobs.obj" \
	"$(INTDIR)\Jpegtran.obj" \
	"$(INTDIR)\Jquant1.obj" \
	"$(INTDIR)\Jquant2.obj" \
	"$(INTDIR)\Jutils.obj"

"$(OUTDIR)\Jpeglib2.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ENDIF 

.c{$(CPP_OBJS)}.obj:
   $(CPP) $(CPP_PROJ) $<  

.cpp{$(CPP_OBJS)}.obj:
   $(CPP) $(CPP_PROJ) $<  

.cxx{$(CPP_OBJS)}.obj:
   $(CPP) $(CPP_PROJ) $<  

.c{$(CPP_SBRS)}.sbr:
   $(CPP) $(CPP_PROJ) $<  

.cpp{$(CPP_SBRS)}.sbr:
   $(CPP) $(CPP_PROJ) $<  

.cxx{$(CPP_SBRS)}.sbr:
   $(CPP) $(CPP_PROJ) $<  

################################################################################
# Begin Target

# Name "jpeglib2 - Win32 Release"
# Name "jpeglib2 - Win32 Debug"

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

################################################################################
# Begin Source File

SOURCE=.\Jversion.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcapistd.c
DEP_CPP_JCAPI=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcapistd.obj" : $(SOURCE) $(DEP_CPP_JCAPI) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jccoefct.c
DEP_CPP_JCCOE=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jccoefct.obj" : $(SOURCE) $(DEP_CPP_JCCOE) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jccolor.c
DEP_CPP_JCCOL=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jccolor.obj" : $(SOURCE) $(DEP_CPP_JCCOL) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcdctmgr.c
DEP_CPP_JCDCT=\
	".\Jconfig.h"\
	".\Jdct.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcdctmgr.obj" : $(SOURCE) $(DEP_CPP_JCDCT) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jchuff.c
DEP_CPP_JCHUF=\
	".\Jchuff.h"\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jchuff.obj" : $(SOURCE) $(DEP_CPP_JCHUF) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jchuff.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcinit.c
DEP_CPP_JCINI=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcinit.obj" : $(SOURCE) $(DEP_CPP_JCINI) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcmainct.c
DEP_CPP_JCMAI=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcmainct.obj" : $(SOURCE) $(DEP_CPP_JCMAI) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcmarker.c
DEP_CPP_JCMAR=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcmarker.obj" : $(SOURCE) $(DEP_CPP_JCMAR) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcmaster.c
DEP_CPP_JCMAS=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcmaster.obj" : $(SOURCE) $(DEP_CPP_JCMAS) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcomapi.c
DEP_CPP_JCOMA=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcomapi.obj" : $(SOURCE) $(DEP_CPP_JCOMA) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jconfig.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcparam.c
DEP_CPP_JCPAR=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcparam.obj" : $(SOURCE) $(DEP_CPP_JCPAR) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcphuff.c
DEP_CPP_JCPHU=\
	".\Jchuff.h"\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcphuff.obj" : $(SOURCE) $(DEP_CPP_JCPHU) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcprepct.c
DEP_CPP_JCPRE=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcprepct.obj" : $(SOURCE) $(DEP_CPP_JCPRE) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcsample.c
DEP_CPP_JCSAM=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcsample.obj" : $(SOURCE) $(DEP_CPP_JCSAM) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jctrans.c
DEP_CPP_JCTRA=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jctrans.obj" : $(SOURCE) $(DEP_CPP_JCTRA) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdapimin.c
DEP_CPP_JDAPI=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdapimin.obj" : $(SOURCE) $(DEP_CPP_JDAPI) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdapistd.c
DEP_CPP_JDAPIS=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdapistd.obj" : $(SOURCE) $(DEP_CPP_JDAPIS) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdatadst.c
DEP_CPP_JDATA=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdatadst.obj" : $(SOURCE) $(DEP_CPP_JDATA) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdatasrc.c
DEP_CPP_JDATAS=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdatasrc.obj" : $(SOURCE) $(DEP_CPP_JDATAS) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdcoefct.c
DEP_CPP_JDCOE=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdcoefct.obj" : $(SOURCE) $(DEP_CPP_JDCOE) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdcolor.c
DEP_CPP_JDCOL=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdcolor.obj" : $(SOURCE) $(DEP_CPP_JDCOL) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdct.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jddctmgr.c
DEP_CPP_JDDCT=\
	".\Jconfig.h"\
	".\Jdct.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jddctmgr.obj" : $(SOURCE) $(DEP_CPP_JDDCT) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdhuff.c
DEP_CPP_JDHUF=\
	".\Jconfig.h"\
	".\Jdhuff.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdhuff.obj" : $(SOURCE) $(DEP_CPP_JDHUF) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdhuff.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdinput.c
DEP_CPP_JDINP=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdinput.obj" : $(SOURCE) $(DEP_CPP_JDINP) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdmainct.c
DEP_CPP_JDMAI=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdmainct.obj" : $(SOURCE) $(DEP_CPP_JDMAI) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdmarker.c
DEP_CPP_JDMAR=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdmarker.obj" : $(SOURCE) $(DEP_CPP_JDMAR) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdmaster.c
DEP_CPP_JDMAS=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdmaster.obj" : $(SOURCE) $(DEP_CPP_JDMAS) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdmerge.c
DEP_CPP_JDMER=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdmerge.obj" : $(SOURCE) $(DEP_CPP_JDMER) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdphuff.c
DEP_CPP_JDPHU=\
	".\Jconfig.h"\
	".\Jdhuff.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdphuff.obj" : $(SOURCE) $(DEP_CPP_JDPHU) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdpostct.c
DEP_CPP_JDPOS=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdpostct.obj" : $(SOURCE) $(DEP_CPP_JDPOS) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdsample.c
DEP_CPP_JDSAM=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdsample.obj" : $(SOURCE) $(DEP_CPP_JDSAM) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jdtrans.c
DEP_CPP_JDTRA=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jdtrans.obj" : $(SOURCE) $(DEP_CPP_JDTRA) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jerror.c
DEP_CPP_JERRO=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	".\Jversion.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jerror.obj" : $(SOURCE) $(DEP_CPP_JERRO) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jerror.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jfdctflt.c
DEP_CPP_JFDCT=\
	".\Jconfig.h"\
	".\Jdct.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jfdctflt.obj" : $(SOURCE) $(DEP_CPP_JFDCT) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jfdctfst.c
DEP_CPP_JFDCTF=\
	".\Jconfig.h"\
	".\Jdct.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jfdctfst.obj" : $(SOURCE) $(DEP_CPP_JFDCTF) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jfdctint.c
DEP_CPP_JFDCTI=\
	".\Jconfig.h"\
	".\Jdct.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jfdctint.obj" : $(SOURCE) $(DEP_CPP_JFDCTI) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jidctflt.c
DEP_CPP_JIDCT=\
	".\Jconfig.h"\
	".\Jdct.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jidctflt.obj" : $(SOURCE) $(DEP_CPP_JIDCT) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jidctfst.c
DEP_CPP_JIDCTF=\
	".\Jconfig.h"\
	".\Jdct.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jidctfst.obj" : $(SOURCE) $(DEP_CPP_JIDCTF) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jidctint.c
DEP_CPP_JIDCTI=\
	".\Jconfig.h"\
	".\Jdct.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jidctint.obj" : $(SOURCE) $(DEP_CPP_JIDCTI) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jidctred.c
DEP_CPP_JIDCTR=\
	".\Jconfig.h"\
	".\Jdct.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jidctred.obj" : $(SOURCE) $(DEP_CPP_JIDCTR) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jinclude.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jmemnobs.c
DEP_CPP_JMEMN=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmemsys.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jmemnobs.obj" : $(SOURCE) $(DEP_CPP_JMEMN) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jmemsys.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jmorecfg.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jpegint.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jpeglib.h

!IF  "$(CFG)" == "jpeglib2 - Win32 Release"

!ELSEIF  "$(CFG)" == "jpeglib2 - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jpegtran.c
DEP_CPP_JPEGT=\
	".\cderror.h"\
	".\cdjpeg.h"\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	".\Jversion.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jpegtran.obj" : $(SOURCE) $(DEP_CPP_JPEGT) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jquant1.c
DEP_CPP_JQUAN=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jquant1.obj" : $(SOURCE) $(DEP_CPP_JQUAN) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jquant2.c
DEP_CPP_JQUANT=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jquant2.obj" : $(SOURCE) $(DEP_CPP_JQUANT) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jutils.c
DEP_CPP_JUTIL=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jutils.obj" : $(SOURCE) $(DEP_CPP_JUTIL) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jcapimin.c
DEP_CPP_JCAPIM=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jcapimin.obj" : $(SOURCE) $(DEP_CPP_JCAPIM) "$(INTDIR)"


# End Source File
################################################################################
# Begin Source File

SOURCE=.\Jmemmgr.c
DEP_CPP_JMEMM=\
	".\Jconfig.h"\
	".\Jerror.h"\
	".\Jinclude.h"\
	".\Jmemsys.h"\
	".\Jmorecfg.h"\
	".\Jpegint.h"\
	".\Jpeglib.h"\
	{$(INCLUDE)}"\sys\types.h"\
	

"$(INTDIR)\Jmemmgr.obj" : $(SOURCE) $(DEP_CPP_JMEMM) "$(INTDIR)"


# End Source File
# End Target
# End Project
################################################################################
