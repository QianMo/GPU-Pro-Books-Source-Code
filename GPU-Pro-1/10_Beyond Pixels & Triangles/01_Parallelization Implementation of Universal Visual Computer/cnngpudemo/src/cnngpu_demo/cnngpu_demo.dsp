# Microsoft Developer Studio Project File - Name="cnngpu_demo" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=cnngpu_demo - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "cnngpu_demo.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "cnngpu_demo.mak" CFG="cnngpu_demo - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "cnngpu_demo - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "cnngpu_demo - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "cnngpu_demo - Win32 Release"

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
# ADD BASE CPP /nologo /W3 /GX /O2 /I "../common" /I "../g_common" /I "../../lib/glut/include" /I "../../lib/glew/include" /I "../../lib/cgsdk/include" /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /I "../common" /I "../g_common" /I "../../lib/glut/include" /I "../../lib/glew/include" /I "../../lib/cgsdk/include" /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /FR /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 glew32.lib cg.lib cgGL.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386 /libpath:"../../lib/glut/lib" /libpath:"../../lib/glew/lib" /libpath:"../../lib/cgsdk/lib"
# ADD LINK32 glew32.lib cg.lib cgGL.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386 /libpath:"../../lib/glut/lib" /libpath:"../../lib/glew/lib" /libpath:"../../lib/cgsdk/lib"

!ELSEIF  "$(CFG)" == "cnngpu_demo - Win32 Debug"

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
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /I "../common" /I "../g_common" /I "../../lib/glut/include" /I "../../lib/glew/include" /I "../../lib/cgsdk/include" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /I "../common" /I "../g_common" /I "../../lib/glut/include" /I "../../lib/glew/include" /I "../../lib/cgsdk/include" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /FR /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 glew32.lib cg.lib cgGL.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept /libpath:"../../lib/glut/lib" /libpath:"../../lib/glew/lib" /libpath:"../../lib/cgsdk/lib"
# ADD LINK32 glew32.lib cg.lib cgGL.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept /libpath:"../../lib/glut/lib" /libpath:"../../lib/glew/lib" /libpath:"../../lib/cgsdk/lib"

!ENDIF 

# Begin Target

# Name "cnngpu_demo - Win32 Release"
# Name "cnngpu_demo - Win32 Debug"
# Begin Group "GarySrc"

# PROP Default_Filter ""
# Begin Source File

SOURCE=..\g_common\g_bell.cpp
# End Source File
# Begin Source File

SOURCE=..\g_common\g_bell.h
# End Source File
# Begin Source File

SOURCE=..\g_common\g_common.cpp
# End Source File
# Begin Source File

SOURCE=..\g_common\g_common.h
# End Source File
# Begin Source File

SOURCE=..\g_common\g_pfm.cpp
# End Source File
# Begin Source File

SOURCE=..\g_common\g_pfm.h
# End Source File
# Begin Source File

SOURCE=..\g_common\g_vector.cpp
# End Source File
# Begin Source File

SOURCE=..\g_common\g_vector.h
# End Source File
# End Group
# Begin Source File

SOURCE=.\cnn_utility.cpp
# End Source File
# Begin Source File

SOURCE=.\cnn_utility.h
# End Source File
# Begin Source File

SOURCE=.\cnngpu_demo.cpp
# End Source File
# Begin Source File

SOURCE=.\shader.cg
# End Source File
# Begin Source File

SOURCE=.\shader.cpp
# End Source File
# Begin Source File

SOURCE=.\shader.h
# End Source File
# End Target
# End Project
