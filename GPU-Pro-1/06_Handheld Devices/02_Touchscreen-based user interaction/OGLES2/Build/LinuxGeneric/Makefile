#
# File			Makefile
# Title			Demo makefile
# Author		Auto-generated
# Date			21/11/2008
#
# Copyright		Copyright 2007 by Imagination Technologies Limited.
#

.PHONY: clean

SDKDIR  = ../../../../../SDKPackage
OUTNAME = OGLES2IntroducingPODTouch

OBJECTS += tex_base.o tex_arm.o FragShader.o VertShader.o Scene.o MDKCamera.o MDKMath.o MDKPrecisionTimer.o MDKTouch.o

include $(SDKDIR)/Builds/OGLES2/LinuxGeneric/make_demo.mak



ifeq ("$(PLATFORM)","LinuxOMAP3")		
	ifneq ("$(TS_SUPPORT)","")
		PLAT_CFLAGS += -DPVRSHELL_OMAP3_TS_SUPPORT
		INCLUDES += -I$(TSLIB_INCLUDE)
		LINK += -L$(TSLIB_LIB) -lts
	endif
	ifeq ("$(TS_SUPPORT)","1")
		PLAT_CFLAGS += -DDEVICE_ZOOM1
	endif
	ifeq ("$(TS_SUPPORT)","2")
		PLAT_CFLAGS += -DDEVICE_ZOOM2
	endif
endif

# Checks on required variables
ifneq ("$(TS_SUPPORT)","")
	ifeq ("$(TSLIB_INCLUDE)", "") 	
		ERROR := error TSLIB_INCLUDE not specified!
	endif
	ifeq ("$(TSLIB_LIB)", "")	
		ERROR := error TSLIB_LIB not specified!
	endif
endif
ifeq ("$(LIBDIR)", "")
	ERROR := error LIBDIR not specified!
endif
ifeq ("$(PLATFORM)", "")	
	ERROR := error PLATFORM not specified!
endif


all:
	$(ERROR)
	
../../Content/tex_base.cpp ../../Content/tex_arm.cpp ../../Content/FragShader.cpp ../../Content/VertShader.cpp ../../Content/Scene.cpp: build_textures_and_shaders 

%.o: ../../Content/%.cpp

	$(CC) -c $(COMMON_INCLUDES) $(PLAT_INCLUDES) $(INCLUDES) $(COMMON_CFLAGS) $(PLAT_CFLAGS) $(CFLAGS) $^ -o$@


clean:
	-rm *.o
	make -C ../.. -f content.mak clean
	

	



