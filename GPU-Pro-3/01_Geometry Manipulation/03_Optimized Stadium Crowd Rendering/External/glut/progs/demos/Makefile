# Makefile for Win32
# Copyright (c) Mark J. Kilgard, 1997, 1998.

# defines
SUBDIRS	= atlantis.dir bluepony.dir bounce.dir chess.dir geoface.dir glflare.dir gliq.dir glutmech.dir ideas.dir lorenz.dir opengl_logo.dir rollercoaster.dir smooth.dir underwater.dir walker.dir yacme.dir skyfly.dir particle.dir newave.dir
CLEAN	= $(SUBDIRS:.dir=.clean)
CLOBBER	= $(SUBDIRS:.dir=.clobber)

# default rule
default	: $(SUBDIRS)

# cleanup rules
clean	: $(CLEAN)
clobber	: $(CLOBBER)


# inference rules
$(SUBDIRS)	:
	@echo.
	@echo Making in $* subdirectory...
	@cd $*
	@nmake -f Makefile.win -nologo
	@cd ..

$(CLEAN)	:
	@del *~
	@echo.
	@echo Cleaning in $* subdirectory...
	@cd $*
	@nmake -f Makefile.win -nologo clean
	@cd ..

$(CLOBBER)	:
	@echo.
	@echo Clobbering in $* subdirectory...
	@cd $*
	@nmake -f Makefile.win -nologo clobber
	@cd ..

