##
##  Makefile for Windows NT
##


# defines
SUBDIRS	= glut.dir mui.dir gle.dir glsmap.dir
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


