# Makefile for Win32

# defines
SUBDIRS	= advanced.dir contrib.dir demos.dir examples.dir mesademos.dir mui.dir redbook.dir texfont.dir gle.dir gameglut.dir bucciarelli.dir advanced97.dir spheremap.dir
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


