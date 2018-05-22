# Makefile for Win32

# defines
SUBDIRS	= lib.dir test.dir progs.dir
CLEAN	= $(SUBDIRS:.dir=.clean)
CLOBBER	= $(SUBDIRS:.dir=.clobber)

# default rule
default	: $(SUBDIRS)

# cleanup rules
clean	: $(CLEAN)
clobber	: $(CLOBBER)

RM = -rm -rf
DIST_NAME = glut-3.7
ZIP = zip
ZIP_NAME = glut37
ZIP_DATA_NAME = glutdata

FIX_FILES = ../util/dos2unix/FixFiles

DIST = README NOTICE README.glut2 README.fortran README.xinput \
       README.inventor README.irix64bit CHANGES Imakefile Makefile.sgi \
       mkmkfiles.win \
       mkmkfiles.imake mkmkfiles.sgi Glut.cf lib test progs man \
       README.ibm-shlib README.irix6 FAQ.glut README.glut3 README.linux \
       linux README.man Makefile IAFA-PACKAGE README.mesa README.ada \
       include adainclude glutdefs README.win README.mui \
       glutmake.bat glutwin32.mak Makefile.win Portability.txt

tree_clobber:
	nmake clobber
	$(RM) disttest $(DIST_NAME) glut.stage1.tar
	find . -name '*.bak' -print | xargs /bin/rm -f
	find . -name '*.obj' -print | xargs /bin/rm -f
	find . -name '*.exe' -print | xargs /bin/rm -f
	find . -name '*.swp' -print | xargs /bin/rm -f
	find . -name 'deleted' -print | xargs /bin/rm -f

DATA_DIST = $(DIST_NAME)/data $(DIST_NAME)/progs/advanced97/data $(DIST_NAME)/progs/advanced97/flame $(DIST_NAME)/progs/advanced97/skull

tar:
	cd util\dos2unix
	nmake /f Makefile.win
	cd ..\..
	$(RM) disttest $(DIST_NAME) glut.stage1.tar glut.tar.gz glut_data.tar.gz
	tar cvf glut.stage1.tar $(DIST) data
	$(RM) $(DIST_NAME)
	mkdir $(DIST_NAME)
	cd $(DIST_NAME)
	tar xvmf ../glut.stage1.tar
	nmake /f Makefile.win tree_clobber
	$(RM) progs/advanced97/skull/skull3d.tiff
	tcsh -c "find . -name 'README' -print | xargs $(FIX_FILES)"
	tcsh -c "$(FIX_FILES) progs/*/*.c++ mkfiles/* linux/*"
	tcsh -c "$(FIX_FILES) README.* CHANGES NOTICE Glut.cf IAFA-PACKAGE glutdefs Portability.txt mkmkfiles.sgi mkmkfiles.imake"
	chmod 755 mkmkfiles.sgi mkmkfiles.imake
	tcsh -c "find . -name '*.c' -print | xargs $(FIX_FILES)"
	tcsh -c "find . -name '*.f' -print | xargs $(FIX_FILES)"
	tcsh -c "find . -name '*.h' -print | xargs $(FIX_FILES)"
	tcsh -c "find . -name '*.man' -print | xargs $(FIX_FILES)"
	tcsh -c "find . -name '*.ads' -print | xargs $(FIX_FILES)"
	tcsh -c "find . -name '*.adb' -print | xargs $(FIX_FILES)"
	tcsh -c "find . -name 'Makefile.sgi' -print | xargs $(FIX_FILES)"
	tcsh -c "find . -name 'Imakefile' -print | xargs $(FIX_FILES)"
	tcsh -c "find . -name 'ObjectType.mk' -print | xargs $(FIX_FILES)"
	cd ..
	tar cvf glut_data.tar $(DATA_DIST)
	$(ZIP) -r $(ZIP_NAME)data.zip $(DATA_DIST)
	$(RM) $(DATA_DIST)
	tar cvf glut.tar $(DIST_NAME)
	$(ZIP) -r $(ZIP_NAME).zip $(DIST_NAME)
	$(RM) $(DIST_NAME) glut.stage1.tar
	gzip -fv glut.tar
	gzip -fv glut_data.tar

zip:
	$(RM) disttest $(DIST_NAME) glut.stage1.tar $(ZIP_NAME).zip
	tar cvf glut.stage1.tar $(DIST)
	$(RM) $(DIST_NAME)
	mkdir $(DIST_NAME)
	cd $(DIST_NAME)
	tar xvmf ../glut.stage1.tar
	$(RM) progs/advanced97/skull/skull3d.tiff
	cd ..
	$(ZIP) -r $(ZIP_NAME).zip $(DIST_NAME)

testdist:
	$(RM) disttest
	mkdir disttest
	cd disttest
	cp ../glut.tar.gz .
	gunzip glut.tar.gz
	tar xvmf glut.tar
	cd $(DIST_NAME)
#	./mkmkfiles.win
	nmake /f Makefile.win
	cd test
	nmake /f Makefile.win test
	cd ../../..
	$(RM) disttest

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


