##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## debug
ProjectName            :=quadtree_simple
ConfigurationName      :=debug
WorkspacePath          := "C:\Users\jdup\Desktop\jdupuy-ltree-22f5e3ff8f99\build\codelite.windows"
ProjectPath            := "C:\Users\jdup\Desktop\jdupuy-ltree-22f5e3ff8f99\build\codelite.windows"
IntermediateDirectory  :=obj/debug/quadtree_simple
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=jdup
Date                   :=03/08/2013
CodeLitePath           :="C:\Program Files (x86)\CodeLite"
LinkerName             :=gcc
SharedObjectLinkerName :=gcc -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.o.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=quadtree_simple.exe
Preprocessors          :=$(PreprocessorSwitch)GLFW_NO_GLU $(PreprocessorSwitch)DEBUG $(PreprocessorSwitch)WINDOWS 
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E 
ObjectsFileList        :="quadtree_simple.txt"
PCHCompileFlags        :=
MakeDirCommand         :=makedir
RcCmpOptions           := 
RcCompilerName         :=windres
LinkOptions            :=  
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch)../../lib $(IncludeSwitch)../../lib/libpng $(IncludeSwitch)../../lib/GLFW 
IncludePCH             := 
RcIncludePath          := 
Libs                   := $(LibrarySwitch)C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/lib/GLFW/libglfw3 $(LibrarySwitch)C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/lib/libpng/zlib.lib $(LibrarySwitch)C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/lib/libpng/libpng14.lib $(LibrarySwitch)opengl32 $(LibrarySwitch)gdi32 $(LibrarySwitch)user32 $(LibrarySwitch)kernel32 
ArLibs                 :=  "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/lib/GLFW/libglfw3.a" "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/lib/libpng/zlib.lib" "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/lib/libpng/libpng14.lib" "opengl32" "gdi32" "user32" "kernel32" 
LibPath                := $(LibraryPathSwitch). $(LibraryPathSwitch)C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/lib/GLFW $(LibraryPathSwitch)C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/lib/libpng 

##
## Common variables
## AR, CXX, CC, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := ar rcus
CXX      := gcc
CC       := gcc
CXXFLAGS :=  -Wall -msse -msse2 -g $(Preprocessors)
CFLAGS   :=  -Wall -msse -msse2 -g $(Preprocessors)


##
## User defined environment variables
##
CodeLiteDir:=C:\Program Files (x86)\CodeLite
UNIT_TEST_PP_SRC_DIR:=C:\UnitTest++-1.3
Objects0=$(IntermediateDirectory)/src_affine$(ObjectSuffix) $(IntermediateDirectory)/src_bstrlib$(ObjectSuffix) $(IntermediateDirectory)/src_buf$(ObjectSuffix) $(IntermediateDirectory)/src_frustum$(ObjectSuffix) $(IntermediateDirectory)/src_ft$(ObjectSuffix) $(IntermediateDirectory)/src_glload$(ObjectSuffix) $(IntermediateDirectory)/src_image$(ObjectSuffix) $(IntermediateDirectory)/src_program$(ObjectSuffix) $(IntermediateDirectory)/src_quadtree$(ObjectSuffix) $(IntermediateDirectory)/src_timer$(ObjectSuffix) \
	

Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

$(IntermediateDirectory)/.d:
	@$(MakeDirCommand) "obj/debug/quadtree_simple"

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/src_affine$(ObjectSuffix): ../../src/affine.c $(IntermediateDirectory)/src_affine$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/affine.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_affine$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_affine$(DependSuffix): ../../src/affine.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_affine$(ObjectSuffix) -MF$(IntermediateDirectory)/src_affine$(DependSuffix) -MM "../../src/affine.c"

$(IntermediateDirectory)/src_affine$(PreprocessSuffix): ../../src/affine.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_affine$(PreprocessSuffix) "../../src/affine.c"

$(IntermediateDirectory)/src_bstrlib$(ObjectSuffix): ../../src/bstrlib.c $(IntermediateDirectory)/src_bstrlib$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/bstrlib.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_bstrlib$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_bstrlib$(DependSuffix): ../../src/bstrlib.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_bstrlib$(ObjectSuffix) -MF$(IntermediateDirectory)/src_bstrlib$(DependSuffix) -MM "../../src/bstrlib.c"

$(IntermediateDirectory)/src_bstrlib$(PreprocessSuffix): ../../src/bstrlib.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_bstrlib$(PreprocessSuffix) "../../src/bstrlib.c"

$(IntermediateDirectory)/src_buf$(ObjectSuffix): ../../src/buf.c $(IntermediateDirectory)/src_buf$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/buf.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_buf$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_buf$(DependSuffix): ../../src/buf.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_buf$(ObjectSuffix) -MF$(IntermediateDirectory)/src_buf$(DependSuffix) -MM "../../src/buf.c"

$(IntermediateDirectory)/src_buf$(PreprocessSuffix): ../../src/buf.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_buf$(PreprocessSuffix) "../../src/buf.c"

$(IntermediateDirectory)/src_frustum$(ObjectSuffix): ../../src/frustum.c $(IntermediateDirectory)/src_frustum$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/frustum.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_frustum$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_frustum$(DependSuffix): ../../src/frustum.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_frustum$(ObjectSuffix) -MF$(IntermediateDirectory)/src_frustum$(DependSuffix) -MM "../../src/frustum.c"

$(IntermediateDirectory)/src_frustum$(PreprocessSuffix): ../../src/frustum.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_frustum$(PreprocessSuffix) "../../src/frustum.c"

$(IntermediateDirectory)/src_ft$(ObjectSuffix): ../../src/ft.c $(IntermediateDirectory)/src_ft$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/ft.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_ft$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_ft$(DependSuffix): ../../src/ft.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_ft$(ObjectSuffix) -MF$(IntermediateDirectory)/src_ft$(DependSuffix) -MM "../../src/ft.c"

$(IntermediateDirectory)/src_ft$(PreprocessSuffix): ../../src/ft.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_ft$(PreprocessSuffix) "../../src/ft.c"

$(IntermediateDirectory)/src_glload$(ObjectSuffix): ../../src/glload.c $(IntermediateDirectory)/src_glload$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/glload.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_glload$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_glload$(DependSuffix): ../../src/glload.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_glload$(ObjectSuffix) -MF$(IntermediateDirectory)/src_glload$(DependSuffix) -MM "../../src/glload.c"

$(IntermediateDirectory)/src_glload$(PreprocessSuffix): ../../src/glload.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_glload$(PreprocessSuffix) "../../src/glload.c"

$(IntermediateDirectory)/src_image$(ObjectSuffix): ../../src/image.c $(IntermediateDirectory)/src_image$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/image.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_image$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_image$(DependSuffix): ../../src/image.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_image$(ObjectSuffix) -MF$(IntermediateDirectory)/src_image$(DependSuffix) -MM "../../src/image.c"

$(IntermediateDirectory)/src_image$(PreprocessSuffix): ../../src/image.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_image$(PreprocessSuffix) "../../src/image.c"

$(IntermediateDirectory)/src_program$(ObjectSuffix): ../../src/program.c $(IntermediateDirectory)/src_program$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/program.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_program$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_program$(DependSuffix): ../../src/program.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_program$(ObjectSuffix) -MF$(IntermediateDirectory)/src_program$(DependSuffix) -MM "../../src/program.c"

$(IntermediateDirectory)/src_program$(PreprocessSuffix): ../../src/program.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_program$(PreprocessSuffix) "../../src/program.c"

$(IntermediateDirectory)/src_quadtree$(ObjectSuffix): ../../src/quadtree.c $(IntermediateDirectory)/src_quadtree$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/quadtree.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_quadtree$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_quadtree$(DependSuffix): ../../src/quadtree.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_quadtree$(ObjectSuffix) -MF$(IntermediateDirectory)/src_quadtree$(DependSuffix) -MM "../../src/quadtree.c"

$(IntermediateDirectory)/src_quadtree$(PreprocessSuffix): ../../src/quadtree.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_quadtree$(PreprocessSuffix) "../../src/quadtree.c"

$(IntermediateDirectory)/src_timer$(ObjectSuffix): ../../src/timer.c $(IntermediateDirectory)/src_timer$(DependSuffix)
	$(CC) $(SourceSwitch) "C:/Users/jdup/Desktop/jdupuy-ltree-22f5e3ff8f99/src/timer.c" $(CFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_timer$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_timer$(DependSuffix): ../../src/timer.c
	@$(CC) $(CFLAGS) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_timer$(ObjectSuffix) -MF$(IntermediateDirectory)/src_timer$(DependSuffix) -MM "../../src/timer.c"

$(IntermediateDirectory)/src_timer$(PreprocessSuffix): ../../src/timer.c
	@$(CC) $(CFLAGS) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_timer$(PreprocessSuffix) "../../src/timer.c"


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) $(IntermediateDirectory)/src_affine$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_affine$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_affine$(PreprocessSuffix)
	$(RM) $(IntermediateDirectory)/src_bstrlib$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_bstrlib$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_bstrlib$(PreprocessSuffix)
	$(RM) $(IntermediateDirectory)/src_buf$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_buf$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_buf$(PreprocessSuffix)
	$(RM) $(IntermediateDirectory)/src_frustum$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_frustum$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_frustum$(PreprocessSuffix)
	$(RM) $(IntermediateDirectory)/src_ft$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_ft$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_ft$(PreprocessSuffix)
	$(RM) $(IntermediateDirectory)/src_glload$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_glload$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_glload$(PreprocessSuffix)
	$(RM) $(IntermediateDirectory)/src_image$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_image$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_image$(PreprocessSuffix)
	$(RM) $(IntermediateDirectory)/src_program$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_program$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_program$(PreprocessSuffix)
	$(RM) $(IntermediateDirectory)/src_quadtree$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_quadtree$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_quadtree$(PreprocessSuffix)
	$(RM) $(IntermediateDirectory)/src_timer$(ObjectSuffix)
	$(RM) $(IntermediateDirectory)/src_timer$(DependSuffix)
	$(RM) $(IntermediateDirectory)/src_timer$(PreprocessSuffix)
	$(RM) $(OutputFile)
	$(RM) $(OutputFile).exe
	$(RM) ".build-debug/quadtree_simple"


