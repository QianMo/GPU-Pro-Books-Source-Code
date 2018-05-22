include(qmake_common_functions.pri)

TEMPLATE = vcapp
TARGET = SSBentCones
CONFIG = debug_and_release console exceptions stl
CONFIG += opengl
DESTDIR = ./


INCLUDEPATH += ./include ./include/nv ./include/glm-0.9.0.5 $(WindowsSdkDir)/include $(VCInstallDir)/include

HEADERS = $$getHeadersFromDir(src) $$getHeadersFromDir(include\\NV)
SOURCES = $$getSourcesFromDir(src) $$getSourcesFromDir(include\\NV) $$getFilesFromDir(shaders)

QMAKE_LIBDIR += lib $(WindowsSdkDir)/lib $(VCInstallDir)/lib

LIBS *= glew32.lib glut32.lib
