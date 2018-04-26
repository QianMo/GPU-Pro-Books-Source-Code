CONFIG += precompile_header
QT += opengl

PRECOMPILED_HEADER = stable.h

HEADERS = \
    stable.h \
    mainwindow.h \
    glview.h \
    glslmgr.h \
    GLee.h

SOURCES = \ 
    main.cpp \
    mainwindow.cpp \
    glview.cpp \
    glslmgr.cpp \
    GLee.c \
    perlin_original.cpp

FORMS = \
    mainwindow.ui \
    glslmgr.ui

RESOURCES = resources.qrc

win32 {
    RC_FILE = jkyprian-akf.rc
    DEFINES += _CRT_SECURE_NO_WARNINGS _BIND_TO_CURRENT_VCLIBS_VERSION=1
    exists( "$$(PROGRAMFILES)/QuickTime SDK/CIncludes" ) {
        DEFINES += HAVE_QUICKTIME
        INCLUDEPATH += "$$(PROGRAMFILES)/QuickTime SDK/CIncludes"
        LIBS += "$$(PROGRAMFILES)/QuickTime SDK/Libraries/QTMLClient.lib" Advapi32.lib
        QMAKE_LFLAGS += /NODEFAULTLIB:libcmt
        HEADERS += quicktime.h
        SOURCES += quicktime.cpp
    }
}

mac {
    ICON = jkyprian-akf.icns
    DEFINES += HAVE_QUICKTIME
    LIBS += -framework QuickTime
    HEADERS += quicktime.h
    SOURCES += quicktime.cpp
}
