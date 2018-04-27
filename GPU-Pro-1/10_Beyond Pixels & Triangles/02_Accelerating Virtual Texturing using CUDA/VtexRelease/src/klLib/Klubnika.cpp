/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#include "shared.h"
#include <stdarg.h>
#include "Klubnika.h"
#include "ConsoleWindow.h"

klCachedGlState glState;

void klExitHardwareNotSupported(void) {
    klFatalError("Your graphics driver does not support all necessary features.\n"
                 "Try again after installing the latest drivers form your vendor.");
}

void klCgToWxError(CGcontext ctx, CGerror err,void * /*appdata*/ ) {
	klLog("Cg Error: %s", cgGetErrorString(err));
	if ( err == CG_COMPILER_ERROR ) {
		const char *listing = cgGetLastListing(ctx);
		if ( listing != NULL ) {
			klLog("Cg Compiler Error(s):\n%s", listing);
		}
	}
}

void klCheckGlErrors(void) {
    int error = glGetError();
    while (error != GL_NO_ERROR) {
        assert(false);
        char *msg;
        switch (error) {
            case GL_INVALID_ENUM: 
                msg = "GL_INVALID_ENUM: An unacceptable	value is specified for an enumerated argument.";
                break;
            case GL_INVALID_VALUE:
                msg = "GL_INVALID_VALUE: A numeric argument is out of range.";
                break;
            case GL_INVALID_OPERATION:
                msg = "GL_INVALID_OPERATION: The specified operation is not allowed	in the current state.";
                break;
            case GL_STACK_OVERFLOW:
                msg = "GL_STACK_OVERFLOW: This command would cause a stack overflow.";
                break;
            case GL_STACK_UNDERFLOW:
                msg = "GL_STACK_OVERFLOW: This command would cause a stack underflow.";
                break;
            case GL_OUT_OF_MEMORY:
                msg = "GL_OUT_OF_MEMORY: There is not enough memory left to	execute	the command.";
                break;
            default:
                msg = "Unknown error...";
        }
        klLog(msg);        
        error = glGetError();
    }
}

FILE *logFile = NULL;

bool klInit(void) {
    logFile = fopen("klLog.txt","wb");
    console.init();  
	klLog("*** Privyet Mir ***");
	cgSetErrorHandler(klCgToWxError, NULL);
	return true;
}

bool klShutDown(void) {
	klLog("*** Poka Mir ***");
    fclose(logFile);
	return true;
}

void klPrint(const char *buffer) {
    if ( logFile ) {
        fprintf(logFile,buffer);
    }
    ConsoleWindow::AppendText(buffer);
}

void klLog(const char *format,...) {
    va_list args;
    int     len;
    char    *buffer;

    // Retrieve the variable arguments
    va_start( args, format );

    // Calculate number of chars needed, one \n and one null term
    len = _vscprintf( format, args ) + 2;
    
    // Alloc on stack
    buffer = (char *)_alloca(len * sizeof(char));

    // Write out
    vsprintf( buffer, format, args );

    // Append newline at end
    buffer[len-2] = '\n';
    buffer[len-1] = 0;

    klPrint(buffer);
}

void klFatalError(const char *format,...) {
    va_list args;
    int     len;
    char    *buffer;

    // Retrieve the variable arguments
    va_start( args, format );

    // Calculate number of chars needed
    len = _vscprintf( format, args ) + 1;
    
    // Alloc on stack
    buffer = (char *)_alloca(len * sizeof(char));

    // Write out
    vsprintf( buffer, format, args );

    klPrint("Fatal Error: ");
    klPrint(buffer);
    klPrint("\n");

    ConsoleWindow::FatalError();
}

void klError(const char *format,...) {
    va_list args;
    int     len;
    char    *buffer;

    // Retrieve the variable arguments
    va_start( args, format );

    // Calculate number of chars needed
    len = _vscprintf( format, args ) + 1;
    
    // Alloc on stack
    buffer = (char *)_alloca(len * sizeof(char));

    // Write out
    vsprintf( buffer, format, args );

    klPrint("Error: ");
    klPrint(buffer);
    klPrint("\n");
}
