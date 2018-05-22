/********************************************************************
glsl.cpp
Version: 1.0.0_rc5
Last update: 2006/11/12 (Geometry Shader Support)

(c) 2003-2006 by Martin Christen. All Rights reserved.
*********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "glsl.h"

using namespace std;
using namespace cwc;

bool    useGLSL = false;
bool    extensions_init = false;
bool    bGeometryShader = false;
bool    bGPUShader4 = false;

// -----------------------------------------------------------------------------

/*! \mainpage
\section s_intro Introduction
This is libglsl - a collection of helper classes to load, compile, link and activate shaders
written in the OpenGL Shading language. Vertex Shaders, Geometry Shaders and Fragment shaders
are supported (if the hardware is capable of supporting them, of course).

   Version info: \ref s_libglslnews

   \section s_examples Examples

   \subsection Loading Vertex and Fragment Shader using Shader Manager.


Initialization:
\verbatim
   glShaderManager SM;
   glShader *shader = SM.loadfromFile("test.vert","test.frag");
   if (shader==0)
      cout << "Error Loading, compiling or linking shader\n";

Render:
   shader->begin();
   shader->setUniform1f("MyFloat", 1.123);
     glutDrawSolidSphere(1.0);
   shader->end();
\endverbatim

\subsection geom_shader Geometry Shader
The easiest way to use Geometry Shaders is through the
Shadermanager.
Initialization:
\verbatim
   SM.SetInputPrimitiveType(GL_TRIANGLES);			// one of: GL_POINTS, GL_LINES, GL_LINES_ADJACENCY_EXT, GL_TRIANGLES, GL_TRIANGLES_ADJACENCY_EXT
   SM.SetOutputPrimitiveType(GL_TRIANGLE_STRIP);	// one of: GL_POINTS, GL_LINE_STRIP, GL_TRIANGLE_STRIP
   SM.SetVerticesOut(3);
   glShader *shader = SM.loadfromFile("test.vert","test.geom","test.frag");
\endverbatim

 */

namespace cwc
{
    // -------------------------------------------------------------------------

    // Error, Warning and Info Strings
    char   *aGLSLStrings[] = {
        (char *) "[e00] GLSL is not available!",
        (char *) "[e01] Not a valid program object!",
        (char *) "[e02] Not a valid object!",
        (char *) "[e03] Out of memory!",
        (char *) "[e04] Unknown compiler error!",
        (char *) "[e05] Linker log is not available!",
        (char *) "[e06] Compiler log is not available!",
        (char *) "[Empty]"
    };

    // -------------------------------------------------------------------------

    // GL ERROR CHECK
    void    CheckGLError (char *file, int line)
    {
        GLenum  glErr = GL_NO_ERROR;

        while ((glErr = glGetError ()) != GL_NO_ERROR)
        {
            const GLubyte *sError = gluErrorString (glErr);
#if 1
            if (sError)
                cout << "GL Error #" << glErr << " (" << sError << ") in File " << file << " at line: " << line << endl;
            else
                cout << "GL Error #" << glErr << " (no message available) in File " << file << " at line: " << line << endl;
#else
            bool valid_error = true;
            if (! strcmp ((const char *) sError, "stack underflow")) valid_error = false;
            if (! strcmp ((const char *) sError, "stack overflow")) valid_error = false;

            if (valid_error)
            {
            if (sError)
                cout << "GL Error #" << glErr << " (" << sError << ") in File " << file << " at line: " << line << endl;
            else
                cout << "GL Error #" << glErr << " (no message available) in File " << file << " at line: " << line << endl;
            }
#endif
        }
    }

    GLenum  CheckGLErrorOutOfMemory (void)
    {
        GLenum  glErr_prev, glErr = GL_NO_ERROR;
		glErr_prev = glErr;

        while ((glErr = glGetError ()) != GL_NO_ERROR)
			glErr_prev = glErr;

		return glErr_prev;
    }

    void    CheckGLErrorNoPrint (void)
    {
        GLenum  glErr;

        while ((glErr = glGetError ()) != GL_NO_ERROR) ;
    }

    // -------------------------------------------------------------------------

    bool    InitOpenGLExtensions (void)
    {
        if (extensions_init) return true;
        extensions_init = true;

#ifdef WIN32
        GLenum  err = glewInit ();
        if (err != GLEW_OK)
        {
            cout << "Error:" << glewGetErrorString (err) << endl;
            extensions_init = false;
            return false;
        }
#endif

// #ifdef NVIDIA
#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

		GLint total_mem_kb = 0;
		glGetIntegerv (GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);

		GLint cur_avail_mem_kb = 0;
		glGetIntegerv (GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);
// #endif
#ifdef ATI
		UINT n = wglGetGPUIDsAMD (0, 0);
		UINT *ids = new UINT[n];
		size_t total_mem_mb = 0;
		wglGetGPUIDsAMD (n, ids);
		wglGetGPUInfoAMD (ids[0], WGL_GPU_RAM_AMD, GL_UNSIGNED_INT, sizeof(size_t), &total_mem_mb);
#endif

        cout << "OpenGL Vendor: "   << (char *) glGetString (GL_VENDOR) << "\n";
        cout << "OpenGL Renderer: " << (char *) glGetString (GL_RENDERER) << 
			"\n    with " << total_mem_kb / 1024.0f << "MB of total memory. "
			"Currently available " << cur_avail_mem_kb / 1024.0f << "MB" << "\n";
        cout << "OpenGL Version: "  << (char *) glGetString (GL_VERSION) << "\n";
     // cout << "OpenGL Extensions:\n" << (char *) glGetString (GL_EXTENSIONS) << "\n";

        HasGLSLSupport ();

		// check for individual required extensions
#ifdef WIN32
        if (! glewGetExtension ("GL_EXT_timer_query"))
#else
        if (strstr ((const char *) glGetString (GL_EXTENSIONS), "GL_EXT_timer_query") == NULL)
#endif
		{
            cout << "[WARNING] GL_EXT_timer_query extension is not available!\n";
            return false;
		}

#ifdef WIN32
        if (! glewGetExtension ("GL_ARB_debug_output"))
#else
        if (strstr ((const char *) glGetString (GL_EXTENSIONS), "GL_ARB_debug_output") == NULL)
#endif
		{
            cout << "[WARNING] GL_ARB_debug_output extension is not available!\n";
            return false;
		}

		cout << endl;

        return true;
    }

    bool    HasGLSLSupport (void)
    {
        bGeometryShader = HasGeometryShaderSupport ();
        bGPUShader4 = HasShaderModel4 ();

        if (useGLSL)
            return true;        // already initialized and GLSL is available
        useGLSL = true;

        if (! extensions_init)
            InitOpenGLExtensions (); // extensions were not yet initialized!!

        if (bGPUShader4)
        {
            cout << "OpenGL Shader Model 4.0 is available!" << endl;
            if (bGeometryShader)
            {
                cout << "OpenGL Geometry shaders are available!" << endl;
            }
        }
		else
		{
			cout << "Shader Model 4.0 is not available!" << endl << "Exiting ..." << endl;
			cout << "Press any key to exit." << endl; getchar ();
			exit (EXIT_FAILURE);
		}

        if (! glewGetExtension ("GL_ARB_fragment_shader"))
        {
            cout << "[WARNING] GL_ARB_fragment_shader extension is not available!\n";
            useGLSL = false;
        }

        if (! glewGetExtension ("GL_ARB_vertex_shader"))
        {
            cout << "[WARNING] GL_ARB_vertex_shader extension is not available!\n";
            useGLSL = false;
        }

        if (! glewGetExtension ("GL_ARB_shader_objects"))
        {
            cout << "[WARNING] GL_ARB_shader_objects extension is not available!\n";
            useGLSL = false;
        }

        if (useGLSL)
        {
			cout << "OpenGL GLSL: " << glGetString (GL_SHADING_LANGUAGE_VERSION) << "\n";
        }
        else
        {
            cout << "[FAILED] OpenGL Shading Language is not available...\n";
        }

		cout << endl;

        return useGLSL;
    }

    bool    HasGeometryShaderSupport (void)
    {
#ifdef WIN32
        if (! glewGetExtension ("GL_EXT_geometry_shader4"))
#else
        if (strstr ((const char *) glGetString (GL_EXTENSIONS), "GL_EXT_geometry_shader4") == NULL)
#endif
            return false;

        return true;
    }

    bool    HasShaderModel4 (void)
    {
#ifdef WIN32
        if (! glewGetExtension ("GL_EXT_gpu_shader4"))
#else
        if (strstr ((const char *) glGetString (GL_EXTENSIONS), "GL_EXT_gpu_shader4") == NULL)
#endif
            return false;

        return true;
    }
}

// -----------------------------------------------------------------------------

// ************************************************************************
// Implementation of glShader class
// ************************************************************************

glShader::glShader ()
{
    InitOpenGLExtensions ();

    ProgramObject = 0;
    linker_log = 0;
    is_linked = false;
    _mM = false;
    _noshader = true;

    if (! useGLSL)
    {
        cout << "**ERROR: OpenGL Shading Language is NOT available!" << endl;
    }
    else
    {
        ProgramObject = glCreateProgram ();
    }
}

// -----------------------------------------------------------------------------

glShader::~glShader ()
{
    if (linker_log != 0)
        free (linker_log);

    if (useGLSL)
    {
        for (unsigned int i = 0; i < ShaderList.size (); i++)
        {
            glDetachShader (ProgramObject, ShaderList[i]->ShaderObject);
            CHECK_GL_ERROR ();  // if you get an error here, you deleted the
                                // Program object first and then
            // the ShaderObject! Always delete ShaderObjects last!
            if (_mM)
                delete  ShaderList[i];
        }

        glDeleteShader (ProgramObject);
        CHECK_GL_ERROR ();
    }
}

// -----------------------------------------------------------------------------

void glShader::addShader (glShaderObject * ShaderProgram)
{
    if (! useGLSL) return;
    if (ShaderProgram == 0) return;

    if (! ShaderProgram->is_compiled)
    {
        cout << "**warning** please compile program before adding object! trying to compile now...\n";

        if (! ShaderProgram->compile ())
        {
            cout << "...compile ERROR!\n";
            return;
        }
        else
        {
            cout << "...ok!\n";
        }
    }

    ShaderList.push_back (ShaderProgram);
}

// -----------------------------------------------------------------------------

void glShader::SetInputPrimitiveType (int nInputPrimitiveType)
{
    _nInputPrimitiveType = nInputPrimitiveType;
}

void glShader::SetOutputPrimitiveType (int nOutputPrimitiveType)
{
    _nOutputPrimitiveType = nOutputPrimitiveType;
}

void glShader::SetVerticesOut (int nVerticesOut)
{
    _nVerticesOut = nVerticesOut;
}

// -----------------------------------------------------------------------------

bool glShader::link (void)
{
    if (! useGLSL) return false;

    if (_bUsesGeometryShader)
    {
        glProgramParameteriEXT (ProgramObject, GL_GEOMETRY_INPUT_TYPE_EXT,
                                _nInputPrimitiveType);
        glProgramParameteriEXT (ProgramObject, GL_GEOMETRY_OUTPUT_TYPE_EXT,
                                _nOutputPrimitiveType);
        glProgramParameteriEXT (ProgramObject, GL_GEOMETRY_VERTICES_OUT_EXT,
                                _nVerticesOut);
        CHECK_GL_ERROR ();
    }

    unsigned int i;

    if (is_linked)              // already linked, detach everything first
    {
        cout << "**warning** Object is already linked, trying to link again" << endl;
        for (i = 0; i < ShaderList.size (); i++)
        {
            glDetachShader (ProgramObject, ShaderList[i]->ShaderObject);
            CHECK_GL_ERROR ();
        }
    }

    for (i = 0; i < ShaderList.size (); i++)
    {
        glAttachShader (ProgramObject, ShaderList[i]->ShaderObject);
        CHECK_GL_ERROR ();
     // cout << "attaching ProgramObj [" << i << "] @ 0x" << hex << ShaderList[i]->ProgramObject << " in ShaderObj @ 0x" << ShaderObject << endl;
    }

    GLint   linked;             // bugfix Oct-06-2006

    glLinkProgram (ProgramObject);
    CHECK_GL_ERROR ();
    glGetProgramiv (ProgramObject, GL_LINK_STATUS, &linked);
    CHECK_GL_ERROR ();

    if (linked)
    {
        is_linked = true;
        return true;
    }
    else
    {
        cout << "**linker error**\n";
    }

    return false;
}

// -----------------------------------------------------------------------------
// Compiler Log: Ausgabe der Compiler Meldungen in String

GLint glShader::getLinkerLogLength (void)
{
    GLint   blen = 0;           // bugfix Oct-06-2006

    glGetProgramiv (ProgramObject, GL_INFO_LOG_LENGTH, &blen);
    CHECK_GL_ERROR ();

    return blen;
}

char   * glShader::getLinkerLog (void)
{
    if (! useGLSL) return aGLSLStrings[0];

    if (ProgramObject == 0)
        return aGLSLStrings[2];

    GLsizei slen = 0;           // bugfix Oct-06-2006
    GLint   blen = getLinkerLogLength ();

    if (blen > 1)
    {
        if (linker_log != 0)
        {
            free (linker_log);
            linker_log = 0;
        }
        if ((linker_log = (GLcharARB *) malloc (blen)) == NULL)
        {
            cout << "Error: could not allocate compiler_log buffer\n";
            return aGLSLStrings[3];
        }

        glGetProgramInfoLog (ProgramObject, blen, &slen, linker_log);
        CHECK_GL_ERROR ();
     // cout << "linker_log: \n", linker_log;
    }

    if (linker_log != 0)
        return (char *) linker_log;
    else
        return aGLSLStrings[5];

    return aGLSLStrings[4];
}

// -----------------------------------------------------------------------------

void glShader::begin (void)
{
    if (! useGLSL) return;
    if (ProgramObject == 0) return;
    if (!_noshader) return;

    if (is_linked)
    {
        glUseProgram (ProgramObject);
        CHECK_GL_ERROR();
    }
}

// -----------------------------------------------------------------------------

void glShader::end (void)
{
    if (! useGLSL) return;
    if (!_noshader) return;

    glUseProgram (0);
    CHECK_GL_ERROR();
}

// -----------------------------------------------------------------------------

bool glShader::setUniform1f (GLcharARB * varname, GLfloat v0, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform1f (loc, v0);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform2f (GLcharARB * varname, GLfloat v0, GLfloat v1, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform2f (loc, v0, v1);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform3f (GLcharARB * varname, GLfloat v0, GLfloat v1, GLfloat v2, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform3f (loc, v0, v1, v2);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform4f (GLcharARB * varname, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform4f (loc, v0, v1, v2, v3);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform1i (GLcharARB * varname, GLint v0, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform1i (loc, v0);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform2i (GLcharARB * varname, GLint v0, GLint v1, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform2i (loc, v0, v1);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform3i (GLcharARB * varname, GLint v0, GLint v1, GLint v2, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform3i (loc, v0, v1, v2);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform4i (GLcharARB * varname, GLint v0, GLint v1, GLint v2, GLint v3, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform4i (loc, v0, v1, v2, v3);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform1ui (GLcharARB * varname, GLuint v0, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform1uiEXT (loc, v0);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform2ui (GLcharARB * varname, GLuint v0, GLuint v1, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform2uiEXT (loc, v0, v1);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform3ui (GLcharARB * varname, GLuint v0, GLuint v1, GLuint v2, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform3uiEXT (loc, v0, v1, v2);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform4ui (GLcharARB * varname, GLuint v0, GLuint v1, GLuint v2, GLuint v3, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform4uiEXT (loc, v0, v1, v2, v3);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform1fv (GLcharARB * varname, GLsizei count, GLfloat * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform1fv (loc, count, value);

    return true;
}

bool glShader::setUniform2fv (GLcharARB * varname, GLsizei count, GLfloat * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform2fv (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform3fv (GLcharARB * varname, GLsizei count, GLfloat * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform3fv (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform4fv (GLcharARB * varname, GLsizei count, GLfloat * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform4fv (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform1iv (GLcharARB * varname, GLsizei count, GLint * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform1iv (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform2iv (GLcharARB * varname, GLsizei count, GLint * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform2iv (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform3iv (GLcharARB * varname, GLsizei count, GLint * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform3iv (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform4iv (GLcharARB * varname, GLsizei count, GLint * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform4iv (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform1uiv (GLcharARB * varname, GLsizei count, GLuint * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform1uivEXT (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform2uiv (GLcharARB * varname, GLsizei count, GLuint * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform2uivEXT (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform3uiv (GLcharARB * varname, GLsizei count, GLuint * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform3uivEXT (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniform4uiv (GLcharARB * varname, GLsizei count, GLuint * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniform4uivEXT (loc, count, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniformMatrix2fv (GLcharARB * varname, GLsizei count, GLboolean transpose, GLfloat * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniformMatrix2fv (loc, count, transpose, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniformMatrix3fv (GLcharARB * varname, GLsizei count, GLboolean transpose, GLfloat * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniformMatrix3fv (loc, count, transpose, value);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setUniformMatrix4fv (GLcharARB * varname, GLsizei count, GLboolean transpose, GLfloat * value, GLint index)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return false;           // can't find variable / invalid index

    glUniformMatrix4fv (loc, count, transpose, value);

    return true;
}

// -----------------------------------------------------------------------------

GLint glShader::GetUniformLocation (const GLcharARB * name)
{
    GLint   loc;

    loc = glGetUniformLocation (ProgramObject, name);
    if (loc == -1)
    {
    //    cout << "Error: can't find uniform variable \"" << name << "\"\n";
    }
    CHECK_GL_ERROR ();

    return loc;
}

// -----------------------------------------------------------------------------

void glShader::getUniformfv (GLcharARB * varname, GLfloat * values, GLint index)
{
    if (! useGLSL) return;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return;                 // can't find variable / invalid index

    glGetUniformfv (ProgramObject, loc, values);
}

// -----------------------------------------------------------------------------

void glShader::getUniformiv (GLcharARB * varname, GLint * values, GLint index)
{
    if (! useGLSL) return;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return;                 // can't find variable / invalid index

    glGetUniformiv (ProgramObject, loc, values);
}

// -----------------------------------------------------------------------------

void glShader::getUniformuiv (GLcharARB * varname, GLuint * values, GLint index)
{
    if (! useGLSL) return;

    GLint   loc;

    if (varname)
        GetUniformLocation (varname);
    else
        loc = index;

    if (loc == -1)
        return;                 // can't find variable / invalid index

    glGetUniformuivEXT (ProgramObject, loc, values);
}

// -----------------------------------------------------------------------------

void glShader::BindAttribLocation (GLint index, GLchar * name)
{
    glBindAttribLocation (ProgramObject, index, name);
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib1f (GLuint index, GLfloat v0)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib1f (index, v0);

    return true;
}

bool glShader::setVertexAttrib2f (GLuint index, GLfloat v0, GLfloat v1)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib2f (index, v0, v1);

    return true;
}

bool glShader::setVertexAttrib3f (GLuint index, GLfloat v0, GLfloat v1, GLfloat v2)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib3f (index, v0, v1, v2);

    return true;
}

bool glShader::setVertexAttrib4f (GLuint index, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib4f (index, v0, v1, v2, v3);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib1d (GLuint index, GLdouble v0)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib1d (index, v0);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib2d (GLuint index, GLdouble v0, GLdouble v1)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib2d (index, v0, v1);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib3d (GLuint index, GLdouble v0, GLdouble v1, GLdouble v2)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib3d (index, v0, v1, v2);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib4d (GLuint index, GLdouble v0, GLdouble v1, GLdouble v2, GLdouble v3)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib4d (index, v0, v1, v2, v3);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib1s (GLuint index, GLshort v0)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib1s (index, v0);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib2s (GLuint index, GLshort v0, GLshort v1)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib2s (index, v0, v1);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib3s (GLuint index, GLshort v0, GLshort v1, GLshort v2)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib3s (index, v0, v1, v2);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib4s (GLuint index, GLshort v0, GLshort v1, GLshort v2, GLshort v3)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib4s (index, v0, v1, v2, v3);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttribNormalizedByte (GLuint index, GLbyte v0, GLbyte v1, GLbyte v2, GLbyte v3)
{
    if (! useGLSL) return false;           // GLSL not available
    if (!_noshader) return true;

    glVertexAttrib4Nub (index, v0, v1, v2, v3);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib1i (GLuint index, GLint v0)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    glVertexAttribI1iEXT (index, v0);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib2i (GLuint index, GLint v0, GLint v1)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    glVertexAttribI2iEXT (index, v0, v1);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib3i (GLuint index, GLint v0, GLint v1, GLint v2)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    glVertexAttribI3iEXT (index, v0, v1, v2);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib4i (GLuint index, GLint v0, GLint v1, GLint v2, GLint v3)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    glVertexAttribI4iEXT (index, v0, v1, v2, v3);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib1ui (GLuint index, GLuint v0)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    glVertexAttribI1uiEXT (index, v0);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib2ui (GLuint index, GLuint v0, GLuint v1)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    glVertexAttribI2uiEXT (index, v0, v1);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib3ui (GLuint index, GLuint v0, GLuint v1, GLuint v2)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    glVertexAttribI3uiEXT (index, v0, v1, v2);

    return true;
}

// -----------------------------------------------------------------------------

bool glShader::setVertexAttrib4ui (GLuint index, GLuint v0, GLuint v1, GLuint v2, GLuint v3)
{
    if (! useGLSL) return false;           // GLSL not available
    if (! bGPUShader4) return false;
    if (!_noshader) return true;

    glVertexAttribI4uiEXT (index, v0, v1, v2, v3);

    return true;
}

// -----------------------------------------------------------------------------

// ************************************************************************
// Shader Program : Manage Shader Programs (Vertex/Fragment)
// ************************************************************************
glShaderObject::glShaderObject ()
{
    InitOpenGLExtensions ();

    compiler_log = 0;
    is_compiled = false;
    program_type = 0;
    ShaderObject = 0;
    ShaderSource = 0;
    _memalloc = false;
}

// -----------------------------------------------------------------------------

glShaderObject::~glShaderObject ()
{
    if (compiler_log != 0)
        free (compiler_log);

    if (ShaderSource != 0)
    {
        if (_memalloc)
            delete[]ShaderSource; // free ASCII Source
    }

    if (is_compiled)
    {
        glDeleteObjectARB (ShaderObject);
        CHECK_GL_ERROR ();
    }
}

// -----------------------------------------------------------------------------

unsigned long getFileLength (ifstream & file)
{
    if (! file.good ())
        return 0;

    // unsigned long pos=file.tellg();
    file.seekg (0, ios::end);
    unsigned long len = file.tellg ();

    file.seekg (ios::beg);

    return len;
}

// -----------------------------------------------------------------------------

int glShaderObject::load_fd (char *filename)
{
    FILE *fd;

    // must read files as binary to prevent problems from newline translation
    fd = fopen (filename, "rb");
    if (fd == NULL)
        return -1;

    unsigned long len;

    fseek (fd, 0, SEEK_END);
    len = ftell (fd);

    if (len == 0)
        return -2;              // "Empty File"

    if (ShaderSource != 0)      // there is already a source loaded, free it!
    {
        if (_memalloc)
            delete [] ShaderSource;
    }

    ShaderSource = (GLubyte *) new char[len+1];

    if (ShaderSource == 0)
        return -3;              // can't reserve memory
    _memalloc = true;

    fseek (fd, 0, SEEK_SET);
    fread (ShaderSource, len, 1, fd);
    fclose (fd);

    ShaderSource[len] = '\0';

    return 0;
}

int glShaderObject::load (char *filename)
{
    ifstream file;

    file.open (filename, ios::in);
    if (! file)
        return -1;

    unsigned long len = getFileLength (file);

    if (len == 0)
        return -2;              // "Empty File"

    if (ShaderSource != 0)      // there is already a source loaded, free it!
    {
        if (_memalloc)
            delete[]ShaderSource;
    }

    ShaderSource = (GLubyte *) new char[len + 1];

    if (ShaderSource == 0)
        return -3;              // can't reserve memory
    _memalloc = true;

    ShaderSource[len] = 0;      // len isn't always strlen cause some
                                // characters are stripped in ascii read...
    // it is important to 0-terminate the real length later, len is just max
    // possible value...
    unsigned int i = 0;

    while (file.good ())
    {
        ShaderSource[i] = file.get (); // get character from file.
        if (! file.eof ())
            i++;
    }

    ShaderSource[i] = 0;        // 0 terminate it.

    file.close ();

    return 0;
}

// -----------------------------------------------------------------------------

void glShaderObject::loadFromMemory (const char *program)
{
    if (ShaderSource != 0)      // there is already a source loaded, free it!
    {
        if (_memalloc)
            delete[]ShaderSource;
    }
    _memalloc = false;
    ShaderSource = (GLubyte *) program;
}

// -----------------------------------------------------------------------------

// Compiler Log: Ausgabe der Compiler Meldungen in String
GLint glShaderObject::getCompilerLogLength (void)
{
    GLint   blen = 0;

    glGetShaderiv (ShaderObject, GL_INFO_LOG_LENGTH, &blen);
    CHECK_GL_ERROR ();

    return blen;
}

// ----------------------------------------------------------------------------
// Compiler Log: Ausgabe der Compiler Meldungen in String
char   * glShaderObject::getCompilerLog (void)
{
    if (! useGLSL) return aGLSLStrings[0];

    if (ShaderObject == 0)
        return aGLSLStrings[1]; // not a valid program object

    GLsizei slen = 0;
    GLint   blen = getCompilerLogLength ();

    if (blen > 1)
    {
        if (compiler_log != 0)
        {
            free (compiler_log);
            compiler_log = 0;
        }
        if ((compiler_log = (GLcharARB *) malloc (blen)) == NULL)
        {
            cout << "Error: Could not allocate compiler_log buffer\n";
            return aGLSLStrings[3];
        }

     // glGetInfoLogARB (ShaderObject, blen, &slen, compiler_log);
        glGetShaderInfoLog (ShaderObject, blen, &slen, compiler_log);
        CHECK_GL_ERROR ();
     // cout << "compiler_log: \n", compiler_log;
    }

    if (compiler_log != 0)
        return (char *) compiler_log;
    else
        return aGLSLStrings[6];

    return aGLSLStrings[4];
}

// -----------------------------------------------------------------------------

bool glShaderObject::compile (void)
{
    if (! useGLSL) return false;

    is_compiled = false;

    GLint   compiled = 0;

    if (ShaderSource == 0)
        return false;

    GLint   length = (GLint) strlen ((const char *) ShaderSource);

 // glShaderSourceARB (ShaderObject, 1, (const GLcharARB **) &ShaderSource, &length);
    glShaderSource (ShaderObject, 1, (const GLchar **) &ShaderSource, NULL);
    CHECK_GL_ERROR ();

 // glCompileShaderARB (ShaderObject);
    glCompileShader (ShaderObject);
    CHECK_GL_ERROR ();
 // glGetObjectParameterivARB (ShaderObject, GL_COMPILE_STATUS, &compiled);
    glGetShaderiv (ShaderObject, GL_COMPILE_STATUS, &compiled);
    CHECK_GL_ERROR ();

    if (compiled)
        is_compiled = true;

    return is_compiled;
}

// -----------------------------------------------------------------------------

GLint glShaderObject::getAttribLocation (char *attribName)
{
    return glGetAttribLocationARB (ShaderObject, attribName);
}

// -----------------------------------------------------------------------------

aVertexShader::aVertexShader ()
{
    program_type = 1;
    if (useGLSL)
    {
        ShaderObject = glCreateShaderObjectARB (GL_VERTEX_SHADER_ARB);
        CHECK_GL_ERROR ();
    }
}

// ----------------------------------------------------
aVertexShader::~aVertexShader ()
{
}

// ----------------------------------------------------

aFragmentShader::aFragmentShader ()
{
    program_type = 2;
    if (useGLSL)
    {
        ShaderObject = glCreateShaderObjectARB (GL_FRAGMENT_SHADER_ARB);
        CHECK_GL_ERROR ();
    }
}

// ----------------------------------------------------

aFragmentShader::~aFragmentShader ()
{
}

// ----------------------------------------------------

aGeometryShader::aGeometryShader ()
{
    program_type = 3;
    if (useGLSL && bGeometryShader)
    {
        ShaderObject = glCreateShaderObjectARB (GL_GEOMETRY_SHADER);
        CHECK_GL_ERROR ();
    }
}

// ----------------------------------------------------

aGeometryShader::~aGeometryShader ()
{
}

// -----------------------------------------------------------------------------
// ShaderManager: Easy use of (multiple) Shaders

glShaderManager::glShaderManager ()
{
#ifdef WIN32
    InitOpenGLExtensions ();
#endif
    _nInputPrimitiveType = GL_TRIANGLES;
    _nOutputPrimitiveType = GL_TRIANGLE_STRIP;
    _nVerticesOut = 3;
}

glShaderManager::~glShaderManager ()
{
    // free objects
    vector <glShader *>::iterator i = _shaderObjectList.begin ();

    while (i != _shaderObjectList.end ())
        i = _shaderObjectList.erase (i);
}

// -----------------------------------------------------------------------------

void glShaderManager::SetInputPrimitiveType (int nInputPrimitiveType)
{
    _nInputPrimitiveType = nInputPrimitiveType;
}

void glShaderManager::SetOutputPrimitiveType (int nOutputPrimitiveType)
{
    _nOutputPrimitiveType = nOutputPrimitiveType;
}

void glShaderManager::SetVerticesOut (int nVerticesOut)
{
    _nVerticesOut = nVerticesOut;
}

// -----------------------------------------------------------------------------

bool glShaderManager::loadfromFile (glShader *o, char *file, GLenum shaderType)
{
    o->setName (file);
    o->UsesGeometryShader (shaderType == GL_GEOMETRY_SHADER ? true : false);

    glShaderObject *shaderObject;

         if (shaderType == GL_VERTEX_SHADER)
        shaderObject = new aVertexShader;
    else if (shaderType == GL_FRAGMENT_SHADER)
        shaderObject = new aFragmentShader;
    else if (shaderType == GL_GEOMETRY_SHADER)
    {
        shaderObject = new aGeometryShader;

        o->SetInputPrimitiveType (_nInputPrimitiveType);
        o->SetOutputPrimitiveType (_nOutputPrimitiveType);
        o->SetVerticesOut (_nVerticesOut);
    }
    CHECK_GL_ERROR ();

    // load the program
    if (file != 0 && shaderObject->load_fd (file) != 0)
    {
        cout << "Error: can't load shader \"" << file << "\" !\n";

        delete  shaderObject;
        return 0;
    }

    // Compile the program
    if (file != 0 && ! shaderObject->compile ())
    {
        cout << "***COMPILER ERROR (Shader: \"" << file << "\"):\n";
        cout << shaderObject->getCompilerLog () << endl;

        delete  shaderObject;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (shaderObject->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Shader: \"" << file << "\"):\n";
		cout << shaderObject->getCompilerLog () << "\n";
	}
#endif
#endif

    // Add to object
    if (file != 0) o->addShader (shaderObject);

    _shaderObjectList.push_back (o);
    o->manageMemory ();

    return 1;
}

bool glShaderManager::link (glShader *o)
{
    // link
    if (! o->link ())
    {
        cout << "**LINKER ERROR (Shader: \"" << o->getName () << "\"):\n";
        cout << o->getLinkerLog () << endl;

        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
    else // if (o->getLinkerLogLength () > 1)
    {
        cout << endl << "***GLSL Linker Log (Shader: \"" << o->getName () << "\"):\n";
        cout << o->getLinkerLog () << endl;
    }
#endif

    return 1;
}

glShader * glShaderManager::loadfromFile (char *vertexFile, char *fragmentFile)
{
    glShader *o = new glShader ();

    o->UsesGeometryShader (false);

    aVertexShader *tVertexShader = new aVertexShader;
    aFragmentShader *tFragmentShader = new aFragmentShader;

    // load vertex program
    if (vertexFile != 0 && tVertexShader->load_fd (vertexFile) != 0)
    {
        cout << "Error: can't load vertex shader \"" << vertexFile << "\" !\n";

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        return 0;
    }

    // Load fragment program
    if (fragmentFile != 0 && tFragmentShader->load_fd (fragmentFile) != 0)
    {
        cout << "Error: can't load fragment shader \"" << fragmentFile << "\" !\n";

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        return 0;
    }

    // Compile vertex program
    if (vertexFile != 0 && ! tVertexShader->compile ())
    {
        cout << "***COMPILER ERROR (Vertex Shader: \"" << vertexFile << "\"):\n";
        cout << tVertexShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
	if (tVertexShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Vertex Shader: \"" << vertexFile << "\"):\n";
		cout << tVertexShader->getCompilerLog () << "\n";
	}
#endif

    // Compile fragment program
    if (fragmentFile != 0 && ! tFragmentShader->compile ())
    {
        cout << "***COMPILER ERROR (Fragment Shader: \"" << fragmentFile << "\"):\n";
        cout << tFragmentShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
	if (tFragmentShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Fragment Shader: \"" << fragmentFile << "\"):\n";
		cout << tFragmentShader->getCompilerLog () << "\n";
	}
#endif

    // Add to object
    if (vertexFile   != 0) o->addShader (tVertexShader);
    if (fragmentFile != 0) o->addShader (tFragmentShader);

    // link
    if (! o->link ())
    {
        cout << "**LINKER ERROR (Shaders vert: \"" << vertexFile << "\" frag: \"" << fragmentFile << "\"):\n";
        cout << o->getLinkerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
    // if (o->getLinkerLogLength () > 1)
    {
        cout << endl << "***GLSL Linker Log (Shaders vert: \"" << vertexFile << "\" frag: \"" << fragmentFile << "\"):\n";
        cout << o->getLinkerLog () << endl;
    }
#endif
    _shaderObjectList.push_back (o);
    o->manageMemory ();

    return o;
}

glShader * glShaderManager::loadfromFile (char *vertexFile, char *geometryFile, char *fragmentFile)
{
    glShader *o = new glShader ();

    o->UsesGeometryShader (true);
    o->SetInputPrimitiveType (_nInputPrimitiveType);
    o->SetOutputPrimitiveType (_nOutputPrimitiveType);
    o->SetVerticesOut (_nVerticesOut);

    aVertexShader *tVertexShader = new aVertexShader;
    aFragmentShader *tFragmentShader = new aFragmentShader;
    aGeometryShader *tGeometryShader = new aGeometryShader;

    // load vertex program
    if (vertexFile != 0 && tVertexShader->load_fd (vertexFile) != 0)
    {
        cout << "Error: can't load vertex shader \"" << vertexFile << "\" !\n";

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }

    // Load geometry program
    if (geometryFile != 0 && tGeometryShader->load_fd (geometryFile) != 0)
    {
        cout << "Error: can't load geometry shader \"" << geometryFile << "\" !\n";

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }

    // Load fragment program
    if (fragmentFile != 0 && tFragmentShader->load_fd (fragmentFile) != 0)
    {
        cout << "Error: can't load fragment shader \"" << fragmentFile << "\" !\n";

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }

    // Compile vertex program
    if (vertexFile != 0 && ! tVertexShader->compile ())
    {
        cout << "***COMPILER ERROR (Vertex Shader \"" << vertexFile << "\"):\n";
        cout << tVertexShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (tVertexShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Vertex Shader \"" << vertexFile << "\"):\n";
		cout << tVertexShader->getCompilerLog () << "\n";
	}
#endif
#endif

    // Compile geometry program
    if (geometryFile != 0 && ! tGeometryShader->compile ())
    {
        cout << "***COMPILER ERROR (Geometry Shader \"" << geometryFile << "\"):\n";
        cout << tGeometryShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (tGeometryShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Geometry Shader \"" << geometryFile << "\"):\n";
		cout << tGeometryShader->getCompilerLog () << "\n";
	}
#endif
#endif

    // Compile fragment program
    if (fragmentFile != 0 && ! tFragmentShader->compile ())
    {
        cout << "***COMPILER ERROR (Fragment Shader \"" << fragmentFile << "\"):\n";
        cout << tFragmentShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (tFragmentShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Fragment Shader \"" << fragmentFile << "\"):\n";
		cout << tFragmentShader->getCompilerLog () << "\n";
	}
#endif
#endif

    // Add to object
    if (vertexFile   != 0) o->addShader (tVertexShader);
    if (geometryFile != 0) o->addShader (tGeometryShader);
    if (fragmentFile != 0) o->addShader (tFragmentShader);

    // link
    if (! o->link ())
    {
        cout << "**LINKER ERROR (Shaders vert: \"" << vertexFile << "\" frag: \"" << fragmentFile << "\" geom: \"" << geometryFile << "\"):\n";
        cout << o->getLinkerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
    // if (o->getLinkerLogLength () > 1)
    {
        cout << endl << "***GLSL Linker Log (Shaders vert: \"" << vertexFile << "\" frag: \"" << fragmentFile << "\" geom: \"" << geometryFile << "\"):\n";
        cout << o->getLinkerLog () << endl;
    }
#endif
    _shaderObjectList.push_back (o);
    o->manageMemory ();

    return o;
}

// -----------------------------------------------------------------------------

bool glShaderManager::loadfromMemory (glShader *o, const char *shaderName, const char *shaderMem, GLenum shaderType)
{
    o->setName ((char *) shaderName);
    o->UsesGeometryShader (shaderType == GL_GEOMETRY_SHADER ? true : false);

    glShaderObject *shaderObject;

         if (shaderType == GL_VERTEX_SHADER)
        shaderObject = new aVertexShader;
    else if (shaderType == GL_FRAGMENT_SHADER)
        shaderObject = new aFragmentShader;
    else if (shaderType == GL_GEOMETRY_SHADER)
    {
        shaderObject = new aGeometryShader;

        o->SetInputPrimitiveType (_nInputPrimitiveType);
        o->SetOutputPrimitiveType (_nOutputPrimitiveType);
        o->SetVerticesOut (_nVerticesOut);
    }
    CHECK_GL_ERROR ();

    // get shader program
    if (shaderMem != 0)
        shaderObject->loadFromMemory (shaderMem);

    // Compile the program
    if (shaderMem != 0 && ! shaderObject->compile ())
    {
        cout << "***COMPILER ERROR (Shader: \"" << shaderName << "\"):\n";
        cout << shaderObject->getCompilerLog () << endl;

        delete  shaderObject;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (shaderObject->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Shader: \"" << shaderName << "\"):\n";
		cout << shaderObject->getCompilerLog () << "\n";
	}
#endif
#endif

    // Add to object
    if (shaderMem != 0) o->addShader (shaderObject);

    _shaderObjectList.push_back (o);
    o->manageMemory ();

    return 1;
}

glShader * glShaderManager::loadfromMemory (const char *vertexName, const char *vertexMem, const char *fragmentName, const char *fragmentMem)
{
    glShader *o = new glShader ();

    o->UsesGeometryShader (false);

    aVertexShader *tVertexShader = new aVertexShader;
    aFragmentShader *tFragmentShader = new aFragmentShader;

    // get vertex program
    if (vertexMem != 0)
        tVertexShader->loadFromMemory (vertexMem);

    // get fragment program
    if (fragmentMem != 0)
        tFragmentShader->loadFromMemory (fragmentMem);

    // Compile vertex program
    if (vertexMem != 0 && ! tVertexShader->compile ())
    {
        cout << "***COMPILER ERROR (Vertex Shader \"" << vertexName << "\"):\n";
        cout << tVertexShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (tVertexShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Vertex Shader \"" << vertexName << "\"):\n";
		cout << tVertexShader->getCompilerLog () << "\n";
	}
#endif
#endif

    // Compile fragment program
    if (fragmentMem != 0 && ! tFragmentShader->compile ())
    {
        cout << "***COMPILER ERROR (Fragment Shader \"" << fragmentName << "\"):\n";
        cout << tFragmentShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (tFragmentShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Fragment Shader \"" << fragmentName << "\"):\n";
		cout << tFragmentShader->getCompilerLog () << "\n";
	}
#endif
#endif

    // Add to object
    if (vertexMem   != 0) o->addShader (tVertexShader);
    if (fragmentMem != 0) o->addShader (tFragmentShader);

    // link
    if (! o->link ())
    {
        cout << "**LINKER ERROR (Shaders vert: \"" << vertexName << "\" frag: \"" << fragmentName << "\"):\n";
        cout << o->getLinkerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
    // if (o->getLinkerLogLength () > 1)
    {
        cout << endl << "***GLSL Linker Log (Shaders vert: \"" << vertexName << "\" frag: \"" << fragmentName << "\"):\n";
        cout << o->getLinkerLog () << endl;
    }
#endif
    _shaderObjectList.push_back (o);
    o->manageMemory ();

    return o;
}

glShader * glShaderManager::loadfromMemory (const char *vertexName, const char *vertexMem, const char *geometryName, const char *geometryMem, const char *fragmentName, const char *fragmentMem)
{
    glShader *o = new glShader ();

    o->UsesGeometryShader (true);
    o->SetInputPrimitiveType (_nInputPrimitiveType);
    o->SetOutputPrimitiveType (_nOutputPrimitiveType);
    o->SetVerticesOut (_nVerticesOut);

    aVertexShader *tVertexShader = new aVertexShader;
    aFragmentShader *tFragmentShader = new aFragmentShader;
    aGeometryShader *tGeometryShader = new aGeometryShader;

    // get vertex program
    if (vertexMem != 0)
        tVertexShader->loadFromMemory (vertexMem);

    // get fragment program
    if (fragmentMem != 0)
        tFragmentShader->loadFromMemory (fragmentMem);

    // get fragment program
    if (geometryMem != 0)
        tGeometryShader->loadFromMemory (geometryMem);

    // Compile vertex program
    if (vertexMem != 0 && ! tVertexShader->compile ())
    {
        cout << "***COMPILER ERROR (Vertex Shader \"" << vertexName << "\"):\n";
        cout << tVertexShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (tVertexShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Vertex Shader \"" << vertexName << "\"):\n";
		cout << tVertexShader->getCompilerLog () << "\n";
	}
#endif
#endif

    // Compile geometry program
    if (geometryMem != 0 && ! tGeometryShader->compile ())
    {
        cout << "***COMPILER ERROR (Geometry Shader \"" << geometryName << "\"):\n";
        cout << tGeometryShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (tVertexShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Geometry Shader \"" << geometryName << "\"):\n";
		cout << tVertexShader->getCompilerLog () << "\n";
	}
#endif
#endif

    // Compile fragment program
    if (fragmentMem != 0 && ! tFragmentShader->compile ())
    {
        cout << "***COMPILER ERROR (Fragment Shader \"" << fragmentName << "\"):\n";
        cout << tFragmentShader->getCompilerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
#if 0
	if (tFragmentShader->getCompilerLogLength () > 1)
	{
		cout << "\n***GLSL Compiler Log (Fragment Shader \"" << fragmentName << "\"):\n";
		cout << tFragmentShader->getCompilerLog () << "\n";
	}
#endif
#endif

    // Add to object
    if (vertexMem   != 0) o->addShader (tVertexShader);
    if (geometryMem != 0) o->addShader (tGeometryShader);
    if (fragmentMem != 0) o->addShader (tFragmentShader);

    // link
    if (! o->link ())
    {
        cout << "**LINKER ERROR (Shaders vert: \"" << vertexName << "\" geom: \"" << geometryName << "\" frag: \"" << fragmentName << "\"):\n";
        cout << o->getLinkerLog () << endl;

        delete  o;
        delete  tVertexShader;
        delete  tFragmentShader;
        delete  tGeometryShader;
        return 0;
    }
#ifndef GLSL_DISABLE_LOGGING
    // if (o->getLinkerLogLength () > 1)
    {
        cout << endl << "***GLSL Linker Log (Shaders vert: \"" << vertexName << "\" geom: \"" << geometryName << "\" frag: \"" << fragmentName << "\"):\n";
        cout << o->getLinkerLog () << endl;
    }
#endif
    _shaderObjectList.push_back (o);
    o->manageMemory ();

    return o;
}

// -----------------------------------------------------------------------------

bool glShaderManager::free (glShader * o)
{
    vector <glShader *>::iterator i = _shaderObjectList.begin ();
    while (i != _shaderObjectList.end ())
    {
        if ((*i) == o)
        {
            _shaderObjectList.erase (i);
            delete  o;
            return true;
        }
        i++;
    }

    return false;
}

