/*    Copyright (C) 2005 Robert Kooima                                       */
/*                                                                           */
/*    obj.[ch] is free software; you can redistribute it and/or modify it    */
/*    under the terms of the  GNU General Public License  as published by    */
/*    the  Free Software Foundation;  either version 2 of the License, or    */
/*    (at your option) any later version.                                    */
/*                                                                           */
/*    This program is distributed in the hope that it will be useful, but    */
/*    WITHOUT  ANY  WARRANTY;  without   even  the  implied  warranty  of    */
/*    MERCHANTABILITY or  FITNESS FOR A PARTICULAR PURPOSE.   See the GNU    */
/*    General Public License for more details.                               */
/*                                                                           */
/*    Modified by Athanasios Gaitatzes                                       */
/*                                                                           */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAXSTR 1024

/*---------------------------------------------------------------------------*/

#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	#include <GL/glew.h>

	#define glGetProcAddress(n) wglGetProcAddress(n)
	#pragma warning (disable : 4996)
#endif

#ifdef __linux__
	// this should be defined before glext.h
	#define GL_GLEXT_PROTOTYPES

	#include <GL/glx.h>
	#include <GL/glu.h>
	#include <GL/glext.h>

	#define glGetProcAddress(n) glXGetProcAddressARB((GLubyte *) n)
#endif

#ifdef __APPLE__
	#include <OpenGL/gl.h>
	#include <OpenGL/glu.h>
#endif

#include "Material3D.h"
#include "Texture2D.h"

#include "obj_vbo.h"

/*===========================================================================*/
/* OpenGL State                                                              */

#ifdef WIN32
static PFNGLENABLEVERTEXATTRIBARRAYARBPROC	glEnableVertexAttribArray;
static PFNGLVERTEXATTRIBPOINTERARBPROC		glVertexAttribPointer;
static PFNGLGENBUFFERSARBPROC				glGenBuffers;
static PFNGLBINDBUFFERARBPROC				glBindBuffer;
static PFNGLBUFFERDATAARBPROC				glBufferData;
static PFNGLBUFFERSUBDATAARBPROC			glBufferSubData;
static PFNGLDELETEBUFFERSARBPROC			glDeleteBuffers;
static PFNGLACTIVETEXTUREARBPROC			glActiveTexture;
#endif

static GLint     GL_max_texture_image_units;
static GLboolean GL_has_vertex_buffer_object;
static GLboolean GL_has_multitexture;
static GLboolean GL_is_initialized;

/*---------------------------------------------------------------------------*/

static GLboolean gl_ext(const char *needle)
{
    const GLubyte *haystack, *c;

    for (haystack = glGetString(GL_EXTENSIONS); *haystack; haystack++)
    {
        for (c = (const GLubyte *) needle; *c && *haystack; c++, haystack++)
            if (*c != *haystack)
                break;

        if ((*c == 0) && (*haystack == ' ' || *haystack == '\0'))
            return GL_TRUE;
    }
    return GL_FALSE;
}

static void obj_init_gl(void)
{
    if (GL_is_initialized)
        return;

    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &GL_max_texture_image_units);

    GL_has_vertex_buffer_object = gl_ext("GL_ARB_vertex_buffer_object");
    GL_has_multitexture         = gl_ext("GL_ARB_multitexture");

#ifdef WIN32
    glEnableVertexAttribArray	= (PFNGLENABLEVERTEXATTRIBARRAYARBPROC)	glGetProcAddress("glEnableVertexAttribArrayARB");
    glVertexAttribPointer		= (PFNGLVERTEXATTRIBPOINTERARBPROC)		glGetProcAddress("glVertexAttribPointerARB");
    glGenBuffers				= (PFNGLGENBUFFERSARBPROC)				glGetProcAddress("glGenBuffersARB");
    glBindBuffer				= (PFNGLBINDBUFFERARBPROC)				glGetProcAddress("glBindBufferARB");
    glBufferData				= (PFNGLBUFFERDATAARBPROC)				glGetProcAddress("glBufferDataARB");
    glBufferSubData				= (PFNGLBUFFERSUBDATAARBPROC)			glGetProcAddress("glBufferSubDataARB");
    glDeleteBuffers				= (PFNGLDELETEBUFFERSARBPROC)			glGetProcAddress("glDeleteBuffersARB");
    glActiveTexture				= (PFNGLACTIVETEXTUREARBPROC)			glGetProcAddress("glActiveTextureARB");
#endif

    GL_is_initialized = 1;
}

/*===========================================================================*/
/* Vector cache                                                              */

struct vec2
{
    float v[2];
    int _ii;
};

struct vec3
{
    float v[3];
    int _ii;
};

struct iset
{
    int vi;
    int gi;

    unsigned long _vi;
    unsigned long _ti;
    unsigned long _ni;
    int _ii;
};

static unsigned long _vc, _vm;
static unsigned long _tc, _tm;
static unsigned long _nc, _nm;
static unsigned long _ic, _im;

static struct vec3 *_vv;
static struct vec2 *_tv;
static struct vec3 *_nv;
static struct iset *_iv;

/*---------------------------------------------------------------------------*/

static unsigned long add__(void **_v, unsigned long *_c, unsigned long *_m, size_t _s)
{
    unsigned long m = (*_m > 0) ? *_m * 2 : 2;
    void *v;

    /* If space remains in the current block, return it. */
    if (*_m > *_c)
        return (*_c)++;

    /* Else, try to increase the size of the block. */
    else if ((v = realloc(*_v, _s * m)))
    {
        *_v = v;
        *_m = m;
        return (*_c)++;
    }

    /* Else, indicate failure. */
    else
        return ULONG_MAX;
}

static unsigned long add_v(void)
{
    return add__((void **) &_vv, &_vc, &_vm, sizeof (struct vec3));
}

static unsigned long add_t(void)
{
    return add__((void **) &_tv, &_tc, &_tm, sizeof (struct vec2));
}

static unsigned long add_n(void)
{
    return add__((void **) &_nv, &_nc, &_nm, sizeof (struct vec3));
}

static unsigned long add_i(void)
{
    return add__((void **) &_iv, &_ic, &_im, sizeof (struct iset));
}

/*===========================================================================*/
/* Handy functions                                                           */

void objMesh3D::compute_normal(Vector3D *n, Vector3D a, Vector3D b, Vector3D c)
{
    Vector3D u = Vector3D (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    Vector3D v = Vector3D (c[0] - a[0], c[1] - a[1], c[2] - a[2]);

    *n = Vector3D::cross(u, v);
    n->normalize();
}

/*===========================================================================*/

static void dirpath(char *pathname)
{
    int i;

    /* Find the path by cutting a file name at the last directory delimiter. */
    for (i = (int) strlen(pathname) - 1; i >= 0; --i)
        if (pathname[i] == DIR_DELIMITER)
        {
            pathname[i] = '\0';
            return;
        }

    /* If no delimiter was found, return the current directory. */
    strcpy(pathname, ".");
}

/*---------------------------------------------------------------------------*/

void objMesh3D::read_image(int mi, int ri, const char *line,
                           const char *path)
{
    char pathname[MAXSTR];
    char map[MAXSTR];

    while (line[0] != '\0' && line[0] != '\r' && line[0] != '\n')
    {
        int n = 0;

        /* Parse a word.  The last word seen is taken to be the file name. */
        if (sscanf(line, " %s%n", map, &n) >= 1)
            line += n;
    }

    /* Apply all parsed property attributes to the material. */
    sprintf(pathname, "%s/%s", path, map);

    obj_set_mtrl_map(mi, ri, pathname);
}

void objMesh3D::read_ambient(int mi, const char *line)
{
    float color[3];

    sscanf(line, "%f %f %f", color, color + 1, color + 2);
    obj_set_mtrl_ambient(mi, color);
}

void objMesh3D::read_diffuse(int mi, const char *line)
{
    float color[3];

    sscanf(line, "%f %f %f", color, color + 1, color + 2);
    obj_set_mtrl_diffuse(mi, color);
}

void objMesh3D::read_specular(int mi, const char *line)
{
    float color[3];

    sscanf(line, "%f %f %f", color, color + 1, color + 2);
    obj_set_mtrl_specular(mi, color);
}

void objMesh3D::read_emission(int mi, const char *line)
{
    float color[3];

    sscanf(line, "%f %f %f", color, color + 1, color + 2);
    obj_set_mtrl_emission(mi, color);
}

void objMesh3D::read_shininess(int mi, const char *line)
{
    int shine;

    sscanf(line, "%d", &shine);
    obj_set_mtrl_shininess(mi, shine);
}

void objMesh3D::read_alpha(int mi, const char *line)
{
    float alpha;

    sscanf(line, "%f", &alpha);
    obj_set_mtrl_alpha(mi, alpha);
}

void objMesh3D::read_material(const char *path, const char *file,
                              const char *name, int mi)
{
    char pathname[MAXSTR];

    char buf[MAXSTR];
    char key[MAXSTR];
    char arg[MAXSTR];

    FILE *fin;

    int scanning = 1;
    int n        = 0;

    sprintf(pathname, "%s/%s", path, file);

    if ((fin = fopen(pathname, "r")))
    {
        /* Process each line of the MTL file. */
        while  (fgets (buf, MAXSTR, fin))
        {
            if (sscanf(buf, "%s%n", key, &n) >= 1)
            {
                const char *c = buf + n;

                if (scanning)
                {
                    /* Determine if we've found the MTL we're looking for. */
                    if (!strcmp(key, "newmtl"))
                    {
                        sscanf(c, "%s", arg);

                        if ((scanning = strcmp(arg, name)) == 0)
                            obj_set_mtrl_name(mi, name);
                    }
                }
                else
                {
                    /* Stop scanning when the next MTL begins. */
                    if (!strcmp(key, "newmtl"))
                        break;
					else
                    /* Parse this material's properties. */
					{
						if (!strcmp(key, "map_Ka")) read_image(mi, OBJ_KA, c, path);
						if (!strcmp(key, "map_Kd")) read_image(mi, OBJ_KD, c, path);
						if (!strcmp(key, "map_Ks")) read_image(mi, OBJ_KS, c, path);
						if (!strcmp(key, "map_Ke")) read_image(mi, OBJ_KE, c, path);
						if (!strcmp(key, "map_Ns")) read_image(mi, OBJ_NS, c, path);

						if (!strcmp(key, "Ka")) read_ambient(mi, c);
						if (!strcmp(key, "Kd")) read_diffuse(mi, c);
						if (!strcmp(key, "Ks")) read_specular(mi, c);
						if (!strcmp(key, "Ke")) read_emission(mi, c);
						if (!strcmp(key, "Ns")) read_shininess(mi, c);
						if (!strcmp(key, "d"))  read_alpha(mi, c);
					}
                }
            }
        }
        fclose(fin);
    }
    else
    {
        EAZD_TRACE ("objMesh3D::read_material() : ERROR - File \"" << file << "\" is corrupt or does not exist.");
    }
}

void objMesh3D::read_mtllib(char *file, const char *line)
{
    /* Parse the first file name from the given line. */
    sscanf(line, "%s", file);
}

int objMesh3D::read_usemtl(const char *path, const char *file,
                           const char *line)
{
    char name[MAXSTR];

    int si;
    int mi;

    sscanf(line, "%s", name);

    /* Create a new material for the incoming definition. */
    if ((mi = obj_add_mtrl()) >= 0)
    {
        /* Create a new surface to contain geometry with the new material. */
        if ((si = obj_add_surf()) >= 0)
        {
            /* Read the material definition and apply it to the new surface. */
            read_material(path, file, name, mi);
            obj_set_surf(si, mi);

            /* Return the surface so that new geometry may be added to it. */
            return si;
        }
    }

    /* On failure, return the default surface. */
    return 0;
}

/*---------------------------------------------------------------------------*/

int objMesh3D::read_face_indices(const char *line, unsigned long *_vi, unsigned long *_ti, unsigned long *_ni)
{
    int n;

    *_vi = 0;
    *_ti = 0;
    *_ni = 0;

    /* Parse a face vertex specification from the given line. */
    if (sscanf(line, "%lu/%lu/%lu%n", _vi, _ti, _ni, &n) >= 3) return n;
    if (sscanf(line, "%lu/%lu%n",     _vi, _ti,      &n) >= 2) return n;
    if (sscanf(line, "%lu//%lu%n",    _vi,      _ni, &n) >= 2) return n;
    if (sscanf(line, "%lu%n",         _vi,           &n) >= 1) return n;

    return 0;
}

int objMesh3D::read_face_vertices(const char *line, int gi)
{
    const char *c = line;

    unsigned long _vi;
    unsigned long _ti;
    unsigned long _ni;
    int _ii;
    int _ij;

    int  dc;
    int  vi;
    int  ic = 0;

    /* Scan the face string, converting index sets to vertices. */
    while ((dc = read_face_indices(c, &_vi, &_ti, &_ni)))
    {
        /* Convert face indices to vector cache indices. */
        _vi += (_vi < 0) ? _vc : ULONG_MAX;
        _ti += (_ti < 0) ? _tc : ULONG_MAX;
        _ni += (_ni < 0) ? _nc : ULONG_MAX;

        /* Initialize a new index set. */
        if ((_ii = add_i()) >= 0)
        {
            _iv[_ii]._vi = _vi;
            _iv[_ii]._ni = _ni;
            _iv[_ii]._ti = _ti;

            /* Search the vector reference list for a repeated index set. */
            for (_ij = _vv[_vi]._ii; _ij >= 0; _ij = _iv[_ij]._ii)
                if (_iv[_ij]._vi == _vi &&
                    _iv[_ij]._ti == _ti &&
                    _iv[_ij]._ni == _ni &&
                    _iv[_ij]. gi ==  gi)
                {
                    /* A repeat has been found.  Link new to old. */
                    _vv[_vi]._ii = _ii;
                    _iv[_ii]._ii = _ij;
                    _iv[_ii]. vi = _iv[_ij].vi;
                    _iv[_ii]. gi = _iv[_ij].gi;

                    break;
                }

            /* If no repeat was found, add a new vertex. */
            if ((_ij < 0) && (vi = obj_add_vert()) >= 0)
            {
                _vv[_vi]._ii = _ii;
                _iv[_ii]._ii =  -1;
                _iv[_ii]. vi =  vi;
                _iv[_ii]. gi =  gi;

                /* Initialize the new vertex using valid cache references. */
                if (0 <= _vi && _vi < _vc)
                    obj_set_vert_coord(vi, _vv[_vi].v);
                if (0 <= _ni && _ni < _nc)
                    obj_set_vert_normal(vi, _nv[_ni].v);
                if (0 <= _ti && _ti < _tc)
                    obj_set_vert_texcoord(vi, _tv[_ti].v);
            }
            ic++;
        }
        c  += dc;
    }
    return ic;
}

void objMesh3D::read_face(const char *line, int si, int gi)
{
    Vector3D n;
    int i, pi;

    /* Create new vertices references for this face. */
    int i0 = _ic;
    int ic = read_face_vertices(line, gi);

    /* If smoothing, apply this face's normal to vertices that need it. */
    if (gi)
    {
        compute_normal(&n, _vv[_iv[i0 + 0]._vi].v,
                           _vv[_iv[i0 + 1]._vi].v,
                           _vv[_iv[i0 + 2]._vi].v);

        for (i = 0; i < ic; ++i)
            if (_iv[i0 + 0]._ni < 0)
            {
                normals[_iv[i0 + i]._vi] += n;
            }
    }

    /* Convert our N new vertex references into N-2 new triangles. */
    for (i = 0; i < ic - 2; ++i)
        if ((pi = obj_add_face(si)) >= 0)
        {
            int vi[3];

            vi[0] = _iv[i0        ].vi;
            vi[1] = _iv[i0 + i + 1].vi;
            vi[2] = _iv[i0 + i + 2].vi;

            obj_set_face(si, pi, vi);
        }
}

/*---------------------------------------------------------------------------*/

void objMesh3D::read_vertex(const char *line)
{
    unsigned long _vi;

    /* Parse a vertex position. */
    if ((_vi = add_v()) >= 0)
    {
        sscanf(line, "%f %f %f", _vv[_vi].v + 0,
                                 _vv[_vi].v + 1,
                                 _vv[_vi].v + 2);
        _vv[_vi]._ii = -1;
    }
}

void objMesh3D::read_texcoord(const char *line)
{
    unsigned long _ti;

    /* Parse a texture coordinate. */
    if ((_ti = add_t()) >= 0)
    {
        sscanf(line, "%f %f", _tv[_ti].v + 0,
                              _tv[_ti].v + 1);
        _tv[_ti]._ii = -1;
    }
}

void objMesh3D::read_normal(const char *line)
{
    unsigned long _ni;

    /* Parse a normal. */
    if ((_ni = add_n()) >= 0)
    {
        sscanf(line, "%f %f %f", _nv[_ni].v + 0,
                                 _nv[_ni].v + 1,
                                 _nv[_ni].v + 2);
        _nv[_ni]._ii = -1;
    }
}

/*---------------------------------------------------------------------------*/

void objMesh3D::obj_read_obj(const char *filename)
{
    char buf[MAXSTR];
    char key[MAXSTR];

    char L[MAXSTR];
    char D[MAXSTR];

    /* Flush the vector caches. */
    _vc = 0;
    _tc = 0;
    _nc = 0;
    _ic = 0;

    FILE *fin;

    /* Add the named file to the given object. */
    if ((fin = fopen(filename, "r")) == NULL)
    {
        EAZD_TRACE ("objMesh3D::readFormat() : ERROR - File \"" << filename << "\" is corrupt or does not exist.");
        return;
    }

    /* Ensure there exists a default surface 0 and default material 0. */
    unsigned long si = obj_add_surf();
    unsigned int  mi = obj_add_mtrl();
    int gi = 0;
    int n;

    obj_set_surf(si, mi);

    /* Extract the directory from the filename for use in MTL loading. */
    strncpy(D, filename, MAXSTR);
    dirpath(D);

    /* Process each line of the OBJ file, invoking the handler for each. */
    while  (fgets (buf, MAXSTR, fin))
    {
        if (sscanf(buf, "%s%n", key, &n) >= 1)
        {
            const char *c = buf + n;

                 if (! strcmp(key, "f" )) read_face (c, si, gi);
            else if (! strcmp(key, "vt")) read_texcoord(c);
            else if (! strcmp(key, "vn")) read_normal(c);
            else if (! strcmp(key, "v" )) read_vertex (c);

            else if (! strcmp(key, "mtllib"))      read_mtllib(L, c);
            else if (! strcmp(key, "usemtl")) si = read_usemtl(D, L, c);
            else if (! strcmp(key, "s"     )) gi = atoi(c);
        }
    }

    fclose(fin);
}

/*===========================================================================*/

unsigned long objMesh3D::obj_add_mtrl(void)
{
    const Vector3D Ka = Vector3D (0.2f, 0.2f, 0.2f);
    const Vector3D Kd = Vector3D (0.8f, 0.8f, 0.8f);
    const Vector3D Ks = Vector3D (0.0f, 0.0f, 0.0f);
    const Vector3D Ke = Vector3D (0.0f, 0.0f, 0.0f);

    unsigned long mi;

    /* Allocate and initialize a new material. */
    if ((mi = add__((void **) &materials, &nummaterials, &memmaterials,
                    sizeof (Material3D))) < ULONG_MAX)
    {
        memset (&(materials[mi]), 0, sizeof (Material3D));

        obj_set_mtrl_ambient(mi, Ka);
        obj_set_mtrl_diffuse(mi, Kd);
        obj_set_mtrl_specular(mi, Ks);
        obj_set_mtrl_emission(mi, Ke);
        obj_set_mtrl_shininess(mi, 8);
        obj_set_mtrl_alpha(mi, 1.0);
    }

    return mi;
}

unsigned long objMesh3D::obj_add_vert(void)
{
    unsigned long tni;
    unsigned long nri;
    unsigned long txi;
    unsigned long vri;

    // allocate the tangent vector
    if ((tni = add__((void **) &tangents, &numtangents, &memtangents,
                     sizeof (Vector3D))) < ULONG_MAX)
        memset (&(tangents[tni]), 0, sizeof (Vector3D));

    // allocate the normal vector
    if ((nri = add__((void **) &normals, &numnormals, &memnormals,
                     sizeof (Vector3D))) < ULONG_MAX)
        memset (&(normals[nri]), 0, sizeof (Vector3D));

    // allocate the texcoord vector
    if ((txi = add__((void **) &(texcoords[0]), &(numtexcoords[0]), &(memtexcoords[0]),
                     sizeof (Vector2D))) < ULONG_MAX)
        memset (&(texcoords[0][txi]), 0, sizeof (Vector2D));

    // allocate the coord vector
    if ((vri = add__((void **) &vertices, &numvertices, &memvertices,
                     sizeof (Vector3D))) < ULONG_MAX)
        memset (&(vertices[vri]), 0, sizeof (Vector3D));

    return vri;
}

unsigned long objMesh3D::obj_add_face(unsigned long si)
{
    unsigned long pi;

    assert_group(si);

    /* Allocate and initialize a new face. */
#ifdef BUFFER_OBJECT
    if ((pi = add__((void **) &(groups[si].faces),
                              &(groups[si].numfaces),
                              &(groups[si].memfaces),
                              sizeof (Face3D))) < ULONG_MAX)
        memset (&(groups[si].faces[pi]), 0, sizeof (Face3D));

    unsigned long ni;

    // allocate the face normal vector
    if ((ni = add__((void **) &(groups[si].fc_normals),
                              &(groups[si].numfc_normals),
                              &(groups[si].memfc_normals),
                    sizeof (Vector3D))) < ULONG_MAX)
        memset (&(groups[si].fc_normals[ni]), 0, sizeof (Vector3D));
#else // COMPILE_LIST
    if ((pi = add__((void **) &(groups[si].faces),
                              &(groups[si].numfaces),
                              &(groups[si].memfaces),
                              sizeof (Triangle3D))) < ULONG_MAX)
        memset (&(groups[si].faces[pi]), 0, sizeof (Triangle3D));
#endif

    return pi;
}

unsigned long objMesh3D::obj_add_surf(void)
{
    unsigned long si;

    /* Allocate and initialize a new surface. */
    if ((si = add__((void **) &groups, &numgroups, &memgroups,
                    sizeof (PrimitiveGroup3D))) < ULONG_MAX)
        memset (&(groups[si]), 0, sizeof (PrimitiveGroup3D));

    return si;
}

void objMesh3D::readFormat(const char *filename)
{
    assert (filename);

    bobject_opaque = 0;

    /* Read the named file. */
    obj_read_obj(filename);

    /* Post-process the loaded object. */
    obj_clean_object();
    // obj_transparency_object();
}

/*---------------------------------------------------------------------------*/

static void obj_rel_mtrl(Material3D *mp)
{
    /* Release any resources held by this material. */
    if (mp->texturestr[OBJ_KA]) free(mp->texturestr[OBJ_KA]);
    if (mp->texturestr[OBJ_KD]) free(mp->texturestr[OBJ_KD]);
    if (mp->texturestr[OBJ_KS]) free(mp->texturestr[OBJ_KS]);
    if (mp->texturestr[OBJ_KE]) free(mp->texturestr[OBJ_KE]);
    if (mp->texturestr[OBJ_NS]) free(mp->texturestr[OBJ_NS]);

    mp->has_texture[OBJ_KA] = false;
    mp->has_texture[OBJ_KD] = false;
    mp->has_texture[OBJ_KS] = false;
    mp->has_texture[OBJ_KE] = false;
    mp->has_texture[OBJ_NS] = false;

    if (mp->texturemap[OBJ_KA]) glDeleteTextures(1, &mp->texturemap[OBJ_KA]);
    if (mp->texturemap[OBJ_KD]) glDeleteTextures(1, &mp->texturemap[OBJ_KD]);
    if (mp->texturemap[OBJ_KS]) glDeleteTextures(1, &mp->texturemap[OBJ_KS]);
    if (mp->texturemap[OBJ_KE]) glDeleteTextures(1, &mp->texturemap[OBJ_KE]);
    if (mp->texturemap[OBJ_NS]) glDeleteTextures(1, &mp->texturemap[OBJ_NS]);
}

static void obj_rel_surf(PrimitiveGroup3D *sp)
{
    if (sp->pibo) glDeleteBuffers(1, &sp->pibo);

    sp->pibo = 0;

    /* Release this surface's face and line vectors. */
    if (sp->faces) free(sp->faces);
}

/*---------------------------------------------------------------------------*/

void objMesh3D::obj_del_object(void)
{
    unsigned long si;
    unsigned int  mi;

    if (bobject_opaque)
    {
        glDeleteBuffers(1, &bobject_opaque);
        bobject_opaque = 0;
    }

    /* Release resources held by this file and it's materials and surfaces. */
    for (mi = 0; mi < nummaterials; ++mi)
        obj_del_mtrl(mi);

    for (si = 0; si < numgroups; ++si)
        obj_rel_surf(&(groups[si]));
}

void objMesh3D::obj_del_mtrl(unsigned int mi)
{
    unsigned long si;

    assert_material(mi);

    /* Remove this material from the material vector. */
    obj_rel_mtrl(&(materials[mi]));

    memmove (&(materials[mi]),
             &(materials[mi + 1]),
           (nummaterials - mi - 1) * sizeof (Material3D));

    nummaterials--;

    /* Remove all references to this material. */
    for (si = numgroups - 1; (int) si >= 0; --si)
    {
        PrimitiveGroup3D *sp = &(groups[si]);

        if (sp->mtrlIdx == mi)
            obj_del_surf(si);
        else
        if (sp->mtrlIdx > mi)
            sp->mtrlIdx--;
    }
}

void objMesh3D::obj_del_face(unsigned long si, unsigned long pi)
{
    assert_face(si, pi);

    /* Remove this face from the surface's face vector. */
    memmove (&(groups[si].faces[pi]),
             &(groups[si].faces[pi + 1]),
#ifdef BUFFER_OBJECT
              (groups[si].numfaces - pi - 1) * sizeof (Face3D));
#else // COMPILE_LIST
              (groups[si].numfaces - pi - 1) * sizeof (Triangle3D));
#endif

    groups[si].numfaces--;
}

void objMesh3D::obj_del_surf(unsigned long si)
{
    assert_group(si);

    /* Remove this surface from the file's surface vector. */
    obj_rel_surf(&(groups[si]));

    memmove (&(groups[si]),
             &(groups[si + 1]),
            (numgroups - si - 1) * sizeof (PrimitiveGroup3D));

    numgroups--;
}

/*---------------------------------------------------------------------------*/

static char *set_name(char *old, const char *src)
{
    char *dst = NULL;

    if (old)
        free(old);

    if (src && (dst = (char *) malloc(strlen(src) + 1)))
        strcpy(dst, src);

    return dst;
}

void objMesh3D::obj_set_mtrl_name(unsigned int mi, const char *name)
{
    assert_material(mi);

    strcpy (materials[mi].name, name);
}

void objMesh3D::obj_set_mtrl_map(unsigned int mi, int ri, const char *str)
{
    assert_prop(mi, ri);

    if (materials[mi].has_texture[ri])
    {
        materials[mi].has_texture[ri] = false;
        glDeleteTextures(1, &(materials[mi].texturemap[ri]));
    }

    materials[mi].has_texture[ri] = true;
    materials[mi].texturemap[ri] = loadTexture(str);
    set_name (materials[mi].texturestr[ri], str);
}

void objMesh3D::obj_set_mtrl_ambient(unsigned int mi, const Vector3D &color)
{
    assert_material(mi);

    materials[mi].ambient = color;
}

void objMesh3D::obj_set_mtrl_diffuse(unsigned int mi, const Vector3D &color)
{
    assert_material(mi);

    materials[mi].diffuse = color;
}

void objMesh3D::obj_set_mtrl_specular(unsigned int mi, const Vector3D &color)
{
    assert_material(mi);

    materials[mi].specular = color;
}

void objMesh3D::obj_set_mtrl_emission(unsigned int mi, const Vector3D &color)
{
    assert_material(mi);

    materials[mi].emission = color;
}

void objMesh3D::obj_set_mtrl_shininess(unsigned int mi, const int shine)
{
    assert_material(mi);

    materials[mi].shininess = shine;
}

void objMesh3D::obj_set_mtrl_alpha(unsigned int mi, const float alpha)
{
    assert_material(mi);

    materials[mi].alpha = alpha;
}

/*---------------------------------------------------------------------------*/

void objMesh3D::invalidate(void)
{
    if (bobject_opaque)
    {
        glDeleteBuffers(1, &bobject_opaque);
        bobject_opaque = 0;
    }
}

void objMesh3D::obj_set_vert_coord(unsigned long vi, const float v[3])
{
    assert_vertex(vi);

    vertices[vi] = Vector3D (v[0], v[1], v[2]);

    invalidate();
}

void objMesh3D::obj_set_vert_texcoord(unsigned long vi, const float t[2])
{
    assert_vertex(vi);

    texcoords[0][vi] = Vector2D (t[0], t[1]);

    invalidate();
}

void objMesh3D::obj_set_vert_normal(unsigned long vi, const float n[3])
{
    assert_vertex(vi);

    normals[vi] = Vector3D (n[0], n[1], n[2]);

    invalidate();
}

/*---------------------------------------------------------------------------*/

void objMesh3D::obj_set_face(unsigned long si, unsigned long pi, const int vi[3])
{
    assert_face(si, pi);

    groups[si].faces[pi].vertIdx[0] = vi[0];
    groups[si].faces[pi].vertIdx[1] = vi[1];
    groups[si].faces[pi].vertIdx[2] = vi[2];
}

void objMesh3D::obj_set_surf(unsigned long si, unsigned int mi)
{
    assert_group(si);

    groups[si].mtrlIdx = mi;
}

/*===========================================================================*/

const char *objMesh3D::obj_get_mtrl_name(unsigned int mi)
{
    assert_material(mi);

    return materials[mi].name;
}

unsigned int objMesh3D::obj_get_mtrl_map(unsigned int mi, int ri)
{
    assert_prop(mi, ri);

    return materials[mi].texturemap[ri];
}

void objMesh3D::obj_get_mtrl_ambient(unsigned int mi, Vector3D &color)
{
    assert_material(mi);

    color = materials[mi].ambient;
}

void objMesh3D::obj_get_mtrl_diffuse(unsigned int mi, Vector3D &color)
{
    assert_material(mi);

    color = materials[mi].diffuse;
}

void objMesh3D::obj_get_mtrl_specular(unsigned int mi, Vector3D &color)
{
    assert_material(mi);

    color = materials[mi].specular;
}

void objMesh3D::obj_get_mtrl_emission(unsigned int mi, Vector3D &color)
{
    assert_material(mi);

    color = materials[mi].emission;
}

/*---------------------------------------------------------------------------*/

void objMesh3D::obj_get_vert_coord(unsigned long vi, float v[3])
{
    assert_vertex(vi);

    v[0] = vertices[vi][0];
    v[1] = vertices[vi][1];
    v[2] = vertices[vi][2];
}

void objMesh3D::obj_get_vert_texcoord(unsigned long vi, float t[2])
{
    assert_vertex(vi);

    t[0] = texcoords[0][vi][0];
    t[1] = texcoords[0][vi][1];
}

void objMesh3D::obj_get_vert_normal(unsigned long vi, float n[3])
{
    assert_vertex(vi);

    n[0] = normals[vi][0];
    n[1] = normals[vi][1];
    n[2] = normals[vi][2];
}

/*---------------------------------------------------------------------------*/

void objMesh3D::obj_get_face(unsigned long si, unsigned long pi, int vi[3])
{
    assert_face(si, pi);

    vi[0] = groups[si].faces[pi].vertIdx[0];
    vi[1] = groups[si].faces[pi].vertIdx[1];
    vi[2] = groups[si].faces[pi].vertIdx[2];
}

int objMesh3D::obj_get_surf(unsigned long si)
{
    assert_group(si);

    return groups[si].mtrlIdx;
}

/*===========================================================================*/

void objMesh3D::obj_clean_object(void)
{
    unsigned long si;
    unsigned int  mi;

    /* Remove empty surfaces. */
    for (si = numgroups - 1; (int) si >= 0; --si)
        if (groups[si].numfaces == 0)
            obj_del_surf(si);

    /* Remove unreferenced materials. */
    for (mi = nummaterials - 1; (int) mi >= 0; --mi)
    {
        int cc = 0;

        for (si = 0; si < numgroups; ++si)
            if (groups[si].mtrlIdx == mi)
                cc++;

        if (cc == 0)
            obj_del_mtrl(mi);
    }
}

void objMesh3D::obj_normals_object(void)
{
    unsigned long si;
    unsigned long pi;

    /* Compute normals for all faces. */
    for (si = 0; si < numgroups; ++si)
        for (pi = 0; pi < groups[si].numfaces; ++pi)
        {
#ifdef BUFFER_OBJECT
            Face3D *p = &(groups[si].faces[pi]);
#else // COMPILE_LIST
            Triangle3D *p = &(groups[si].faces[pi]);
#endif

            Vector3D *v0 = &(vertices[p->vertIdx[0]]);
            Vector3D *v1 = &(vertices[p->vertIdx[1]]);
            Vector3D *v2 = &(vertices[p->vertIdx[2]]);

            /* Compute the normal formed by these 3 vertices. */
#ifdef BUFFER_OBJECT
            compute_normal(&(groups[si].fc_normals[pi]), *v0, *v1, *v2);
#else // COMPILE_LIST
            compute_normal(&(p->fc_normal), *v0, *v1, *v2);
#endif
        }
}

void objMesh3D::obj_tangents_object(void)
{
    unsigned long si;
    unsigned long pi;
    unsigned long vi;

    assert (numnormals == numtangents);

    /* Normalize all normals.  Zero all tangent vectors. */
    for (vi = 0; vi < numnormals; ++vi)
    {
        normals[vi].normalize();

        tangents[vi] = Vector3D(0.0f, 0.0f, 0.0f);
    }

    /* Compute tangent vectors for all vertices. */
    for (si = 0; si < numgroups; ++si)
        for (pi = 0; pi < groups[si].numfaces; ++pi)
        {
#ifdef BUFFER_OBJECT
            Face3D *p = &(groups[si].faces[pi]);
#else // COMPILE_LIST
            Triangle3D *p = &(groups[si].faces[pi]);
#endif

            Vector3D *v0 = &(vertices[p->vertIdx[0]]);
            Vector3D *v1 = &(vertices[p->vertIdx[1]]);
            Vector3D *v2 = &(vertices[p->vertIdx[2]]);

            Vector2D *tx0 = &(texcoords[0][p->vertIdx[0]]);
            Vector2D *tx1 = &(texcoords[0][p->vertIdx[1]]);
            Vector2D *tx2 = &(texcoords[0][p->vertIdx[2]]);

            Vector3D *tn0 = &(tangents[p->vertIdx[0]]);
            Vector3D *tn1 = &(tangents[p->vertIdx[1]]);
            Vector3D *tn2 = &(tangents[p->vertIdx[2]]);

            Vector3D dv1, dv2, t;

            /* Compute the tangent vector for this face. */
            dv1 = Vector3D ((*v1)[0] - (*v0)[0],
                            (*v1)[1] - (*v0)[1],
                            (*v1)[2] - (*v0)[2]);

            dv2 = Vector3D ((*v2)[0] - (*v0)[0],
                            (*v2)[1] - (*v0)[1],
                            (*v2)[2] - (*v0)[2]);

            float dt1 = (*tx1)[1] - (*tx0)[1];
            float dt2 = (*tx2)[1] - (*tx0)[1];

            t = Vector3D (dt2 * dv1[0] - dt1 * dv2[0],
                          dt2 * dv1[1] - dt1 * dv2[1],
                          dt2 * dv1[2] - dt1 * dv2[2]);
            t.normalize();

            /* Accumulate the tangent vectors for this face's vertices. */
            *tn0 += t;
            *tn1 += t;
            *tn2 += t;
        }

    /* Orthonormalize each tangent basis. */
    for (vi = 0; vi < numnormals; ++vi)
    {
        Vector3D *n = &(normals[vi]);
        Vector3D *u = &(tangents[vi]);

        Vector3D v = Vector3D::cross(*n, *u);
        *u = Vector3D::cross(v, *n);
        u->normalize();
    }
}

void objMesh3D::obj_transparency_object(void)
{
    unsigned long si;
    unsigned long sj;

    /* Sort surfaces such that transparent ones appear later. */
    for (si = 0; si < numgroups; ++si)
        for (sj = si + 1; sj < numgroups; ++sj)
            if (materials[groups[si].mtrlIdx].alpha <
                materials[groups[sj].mtrlIdx].alpha)
            {
                /* Swap the best group into the current position. */
                PrimitiveGroup3D t = groups[si];
                groups[si]         = groups[sj];
                groups[sj]         = t;
            }
}

#ifdef BUFFER_OBJECT
void objMesh3D::obj_init(void)
{
    if (! (bobject_opaque == 0 && GL_has_vertex_buffer_object))
        return;

    /* Store all vertex data in a vertex buffer object. */
    glGenBuffers(1, &bobject_opaque);
    glBindBuffer(GL_ARRAY_BUFFER, bobject_opaque);

    GLsizeiptr sizetangents  = numtangents     * sizeof (Vector3D);
    GLsizeiptr sizenormals   = numnormals      * sizeof (Vector3D);
    GLsizeiptr sizetexcoords = numtexcoords[0] * sizeof (Vector2D);
    GLsizeiptr sizevertices  = numvertices     * sizeof (Vector3D);

    // pointers relative to the start of the vbo in video memory
    ptrtangents  = 0;
    ptrnormals   = ptrtangents  + sizetangents;
    ptrtexcoords = ptrnormals   + sizenormals;
    ptrvertices  = ptrtexcoords + sizetexcoords;

    // upload the data to the graphics memory
    glBufferData(GL_ARRAY_BUFFER,
        sizetangents + sizenormals + sizetexcoords + sizevertices,
        0, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, ptrtangents,  sizetangents,  tangents);
    glBufferSubData(GL_ARRAY_BUFFER, ptrnormals,   sizenormals,   normals);
    glBufferSubData(GL_ARRAY_BUFFER, ptrtexcoords, sizetexcoords, texcoords[0]);
    glBufferSubData(GL_ARRAY_BUFFER, ptrvertices,  sizevertices,  vertices);

    unsigned long si;

    /* Store all index data in index buffer objects. */
    for (si = 0; si < numgroups; ++si)
    {
        if (groups[si].numfaces > 0)
        {
            glGenBuffers(1, &(groups[si].pibo));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, groups[si].pibo);

            glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                         groups[si].numfaces * sizeof (Face3D),
                         groups[si].faces, GL_STATIC_DRAW);
        }
    }
}
#endif

/*---------------------------------------------------------------------------*/

void objMesh3D::obj_sort_object(int qc)
{
    const int vc = numvertices;

    struct vert
    {
        int  qs;    /* Cache insertion serial number */
        int *iv;    /* face reference list buffer */
        int  ic;    /* face reference list length */
    };

    /* Vertex optimization data; vertex FIFO cache */
    struct vert *vv = (struct vert *) malloc(vc * sizeof (struct vert));
    int         *qv = (int         *) malloc(qc * sizeof (int        ));

    int qs = 1;   /* Current cache insertion serial number */
    int qi = 0;   /* Current cache insertion point [0, qc) */

    unsigned long si;
    int pi;
    int vi;
    int ii;
    int qj;

    /* Initialize the vertex cache to empty. */
    for (qj = 0; qj < qc; ++qj)
        qv[qj] = -1;

    /* Process each surface of this file in turn. */
    for (si = 0; si < numgroups; ++si)
    {
        const int pc = groups[si].numfaces;

        /* Allocate the face reference list buffers. */
        int *ip, *iv = (int *) malloc(3 * pc * sizeof (int));

        /* Count the number of face references per vertex. */
        memset (vv, 0, vc * sizeof (struct vert));

        for (pi = 0; pi < pc; ++pi)
        {
            unsigned long *i = groups[si].faces[pi].vertIdx;

            vv[i[0]].ic++;
            vv[i[1]].ic++;
            vv[i[2]].ic++;
        }

        /* Initialize all vertex optimization data. */
        for (vi = 0, ip = iv; vi < vc; ++vi)
        {
            vv[vi].qs = -qc;
            vv[vi].iv =  ip;
            ip += vv[vi].ic;
            vv[vi].ic =   0;
        }

        /* Fill the face reference list buffers. */
        for (pi = 0; pi < pc; ++pi)
        {
            unsigned long *i = groups[si].faces[pi].vertIdx;

            vv[i[0]].iv[vv[i[0]].ic++] = pi;
            vv[i[1]].iv[vv[i[1]].ic++] = pi;
            vv[i[2]].iv[vv[i[2]].ic++] = pi;
        }

        /* Iterate over the face array of this surface. */
        for (pi = 0; pi < pc; ++pi)
        {
            unsigned long *i = groups[si].faces[pi].vertIdx;

            int qd = qs - qc;

            int dk = -1;    /* The best face score */
            int pk = pi;    /* The best face index */

            /* Find the best face among those referred-to by the cache. */
            for (qj = 0; qj < qc; ++qj)
                if (qv[qj] >= 0)

                    for (ii = 0;  ii < vv[qv[qj]].ic; ++ii)
                    {
                        int pj = vv[qv[qj]].iv[ii];
                        int dj = 0;

                        unsigned long *j = groups[si].faces[pj].vertIdx;

                        /* Recently-used vertex bonus. */
                        if (vv[j[0]].qs > qd) dj += vv[j[0]].qs - qd;
                        if (vv[j[1]].qs > qd) dj += vv[j[1]].qs - qd;
                        if (vv[j[2]].qs > qd) dj += vv[j[2]].qs - qd;

                        /* Low-valence vertex bonus. */
                        dj -= vv[j[0]].ic;
                        dj -= vv[j[1]].ic;
                        dj -= vv[j[2]].ic;

                        if (dk < dj)
                        {
                            dk = dj;
                            pk = pj;
                        }
                    }

            if (pk != pi)
            {
                /* Update the face reference list. */
                for (vi = 0; vi < 3; ++vi)
                    for (ii = 0; ii < vv[i[vi]].ic; ++ii)
                        if (vv[i[vi]].iv[ii] == pi)
                        {
                            vv[i[vi]].iv[ii] =  pk;
                            break;
                        }

                /* Swap the best face into the current position. */
#ifdef BUFFER_OBJECT
                Face3D t;
#else // COMPILE_LIST
                Triangle3D t;
#endif
                                   t = groups[si].faces[pi];
                groups[si].faces[pi] = groups[si].faces[pk];
                groups[si].faces[pk] = t;
            }

            /* Iterate over the current face's vertices. */
            for (vi = 0; vi < 3; ++vi)
            {
                struct vert *vp = vv + i[vi];

                /* If this vertex was a cache miss then queue it. */
                if (qs - vp->qs >= qc)
                {
                    vp->qs = qs++;
                    qv[qi] = i[vi];
                    qi = (qi + 1) % qc;
                }

                /* Remove the current face from the reference list. */
                vp->ic--;

                for (ii = 0; ii < vp->ic; ++ii)
                    if (vp->iv[ii] == pk)
                    {
                        vp->iv[ii] = vp->iv[vp->ic];
                        break;
                    }
            }
        }
        free(iv);
    }
    free(qv);
    free(vv);
}

float objMesh3D::obj_acmr_object(int qc)
{
    int *vs = (int *) malloc(numvertices * sizeof (int));
    int  qs = 1;

    unsigned long si;
    unsigned long vi;
    unsigned long pi;

    int nn = 0;
    int dd = 0;

    for (si = 0; si < numgroups; ++si)
    {
        for (vi = 0; vi < numvertices; ++vi)
            vs[vi] = -qc;

        for (pi = 0; pi < groups[si].numfaces; ++pi)
        {
            unsigned long *i = groups[si].faces[pi].vertIdx;

            if (qs - vs[i[0]] >= qc) { vs[i[0]] = qs++; nn++; }
            if (qs - vs[i[1]] >= qc) { vs[i[1]] = qs++; nn++; }
            if (qs - vs[i[2]] >= qc) { vs[i[2]] = qs++; nn++; }

            dd++;
        }
    }

    return (float) nn / (float) dd;
}

/*---------------------------------------------------------------------------*/

void objMesh3D::obj_draw_prop(int mi, int ri)
{
    Material3D *km = &(materials[mi]);

    if (km->has_texture[ri])
    {
        GLenum wrap = GL_REPEAT;

        /* Bind the property map. */
        glBindTexture(GL_TEXTURE_2D, km->texturemap[ri]);
        glEnable(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);

        glMatrixMode(GL_MODELVIEW);
    }
}

#define BUFFER_OFFSET(i) ((char *) NULL + (i))

void objMesh3D::obj_draw_vert(void)
{
    GLsizei s = 0;

    obj_init_gl();

    /* Enable all necessary vertex attribute pointers. */
    glEnableVertexAttribArray(6);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    if (GL_has_vertex_buffer_object)
    {
        /* Bind attributes to a vertex buffer object. */
        glBindBuffer(GL_ARRAY_BUFFER, bobject_opaque);

        glVertexAttribPointer(6, 3, GL_FLOAT, 0, s, BUFFER_OFFSET(ptrtangents));
        glNormalPointer      (      GL_FLOAT,    s, BUFFER_OFFSET(ptrnormals));
        glTexCoordPointer    (2,    GL_FLOAT,    s, BUFFER_OFFSET(ptrtexcoords));
        glVertexPointer      (3,    GL_FLOAT,    s, BUFFER_OFFSET(ptrvertices));
    }
    else
    {
        /* Bind attributes in main memory. */
        glVertexAttribPointer(6, 3, GL_FLOAT, 0, s, tangents);
        glNormalPointer      (      GL_FLOAT,    s, normals);
        glTexCoordPointer    (2,    GL_FLOAT,    s, texcoords[0]);
        glVertexPointer      (3,    GL_FLOAT,    s, vertices);
    }
}

void objMesh3D::obj_draw_mtrl(int mi)
{
    obj_init_gl();

    /* Bind as many of the texture maps as the GL implementation will allow. */
    if (GL_has_multitexture)
    {
        if (GL_max_texture_image_units > 3 && obj_get_mtrl_map(mi, OBJ_NS))
        {
            glActiveTexture(GL_TEXTURE3);
            obj_draw_prop(mi, OBJ_NS);
        }
        if (GL_max_texture_image_units > 2 && obj_get_mtrl_map(mi, OBJ_KS))
        {
            glActiveTexture(GL_TEXTURE2);
            obj_draw_prop(mi, OBJ_KS);
        }
        if (GL_max_texture_image_units > 1 && obj_get_mtrl_map(mi, OBJ_KA))
        {
            glActiveTexture(GL_TEXTURE1);
            obj_draw_prop(mi, OBJ_KA);
        }
        if (GL_max_texture_image_units > 0 && obj_get_mtrl_map(mi, OBJ_KD))
        {
            glActiveTexture(GL_TEXTURE0);
            obj_draw_prop(mi, OBJ_KD);
        }
    }
    else
        obj_draw_prop(mi, OBJ_KD);

    /* Apply the material properties. */
    float attrib[4];
    attrib[0] = materials[mi].ambient[0];
    attrib[1] = materials[mi].ambient[1];
    attrib[2] = materials[mi].ambient[2];
    attrib[3] = materials[mi].alpha;
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,  attrib);
    attrib[0] = materials[mi].diffuse[0];
    attrib[1] = materials[mi].diffuse[1];
    attrib[2] = materials[mi].diffuse[2];
    attrib[3] = materials[mi].alpha;
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, attrib);
    attrib[0] = materials[mi].specular[0];
    attrib[1] = materials[mi].specular[1];
    attrib[2] = materials[mi].specular[2];
    attrib[3] = 1.0f;
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, attrib);
    glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, materials[mi].shininess);
}

void objMesh3D::obj_draw_surf(unsigned long si)
{
    PrimitiveGroup3D *sp = &(groups[si]);

    obj_init_gl();

    if (0 < sp->numfaces)
    {
        /* Apply this surface's material. */
        if (0 <= sp->mtrlIdx && sp->mtrlIdx < nummaterials)
            obj_draw_mtrl(sp->mtrlIdx);

        /* Render all faces. */
        if (sp->pibo)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sp->pibo);
            glDrawElements(GL_TRIANGLES, 3 * sp->numfaces,
                           GL_UNSIGNED_INT, BUFFER_OFFSET(0));
        }

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
}

void objMesh3D::obj_draw_object(void)
{
    unsigned long si;

    obj_init_gl();
    obj_init();

    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_ENABLE_BIT);
    {
        glDisable(GL_COLOR_MATERIAL);

        /* Load the vertex buffer. */
        obj_draw_vert();

        /* Render each surface. */
        for (si = 0; si < numgroups; ++si)
            obj_draw_surf(si);
    }
    glPopAttrib();
    glPopClientAttrib();
}

void objMesh3D::obj_draw_axes(float k)
{
    unsigned long vi;

    obj_init_gl();
    obj_init();

    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT);
    {
        glEnable(GL_COLOR_MATERIAL);

        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);

        glBegin(GL_LINES);
        {
            for (vi = 0; vi < numvertices; ++vi)
            {
                Vector3D p = vertices[vi];
                Vector3D x = tangents[vi];
                Vector3D z = normals[vi];

                /* Compute the bitangent vector. */
                Vector3D y = Vector3D (
                       z[1] * x[2] - z[2] * x[1],
                       z[2] * x[0] - z[0] * x[2],
                       z[0] * x[1] - z[1] * x[0]);

                /* Render the tangent-bitangent-normal basis. */
                glColor3f(1.0f, 0.0f, 0.0f);

                glVertex3f(p[0],            p[1],            p[2]);
                glVertex3f(p[0] + x[0] * k, p[1] + x[1] * k, p[2] + x[2] * k);

                glColor3f(0.0f, 1.0f, 0.0f);

                glVertex3f(p[0],            p[1],            p[2]);
                glVertex3f(p[0] + y[0] * k, p[1] + y[1] * k, p[2] + y[2] * k);

                glColor3f(0.0f, 0.0f, 1.0f);

                glVertex3f(p[0],            p[1],            p[2]);
                glVertex3f(p[0] + z[0] * k, p[1] + z[1] * k, p[2] + z[2] * k);
            }
        }
        glEnd();
    }
    glPopAttrib();
}

/*===========================================================================*/

void objMesh3D::obj_bound_object(void)
{
    Vector3D vmin, vmax, sz;

    unsigned long vi;

    /* Compute the bounding box of this object. */
    if (numvertices > 0)
    {
        Vector3D v = vertices[0];

        vmin.x = vmax.x = v[0];
        vmin.y = vmax.y = v[1];
        vmin.z = vmax.z = v[2];
    }

    for (vi = 0; vi < numvertices; ++vi)
    {
        Vector3D v = vertices[vi];

        if (vmin.x > v[0]) vmin.x = v[0];
        if (vmin.y > v[1]) vmin.y = v[1];
        if (vmin.z > v[2]) vmin.z = v[2];

        if (vmax.x < v[0]) vmax.x = v[0];
        if (vmax.y < v[1]) vmax.y = v[1];
        if (vmax.z < v[2]) vmax.z = v[2];
    }
    center.x = (vmax.x + vmin.x) * 0.5f;
    center.y = (vmax.y + vmin.y) * 0.5f;
    center.z = (vmax.z + vmin.z) * 0.5f;
    sz = vmax - vmin;
    bbox.setSymmetrical (center,sz);
}

/*===========================================================================*/

void objMesh3D::obj_write_map(FILE *fout, int mi, int ri, const char *s)
{
    Material3D *km = &(materials[mi]);

    /* If this property has a map... */
    if (km->texturestr[ri])
    {
        fprintf(fout, "map_%s ", s);

        /* Store the map image file name. */
        fprintf(fout, "%s\n", km->texturestr[ri]);
    }
}

void objMesh3D::obj_write_mtl(const char *mtl)
{
    FILE *fout;
    unsigned long mi;

    if ((fout = fopen(mtl, "w")))
    {
        for (mi = 0; mi < nummaterials; ++mi)
        {
            Material3D *mp = &(materials[mi]);

            /* Start a new material. */
            fprintf(fout, "newmtl %s\n", (mp->name) ? mp->name : "default");

            /* Store all material property colors. */
            fprintf(fout, "Ka %12.8f %12.8f %12.8f\n"
                          "Kd %12.8f %12.8f %12.8f\n"
                          "Ks %12.8f %12.8f %12.8f\n"
                          "Ke %12.8f %12.8f %12.8f\n",
                    mp->ambient [0], mp->ambient [1], mp->ambient [2],
                    mp->diffuse [0], mp->diffuse [1], mp->diffuse [2],
                    mp->specular[0], mp->specular[1], mp->specular[2],
                    mp->emission[0], mp->emission[1], mp->emission[2]);

            fprintf(fout, "Ns %d\n", mp->shininess);
            fprintf(fout, "d %12.8f\n", mp->alpha);

            /* Store all material property maps. */
            obj_write_map(fout, mi, OBJ_KA, "Ka");
            obj_write_map(fout, mi, OBJ_KD, "Kd");
            obj_write_map(fout, mi, OBJ_KS, "Ks");
            obj_write_map(fout, mi, OBJ_KE, "Ke");
            obj_write_map(fout, mi, OBJ_NS, "Ns");
        }
    }
    fclose(fout);
}

void objMesh3D::obj_write_obj(const char *obj, const char *mtl)
{
    FILE *fout;

    if ((fout = fopen(obj, "w")) == NULL)
        return;

    unsigned long si;
    unsigned long vi;
    unsigned long pi;

    if (mtl) fprintf(fout, "mtllib %s\n\n", mtl);

    /* Store all vertex data. */
    fprintf (fout, "# vertices: %lu\n", numvertices);
    for (vi = 0; vi < numvertices; ++vi)
        fprintf(fout, "v  %12.8f %12.8f %12.8f\n",
                vertices[vi][0], vertices[vi][1], vertices[vi][2]);
    fprintf (fout, "\n# tex coords: %lu\n", numtexcoords[0]);
    for (vi = 0; vi < numtexcoords[0]; ++vi)
        fprintf(fout, "vt %12.8f %12.8f\n",
                texcoords[0][vi][0], texcoords[0][vi][1]);
    fprintf (fout, "\n# normals: %lu\n", numnormals);
    for (vi = 0; vi < numnormals; ++vi)
        fprintf(fout, "vn %12.8f %12.8f %12.8f\n",
                normals[vi][0], normals[vi][1], normals[vi][2]);
    fprintf (fout, "\n");

    for (si = 0; si < numgroups; ++si)
    {
        unsigned long mi = groups[si].mtrlIdx;

        /* Store the surface's material reference */
        if (0 <= mi && mi < nummaterials && materials[mi].name)
            fprintf(fout, "usemtl %s\n", materials[groups[si].mtrlIdx].name);
        else
            fprintf(fout, "usemtl default\n");

        /* Store all face definitions. */
        fprintf (fout, "\n# faces: %lu\n", groups[si].numfaces);
        for (pi = 0; pi < groups[si].numfaces; pi++)
        {
            int vi0 = groups[si].faces[pi].vertIdx[0] + 1;
            int vi1 = groups[si].faces[pi].vertIdx[1] + 1;
            int vi2 = groups[si].faces[pi].vertIdx[2] + 1;

            fprintf(fout, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", vi0, vi0, vi0,
                                                            vi1, vi1, vi1,
                                                            vi2, vi2, vi2);
        }
    }

    fclose(fout);
}

static char   *
getFileExtension (const char *fileName)
{
    char *ext;

    /* look for final "." in fileName */
    if ((ext = (char *) strrchr (fileName, '.')) != NULL)
        /* advance "ext" past the period character */
        ++ext;

    return ext;
}

void objMesh3D::writeFormat(const char *filename)
{
    assert (filename);

    char mtlname[256];
    strncpy (mtlname, filename, 256);
    char * ext = getFileExtension (mtlname);
    strcpy (ext, "mtl");

    obj_write_obj(filename, mtlname);
    obj_write_mtl(mtlname);
}

