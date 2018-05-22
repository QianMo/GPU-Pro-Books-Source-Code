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

#pragma once

#include <assert.h>

#include "Vector3D.h"
#include "Mesh3D.h"

/*---------------------------------------------------------------------------*/

#ifdef EAZD_DEBUG
#define assert_group(j)    assert(0 <= j && j < numgroups)
#define assert_face(j, k)  assert_group(j); assert(0 <= k && k < groups[j].numfaces)
#define assert_vertex(j)   assert(0 <= j && j < numvertices)
#define assert_material(j) assert(0 <= j && j < nummaterials)
#define assert_prop(j, k)  assert_material(j); assert(0 <= k && k < OBJ_COUNT)
#else
#define assert_group(j)
#define assert_face(j, k)
#define assert_vertex(j)
#define assert_material(j)
#define assert_prop(j, k)
#endif

/*===========================================================================*/

#define OBJ_KA 0
#define OBJ_KD 1
#define OBJ_KS 2
#define OBJ_KE 3
#define OBJ_NS 4

#define OBJ_COUNT (OBJ_NS+1)

/*===========================================================================*/

/*---------------------------------------------------------------------------*/
/* Global OBJ State                                                          */

class objMesh3D : public Mesh3D
{
public:
    void  readFormat (const char *);
    void  writeFormat (const char *);

    void  obj_sort_object (int);
    float obj_acmr_object (int);
    void  obj_bound_object (void);
    void  obj_draw_object (void);
    void  obj_del_object (void);

    void  obj_normals_object (void);
    void  obj_tangents_object (void);

private:
    unsigned long obj_add_mtrl (void);
    unsigned long obj_add_vert (void);
    unsigned long obj_add_face (unsigned long);
    unsigned long obj_add_surf (void);

    unsigned long obj_num_mtrl(void) { return nummaterials; }
    unsigned long obj_num_vert(void) { return numvertices; }
    unsigned long obj_num_surf(void) { return numgroups; }

    unsigned long obj_num_face (unsigned long si)
    {
        assert_group(si);

        return groups[si].numfaces;
    }

    void obj_del_mtrl (unsigned int);
    void obj_del_vert (int);
    void obj_del_face (unsigned long, unsigned long);
    void obj_del_surf (unsigned long);

/*---------------------------------------------------------------------------*/

    void obj_set_mtrl_name (unsigned int, const char *);
    void obj_set_mtrl_map (unsigned int, int, const char *);
    void obj_set_mtrl_ambient (unsigned int, const Vector3D&);
    void obj_set_mtrl_diffuse (unsigned int, const Vector3D&);
    void obj_set_mtrl_specular (unsigned int, const Vector3D&);
    void obj_set_mtrl_emission (unsigned int, const Vector3D&);
    void obj_set_mtrl_shininess (unsigned int, const int);
    void obj_set_mtrl_alpha (unsigned int, const float);
    void obj_set_mtrl_texcoord_offset (unsigned int, int, const float[3]);
    void obj_set_mtrl_texcoord_scale (unsigned int, int, const float[3]);

    void obj_set_vert_coord (unsigned long, const float[3]);
    void obj_set_vert_texcoord (unsigned long, const float[2]);
    void obj_set_vert_normal (unsigned long, const float[3]);

    void obj_set_face (unsigned long, unsigned long, const int[3]);
    void obj_set_surf (unsigned long, unsigned int);

/*---------------------------------------------------------------------------*/

    const char  *obj_get_mtrl_name (unsigned int);
    unsigned int obj_get_mtrl_map (unsigned int, int);
    void         obj_get_mtrl_ambient (unsigned int, Vector3D&);
    void         obj_get_mtrl_diffuse (unsigned int, Vector3D&);
    void         obj_get_mtrl_specular (unsigned int, Vector3D&);
    void         obj_get_mtrl_emission (unsigned int, Vector3D&);
    void         obj_get_mtrl_texcoord_offset (unsigned int, int, float[3]);
    void         obj_get_mtrl_texcoord_scale (unsigned int, int, float[3]);

    void obj_get_vert_coord (unsigned long, float[3]);
    void obj_get_vert_texcoord (unsigned long, float[2]);
    void obj_get_vert_normal (unsigned long, float[3]);

    void obj_get_face (unsigned long, unsigned long, int[3]);
    int  obj_get_surf (unsigned long);

/*---------------------------------------------------------------------------*/

    void  compute_normal(Vector3D *, Vector3D, Vector3D, Vector3D);

    void  obj_clean_object (void);
    void  obj_transparency_object (void);

    void  obj_draw_vert (void);
    void  obj_draw_mtrl (int);
    void  obj_draw_surf (unsigned long);
    void  obj_draw_axes (float);

private:
#if 0
    GLuint bobject;             // vertex buffer object

    GLintptr ptrtangents;
    GLintptr ptrnormals;
    GLintptr ptrtexcoords;
    GLintptr ptrvertices;

    // memory allocated
    unsigned long memtangents;
    unsigned long memtexcoords[3];
    unsigned long memvertices;
    unsigned long memnormals;

    unsigned long memmaterials;
    unsigned long memgroups;
#endif
    void read_image (int mi, int ri, const char *line,
                                     const char *path);
    void read_ambient (int mi, const char *line);
    void read_diffuse (int mi, const char *line);
    void read_specular (int mi, const char *line);
    void read_emission (int mi, const char *line);
    void read_alpha (int mi, const char *line);
    void read_shininess (int mi, const char *line);
    void read_material (const char *path, const char *file,
                        const char *name, int mi);
    void read_mtllib (char *file, const char *line);
    int  read_usemtl (const char *path, const char *file,
                      const char *line);
    int  read_face_indices (const char *line, unsigned long *_vi, unsigned long *_ti, unsigned long *_ni);
    int  read_face_vertices (const char *line, int gi);
    void read_face (const char *line, int si, int gi);
    void read_vertex (const char *line);
    void read_texcoord (const char *line);
    void read_normal (const char *line);

    void obj_init (void);

    void obj_read_obj (const char *filename);
    void obj_draw_prop (int mi, int ri);

    void obj_write_map (FILE *fout, int mi, int ri, const char *s);
    void obj_write_mtl (const char *mtl);
    void obj_write_obj (const char *obj, const char *mtl);

    void invalidate (void);
};

