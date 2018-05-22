
#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	#pragma warning (disable : 4996)
#endif

#include <stdlib.h>

#ifdef INTEL_COMPILER
	#include <dvec.h>
	#include <mathimf.h>
#else
	#include <math.h>
#endif

#include <string.h>
#include <float.h>
#include <fstream>

#ifdef WIN32
	#include <GL/glew.h>
#else
	// this should be defined before glext.h
	#define GL_GLEXT_PROTOTYPES

	#include <GL/gl.h>
	#include <GL/glext.h>
#endif

#include "Material3D.h"
#include "Mesh3D.h"

#ifdef BUFFER_OBJECT
	#include "obj_vbo.h"
#else // COMPILE_LIST
	#include "obj.h"
#endif

#include "raw.h"

Mesh3D::Mesh3D(char *filename, vector<char *>searchpaths)
{
	addSearchPaths(searchpaths);
	this->filename[0] = '\0';
	strncpy(this->filename, strrchr (filename, DIR_DELIMITER) + 1, 256);
	
	vertices = NULL;
	normals = NULL;
	tangents = NULL;
	texcoords[0] = texcoords[1] = texcoords[2] = NULL;
	materials = NULL;
	groups = NULL;
	spatialSubd = NULL;
	if (paths.empty())
		paths.push_back(STR_DUP("."));
	
	center = Vector3D(0,0,0);
	
	double_sided = false;
	numfaces = numvertices = numnormals = numtangents =
		numtexcoords[0] = numtexcoords[1] = numtexcoords[2] =
		numgroups = nummaterials = 0;
#ifdef BUFFER_OBJECT
	memtangents = memvertices = memnormals =
		memtexcoords[0] = memtexcoords[1] = memtexcoords[2] =
		memmaterials = memgroups = 0;
	bobject_opaque = 0;
#endif
	
	dlist_opaque = dlist_transparent = 0;
	
	readFormat(filename);
	processData();
}

Mesh3D::Mesh3D(char *filename)
{
	vector <char *> searchpaths;
	
	// if at this point you call the above constructor
	// (since the code is the same) the destructor
	// is being called removing all data (go figure)
	
	addSearchPaths(searchpaths);
	this->filename[0] = '\0';
	strncpy(this->filename, strrchr (filename, DIR_DELIMITER) + 1, 256);
	
	vertices = NULL;
	normals = NULL;
	tangents = NULL;
	texcoords[0] = texcoords[1] = texcoords[2] = NULL;
	materials = NULL;
	groups = NULL;
	spatialSubd = NULL;
	if (paths.empty())
		paths.push_back(STR_DUP("."));
	
	center = Vector3D(0,0,0);
	
	double_sided = false;
	numfaces = numvertices = numnormals = numtangents =
		numtexcoords[0] = numtexcoords[1] = numtexcoords[2] =
		numgroups = nummaterials = 0;
#ifdef BUFFER_OBJECT
	memtangents = memvertices = memnormals =
		memtexcoords[0] = memtexcoords[1] = memtexcoords[2] =
		memmaterials = memgroups = 0;
	bobject_opaque = 0;
#endif
	
	dlist_opaque = dlist_transparent = 0;
	
	readFormat(filename);
	processData();
}

Mesh3D::Mesh3D(void)
{
	vertices = NULL;
	normals = NULL;
	tangents = NULL;
	texcoords[0] = texcoords[1] = texcoords[2] = NULL;
	materials = NULL;
	groups = NULL;
	spatialSubd = NULL;
	paths.push_back(STR_DUP("."));
	this->filename[0] = '\0';
	
	center = Vector3D(0,0,0);
	
	double_sided = false;
	numfaces = numvertices = numnormals = numtangents =
		numtexcoords[0] = numtexcoords[1] = numtexcoords[2] =
		numgroups = nummaterials = 0;
#ifdef BUFFER_OBJECT
	memtangents = memvertices = memnormals =
		memtexcoords[0] = memtexcoords[1] = memtexcoords[2] =
		memmaterials = memgroups = 0;
	bobject_opaque = 0;
#endif
	
	dlist_opaque = dlist_transparent = 0;
}

Mesh3D::~Mesh3D(void)
{
	if (vertices) free(vertices);
	if (normals) free(normals);
	if (numtexcoords[0] && texcoords[0]) free(texcoords[0]);
	if (numtexcoords[1] && texcoords[1]) free(texcoords[1]);
	if (numtexcoords[2] && texcoords[2]) free(texcoords[2]);
	if (materials) free(materials);
	for (unsigned long i=0;i<numgroups;i++)
	    if (groups[i].faces) free(groups[i].faces);
	if (groups) free(groups);
	
	if (spatialSubd)
		delete spatialSubd;
	for (unsigned int i=0;i<paths.size();i++)
		free(paths.at(i));
	paths.clear();
}

void Mesh3D::processData(void)
{
	prepareFaceNormals();
	prepareVertexTangents();
#ifdef BUFFER_OBJECT
	bufferObject();
#else
	compileLists();
#endif
	calcBBox();
	calcBSphere();
    calcTightBSphere();
}

bool Mesh3D::intersectAll(Ray3D &r)
{
	Vector3D isect;
	Vector3D normal;
	bool found = false;
#ifndef BUFFER_OBJECT
	DBL t = 1000000.0f;
	for (unsigned long j=0;j<numgroups;j++)
	for (unsigned long i=0;i<groups[i].numfaces;i++)
	{
		if (FindIntersection(groups[j].faces[i], r, vertices, normals))
		{
			found = true;
			if (r.t<t && r.t>0)
			{
				isect  = r.p_isect;
				normal = r.n_isect;
				t = r.t;
			}
		}
	}
	r.p_isect = isect;
	r.n_isect = normal;
#endif
	return found;
}

bool Mesh3D::intersect(Ray3D &r)
{
	return spatialSubd->intersect(r);
}

float Mesh3D::getMaxDimension()
{
	float w,h,d;

	w = bbox.getSize().x;
    h = bbox.getSize().y;
    d = bbox.getSize().z;

	if(w>h)
		return (w>d) ? w : d;
	else
		return (h>d) ? h : d;
}

/* Find material id NAME in MODEL */
unsigned int Mesh3D::findMaterial(char *name)
{
    GLuint i;

    for (i = 0; i < nummaterials; i++)
        if (STR_EQUAL(materials[i].name, name))
            return i;

    /* didn't find the name, so return a huge value */
    return INT_MAX;
}

void Mesh3D::calcFaceNormal(Vector3D *nrm,Vector3D &v1,Vector3D &v2,Vector3D &v3,bool normalized)
{
	Vector3D ret;
	Vector3D v4,v5;

	v4.x = v2.x-v1.x;
	v4.y = v2.y-v1.y;
	v4.z = v2.z-v1.z;
	v5.x = v3.x-v1.x;
	v5.y = v3.y-v1.y;
	v5.z = v3.z-v1.z;
	ret = v4.cross(v5);

	if(! normalized)
		ret.normalize();
	*nrm = ret;
}

unsigned int Mesh3D::loadTexture(const char *fname)
{
	char * texpath = getFullPath((char*)fname);
	if (!texpath)
	{
		EAZD_TRACE ("Mesh3D::loadTexture() : Warning - Could not find texture " << fname);
		return 0;
	}

	return TextureManager3D::loadTexture(texpath);
}

void Mesh3D::dumpAll(void)
{
    unsigned long i;

    fprintf (stdout, "Model \"%s\" info:\n{\n", getName ());
    fprintf (stdout, "\tNum of Vertices     : %lu\n", numvertices);
    for (i = 0; i < numvertices; i++)
        fprintf (stdout, "\t\t% f % f % f\n", vertices[i][0], vertices[i][1], vertices[i][2]);

    fprintf (stdout, "\tNum of Normals      : %lu\n", numnormals);
    for (i = 0; i < numnormals; i++)
        fprintf (stdout, "\t\t% f % f % f\n", normals[i][0], normals[i][1], normals[i][2]);

    fprintf (stdout, "\tNum of Tangents     : %lu\n", numtangents);
    for (i = 0; i < numtangents; i++)
        fprintf (stdout, "\t\t% f % f % f\n", tangents[i][0], tangents[i][1], tangents[i][2]);

    fprintf (stdout, "\tNum of TexCoords[0] : %lu\n", numtexcoords[0]);
    for (i = 0; i < numtexcoords[0]; i++)
        fprintf (stdout, "\t\t% f % f\n", texcoords[0][i][0], texcoords[0][i][1]);

    fprintf (stdout, "\tNum of TexCoords[1] : %lu\n", numtexcoords[1]);
    for (i = 0; i < numtexcoords[1]; i++)
        fprintf (stdout, "\t\t% f % f\n", texcoords[1][i][0], texcoords[1][i][1]);

    fprintf (stdout, "\tNum of TexCoords[2] : %lu\n", numtexcoords[2]);
    for (i = 0; i < numtexcoords[2]; i++)
        fprintf (stdout, "\t\t% f % f\n", texcoords[2][i][0], texcoords[2][i][1]);

    fprintf (stdout, "\n");
    fprintf (stdout, "\tNum of Materials    : %lu\n", nummaterials);
    for (i = 0; i < nummaterials; i++)
        materials[i].dump ();

    fprintf (stdout, "\n");
    fprintf (stdout, "\tNum of Groups : %lu\n", numgroups);
    for (i = 0; i < numgroups; i++)
        groups[i].dumpAll();
    fprintf (stdout, "}\n");
    fflush  (stdout);
}

void Mesh3D::dump(void)
{
    unsigned long i;

    fprintf (stdout, "Model \"%s\" info:\n{\n", getName ());
    fprintf (stdout, "\tNum of Vertices     : %lu\n", numvertices);
    fprintf (stdout, "\tNum of Normals      : %lu\n", numnormals);
    fprintf (stdout, "\tNum of Tangents     : %lu\n", numtangents);
    fprintf (stdout, "\tNum of TexCoords[0] : %lu\n", numtexcoords[0]);
    fprintf (stdout, "\tNum of TexCoords[1] : %lu\n", numtexcoords[1]);
    fprintf (stdout, "\tNum of TexCoords[2] : %lu\n", numtexcoords[2]);

    fprintf (stdout, "\n");
    fprintf (stdout, "\tNum of Materials    : %lu\n", nummaterials);
    for (i = 0; i < nummaterials; i++)
        materials[i].dump ();

    fprintf (stdout, "\n");
    fprintf (stdout, "\tNum of Groups : %lu\n", numgroups);
    for (i = 0; i < numgroups; i++)
        groups[i].dump ();
    fprintf (stdout, "}\n");
    fflush  (stdout);
}

void Mesh3D::calcBBox()
{
	long unsigned int i;

	Vector3D vmin, vmax, sz;
	vmin = vertices[1];
	vmax = vertices[1];
	for(i=1;i<numvertices;i++)
	{
		if(vertices[i].x<vmin.x)
		   vmin.x=vertices[i].x;
		if(vertices[i].y<vmin.y)
		   vmin.y=vertices[i].y;
		if(vertices[i].z<vmin.z)
		   vmin.z=vertices[i].z;
		if(vertices[i].x>vmax.x)
		   vmax.x=vertices[i].x;
		if(vertices[i].y>vmax.y)
		   vmax.y=vertices[i].y;
		if(vertices[i].z>vmax.z)
		   vmax.z=vertices[i].z;
	}
	center.x = (float)((vmax.x+vmin.x)*0.5);
	center.y = (float)((vmax.y+vmin.y)*0.5);
	center.z = (float)((vmax.z+vmin.z)*0.5);
	sz = vmax-vmin;
	bbox.setSymmetrical(center,sz);
}

void Mesh3D::calcBSphere()
{
    bsphere.expandBy (bbox);
}

void Mesh3D::calcTightBSphere()
{
    btsphere.makeEmpty ();
    btsphere.set (bbox.getCenter (), -FLT_MAX);

    DBL radius;

    // check the istance of each vertex from the center
    for (unsigned long i = 0; i < numvertices; i++)
    {
        // check the squared distance
        radius = (btsphere.getCenter () - vertices[i]).length2 ();

        if (btsphere.getRadius () < radius)
            btsphere.setRadius (radius);
    }

    // get the correct radius
    btsphere.setRadius ((DBL) sqrt (btsphere.getRadius ()));
} // end of getBoundingSphere

// from obj_normals_object(void)
void Mesh3D::prepareFaceNormals(void)
{
    unsigned long si;
    unsigned long pi;
	numfaces = 0;
    /* Compute normals for all faces. */
    for (si = 0; si < numgroups; ++si)
        for (pi = 0; pi < groups[si].numfaces; ++pi)
        {
			numfaces++;
#ifdef BUFFER_OBJECT
            Face3D *p = &(groups[si].faces[pi]);
#else // COMPILE_LIST
            Triangle3D *p = &(groups[si].faces[pi]);
#endif

            Vector3D v0 = vertices[p->vertIdx[0]];
            Vector3D v1 = vertices[p->vertIdx[1]];
            Vector3D v2 = vertices[p->vertIdx[2]];

            /* Compute the normal formed by these 3 vertices. */
#ifdef BUFFER_OBJECT
            calcFaceNormal(&(groups[si].fc_normals[pi]), v0, v1, v2, 0);
#else // COMPILE_LIST
            calcFaceNormal(&(p->fc_normal), v0, v1, v2, 0);
#endif
        }
}

#ifdef BUFFER_OBJECT
// from obj_tangents_object(void)
// This routine assumes the mesh has been exploded so that
// numvertices == numnormals == numtangents == numtextures
// Also it only computes tangents based on the first set of
// texture coordinates.
void Mesh3D::prepareVertexTangents(void)
{
    unsigned long si;
    unsigned long pi;
    unsigned long vi;

	// the BUFFER_OBJECT method allocates the data itself
#ifndef BUFFER_OBJECT
	numtangents = numnormals;
	tangents = (Vector3D *) malloc (numtangents * sizeof (Vector3D));
#endif

    EAZD_ASSERTALWAYS (numnormals == numtangents);

    /* Normalize all normals and zero out all tangent vectors. */
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
#else
// from obj_tangents_object(void)
// it computes tangents based on the first set of
// texture coordinates.
void Mesh3D::prepareVertexTangents(void)
{
    unsigned long si;
    unsigned long pi;
    unsigned long vi;

	numtangents = 0;
	for (si = 0; si < numgroups; ++si)
        numtangents+=3*groups[si].numfaces;
	tangents = (Vector3D *) malloc (numtangents * sizeof (Vector3D));

    /* Normalize all normals and zero out all tangent vectors. */
    for (vi = 0; vi < numnormals; ++vi)
        normals[vi].normalize();
	for (vi = 0; vi < numtangents; ++vi)
        tangents[vi] = Vector3D(0.0f, 0.0f, 0.0f);
    
	long tan_counter=0;
    /* Compute tangent vectors for all vertices. */
    for (si = 0; si < numgroups; ++si)
        for (pi = 0; pi < groups[si].numfaces; ++pi)
        {
            Triangle3D *p = &(groups[si].faces[pi]);

            Vector3D *v0 = &(vertices[p->vertIdx[0]]);
            Vector3D *v1 = &(vertices[p->vertIdx[1]]);
            Vector3D *v2 = &(vertices[p->vertIdx[2]]);

			Vector2D *tx0 = &(texcoords[0][p->texcIdx[0]]);
			Vector2D *tx1 = &(texcoords[0][p->texcIdx[1]]);
			Vector2D *tx2 = &(texcoords[0][p->texcIdx[2]]);

			Vector3D n = p->fc_normal;
			
			Vector3D *tn0 = &(tangents[tan_counter++]);
			Vector3D *tn1 = &(tangents[tan_counter++]);
			Vector3D *tn2 = &(tangents[tan_counter++]);
			
			// gepap: implementation of the method found in Math. for CG and Game progr., E. Lengyel.
			
			float u21,v21,u31,v31, det; 
			u21 = tx1->x - tx0->x;
			v21 = tx1->y - tx0->y;
			u31 = tx2->x - tx0->x;
			v31 = tx2->y - tx0->y;
			det = u21*v31-u31*v21;
			Vector3D q2 = (*v1)-(*v0);
			Vector3D q3 = (*v2)-(*v0);
			Vector3D t;
			if (det!=0.0f)
			{
				t.x = (v31*q2.x-v21*q3.x)/det;
				t.y = (v31*q2.y-v21*q3.y)/det;
				t.z = (v31*q2.z-v21*q3.z)/det;
				t.normalize();
			}
			else
			{
				t = Vector3D(0.0f,0.0f,1.0f);
				if (n.dot(t)==1.0f)
					t = Vector3D(0.0f,1.0f,0.0f);
				t = t.cross(n);
			}
			*tn0 = t;
			*tn1 = t;
			*tn2 = t;
		}

	tan_counter=0;
    for (si = 0; si < numgroups; ++si)
        for (pi = 0; pi < groups[si].numfaces; ++pi)
		{	
			Triangle3D *p = &(groups[si].faces[pi]);

			for (vi = 0; vi < 3; ++vi)
			{
				Vector3D *n = &(normals[p->normIdx[vi]]);
				Vector3D *u = &(tangents[tan_counter++]);
				Vector3D v = Vector3D::cross(*n, *u);
				*u = Vector3D::cross(v, *n);
				u->normalize();
			}
		}
}
#endif


char   *
getFileExtension (const char *filename)
{
    char *ext;

    /* look for final "." in filename */
    if ((ext = (char *) strrchr (filename, '.')) != NULL)
        /* advance "ext" past the period character */
        ++ext;

    return ext;
}

char   *
getLowerCaseFileExtension (const char *filename)
{
    unsigned int i;
    char *itr, *ext = getFileExtension (filename);

    if (ext != NULL)
        for (i = 0, itr = ext; i < strlen (ext); i++, itr++)
            *itr = (char) tolower (*itr);

    return ext;
}

void Mesh3D::readFormat (const char * filename)
{
    char * ext = getLowerCaseFileExtension (filename);

    if (STR_EQUAL (ext, "obj"))
    {
        objMesh3D * o_mesh = reinterpret_cast <objMesh3D *> (this);
        if (o_mesh)
            o_mesh->readFormat (filename);
        else
            EAZD_TRACE ("Mesh3D::readFormat() : ERROR - Could not convert \"this\" to objMesh3D for file " << filename);
    }
    else if (STR_EQUAL (ext, "raw"))
    {
        rawMesh3D * r_mesh = reinterpret_cast <rawMesh3D *> (this);
        if (r_mesh)
            r_mesh->readFormat (filename);
        else
            EAZD_TRACE ("Mesh3D::readFormat() : ERROR - Could not convert \"this\" to rawMesh3D for file " << filename);
    }
}

void Mesh3D::writeFormat (const char * filename)
{
    char * ext = getLowerCaseFileExtension (filename);

    if (STR_EQUAL (ext, "obj"))
    {
        objMesh3D * o_mesh = reinterpret_cast <objMesh3D *> (this);
        if (o_mesh)
            o_mesh->writeFormat (filename);
        else
            EAZD_TRACE ("Mesh3D::writeFormat() : ERROR - Could not convert \"this\" to objMesh3D for file " << filename);
    }
    else if (STR_EQUAL (ext, "raw"))
    {
        rawMesh3D * r_mesh = reinterpret_cast <rawMesh3D *> (this);
        if (r_mesh)
            r_mesh->writeFormat (filename);
        else
            EAZD_TRACE ("Mesh3D::writeFormat() : ERROR - Could not convert \"this\" to rawMesh3D for file " << filename);
    }
}

#ifndef BUFFER_OBJECT
void Mesh3D::compileLists()
{
	long unsigned int i,j;

	if (dlist_opaque)
	{
		glDeleteLists (dlist_opaque, 1);
		dlist_opaque = 0;
	}
	
	dlist_opaque=glGenLists(1);
	glNewList(dlist_opaque,GL_COMPILE);
	glEnable(GL_NORMALIZE);
	glDisable(GL_COLOR_MATERIAL);
	
	long curtangent=0;

	for(i=0;i<numgroups;i++)
	{
		if (groups[i].numfaces==0)
			continue;
		if (materials[ groups[i].mtrlIdx].alpha<1.0f)
		{
			curtangent+=groups[i].numfaces*3;
			continue;
		}

		// ALBEDO TEXTURES (2 layers, DIFFUSE0,1 - TEXTURE UNITS 0,1)
		if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_DIFFUSE0])
		{
			if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_DIFFUSE0] > 0)
			{
				glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_DIFFUSE0 );
				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_DIFFUSE0]);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16);
			}
			if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_DIFFUSE0])
			{
				if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_DIFFUSE0] > 0)
				{
					glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_DIFFUSE1 );
					glEnable(GL_TEXTURE_2D);
					glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_DIFFUSE1]);
					glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16);
				}
			}
		}
		else
		{
			glActiveTextureARB( GL_TEXTURE0_ARB );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
			glActiveTextureARB( GL_TEXTURE1_ARB );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
		}
		
		// BUMP TEXTURE (BUMP - TEXTURE UNIT 2)
		if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_BUMP])
		{
			if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_BUMP] > 0)
			{
				glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_BUMP );
				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_BUMP]);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16);
			}
		}
		else
		{
			glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_BUMP );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
		}

		// SPECULAR TEXTURE (SPECULAR COLOR+EXP - TEXTURE UNIT 3)
		if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_SPECULAR])
		{
			if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_SPECULAR] > 0)
			{
				glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_SPECULAR );
				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_SPECULAR]);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16);
			}
		}
		else
		{
			glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_SPECULAR );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
		}
        
		// EMISSION TEXTURE (EMISSIVENESS - TEXTURE UNIT 4)
		if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_EMISSION])
		{
			if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_EMISSION] > 0)
			{
				glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_EMISSION );
				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_EMISSION]);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 4);
			}
		}
		else
		{
			glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_EMISSION );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
		}

        // apply the material properties.
        materials[groups[i].mtrlIdx].draw ();

		glBegin(GL_TRIANGLES);
		for(j=0;j< groups[i].numfaces;j++)
		{
            Triangle3D *t = &(groups[i].faces[j]);

			if( groups[i].has_normals)
			{
				glNormal3f( normals[t->normIdx[0]].x,
				            normals[t->normIdx[0]].y,
							normals[t->normIdx[0]].z);
				glVertexAttrib3f(1,tangents[curtangent].x,
								 tangents[curtangent].y,
								 tangents[curtangent].z);
			}
			else
				glNormal3f( t->fc_normal.x,
				            t->fc_normal.y,
							t->fc_normal.z);
			curtangent++;
			if( groups[i].has_texcoords[0])
				glMultiTexCoord2fARB( GL_TEXTURE0_ARB,
				                      texcoords[0][t->texcIdx[0]].x,
				                      texcoords[0][t->texcIdx[0]].y);
			glVertex3f( vertices[t->vertIdx[0]].x,
				        vertices[t->vertIdx[0]].y,
						vertices[t->vertIdx[0]].z);

			if( groups[i].has_normals)
			{	
				glNormal3f( normals[t->normIdx[1]].x,
				            normals[t->normIdx[1]].y,
							normals[t->normIdx[1]].z);
   				glVertexAttrib3f(1,tangents[curtangent].x,
								 tangents[curtangent].y,
								 tangents[curtangent].z);
			}
			else
				glNormal3f( t->fc_normal.x,
				            t->fc_normal.y,
							t->fc_normal.z);
			curtangent++;
			if( groups[i].has_texcoords[0])
				glMultiTexCoord2fARB( GL_TEXTURE0_ARB,
				                      texcoords[0][t->texcIdx[1]].x,
									  texcoords[0][t->texcIdx[1]].y);
			glVertex3f( vertices[t->vertIdx[1]].x,
				        vertices[t->vertIdx[1]].y,
						vertices[t->vertIdx[1]].z);

			if( groups[i].has_normals)
			{	
				glNormal3f( normals[t->normIdx[2]].x,
				            normals[t->normIdx[2]].y,
							normals[t->normIdx[2]].z);
   				glVertexAttrib3f(1,tangents[curtangent].x,
								 tangents[curtangent].y,
								 tangents[curtangent].z);
			}
			else
				glNormal3f( t->fc_normal.x,
				            t->fc_normal.y,
							t->fc_normal.z);
			curtangent++;
			if( groups[i].has_texcoords[0])
				glMultiTexCoord2fARB( GL_TEXTURE0_ARB,
				                      texcoords[0][t->texcIdx[2]].x,
									  texcoords[0][t->texcIdx[2]].y);
			glVertex3f( vertices[t->vertIdx[2]].x,
				        vertices[t->vertIdx[2]].y,
						vertices[t->vertIdx[2]].z);
		}
		glEnd();
	}
	glEndList();

	if (dlist_transparent)
	{
		glDeleteLists (dlist_transparent, 1);
		dlist_transparent = 0;
	}
	
	dlist_transparent=glGenLists(1);
	glNewList(dlist_transparent,GL_COMPILE);
	glEnable(GL_NORMALIZE);
	glDisable(GL_COLOR_MATERIAL);
		
	for(int polymode=GL_CW;polymode<=GL_CCW;polymode++)
	for(i=0;i<numgroups;i++)
	{
		glFrontFace(polymode);
		if (groups[i].numfaces==0)
			continue;
		if (materials[ groups[i].mtrlIdx].alpha>0.99999f)
			continue;

		// ALBEDO TEXTURES (2 layers, DIFFUSE0,1 - TEXTURE UNITS 0,1)
		if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_DIFFUSE0])
		{
			if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_DIFFUSE0] > 0)
			{
				glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_DIFFUSE0 );
				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_DIFFUSE0]);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16);
			}
			if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_DIFFUSE0])
			{
				if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_DIFFUSE0] > 0)
				{
					glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_DIFFUSE1 );
					glEnable(GL_TEXTURE_2D);
					glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_DIFFUSE1]);
					glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16);
				}
			}
		}
		else
		{
			glActiveTextureARB( GL_TEXTURE0_ARB );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
			glActiveTextureARB( GL_TEXTURE1_ARB );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
		}
		
		// BUMP TEXTURE (BUMP - TEXTURE UNIT 2)
		if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_BUMP])
		{
			if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_BUMP] > 0)
			{
				glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_BUMP );
				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_BUMP]);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8);
			}
		}
		else
		{
			glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_BUMP );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
		}

		// SPECULAR TEXTURE (SPECULAR COLOR+EXP - TEXTURE UNIT 3)
		if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_SPECULAR])
		{
			if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_SPECULAR] > 0)
			{
				glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_SPECULAR );
				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_SPECULAR]);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8);
			}
		}
		else
		{
			glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_SPECULAR );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
		}
        
		// EMISSION TEXTURE (EMISSIVENESS - TEXTURE UNIT 4)
		if( materials[ groups[i].mtrlIdx].has_texture[MATERIAL_MAP_EMISSION])
		{
			if( materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_EMISSION] > 0)
			{
				glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_EMISSION );
				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, materials[ groups[i].mtrlIdx].texturemap[MATERIAL_MAP_EMISSION]);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 4);
			}
		}
		else
		{
			glActiveTextureARB( GL_TEXTURE0_ARB+MATERIAL_MAP_EMISSION );
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D,  0);
		}

        // apply the material properties.
		materials[groups[i].mtrlIdx].draw ();

		glBegin(GL_TRIANGLES);
		for(j=0;j< groups[i].numfaces;j++)
		{
            Triangle3D *t = &(groups[i].faces[j]);

			if (polymode == GL_CCW)
			{
			if( groups[i].has_normals)
				glNormal3f( normals[t->normIdx[0]].x,
				            normals[t->normIdx[0]].y,
							normals[t->normIdx[0]].z);
			else
				glNormal3f( t->fc_normal.x,
				            t->fc_normal.y,
					        t->fc_normal.z);
			}
			else
			{
			if( groups[i].has_normals)
				glNormal3f( -normals[t->normIdx[0]].x,
				            -normals[t->normIdx[0]].y,
							-normals[t->normIdx[0]].z);
			else
				glNormal3f( -t->fc_normal.x,
				            -t->fc_normal.y,
					        -t->fc_normal.z);
			}
			if( groups[i].has_texcoords[0])
				glMultiTexCoord2fARB( GL_TEXTURE0_ARB,
				                      texcoords[0][t->texcIdx[0]].x,
				                      texcoords[0][t->texcIdx[0]].y);
			glVertex3f( vertices[t->vertIdx[0]].x,
				        vertices[t->vertIdx[0]].y,
						vertices[t->vertIdx[0]].z);

			if (polymode == GL_CCW)
			{
			if( groups[i].has_normals)
				glNormal3f( normals[t->normIdx[1]].x,
				            normals[t->normIdx[1]].y,
							normals[t->normIdx[1]].z);
			else
				glNormal3f( t->fc_normal.x,
				            t->fc_normal.y,
					        t->fc_normal.z);
			}
			else
			{
			if( groups[i].has_normals)
				glNormal3f( -normals[t->normIdx[1]].x,
				            -normals[t->normIdx[1]].y,
							-normals[t->normIdx[1]].z);
			else
				glNormal3f( -t->fc_normal.x,
				            -t->fc_normal.y,
					        -t->fc_normal.z);
			}
			if( groups[i].has_texcoords[0])
				glMultiTexCoord2fARB( GL_TEXTURE0_ARB,
				                      texcoords[0][t->texcIdx[1]].x,
									  texcoords[0][t->texcIdx[1]].y);
			glVertex3f( vertices[t->vertIdx[1]].x,
				        vertices[t->vertIdx[1]].y,
						vertices[t->vertIdx[1]].z);

			if (polymode == GL_CCW)
			{
			if( groups[i].has_normals)
				glNormal3f( normals[t->normIdx[2]].x,
				            normals[t->normIdx[2]].y,
							normals[t->normIdx[2]].z);
			else
				glNormal3f( t->fc_normal.x,
				            t->fc_normal.y,
				            t->fc_normal.z);
			}
			else
			{
			if( groups[i].has_normals)
				glNormal3f( -normals[t->normIdx[2]].x,
				            -normals[t->normIdx[2]].y,
							-normals[t->normIdx[2]].z);
			else
				glNormal3f( -t->fc_normal.x,
				            -t->fc_normal.y,
				            -t->fc_normal.z);
			}
			if( groups[i].has_texcoords[0])
				glMultiTexCoord2fARB( GL_TEXTURE0_ARB,
				                      texcoords[0][t->texcIdx[2]].x,
									  texcoords[0][t->texcIdx[2]].y);
			glVertex3f( vertices[t->vertIdx[2]].x,
				        vertices[t->vertIdx[2]].y,
						vertices[t->vertIdx[2]].z);
		}
		glEnd();
	}
	glEndList();
}
#endif

#ifdef BUFFER_OBJECT
void Mesh3D::drawProp(int mi, int ri)
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

void Mesh3D::drawVert(void)
{
    GLsizei s = 0;

    /* Enable all necessary vertex attribute pointers. */
    glEnableVertexAttribArray(6);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    /* Bind attributes to a vertex buffer object. */
    glBindBuffer(GL_ARRAY_BUFFER, bobject_opaque);

    glVertexAttribPointer(6, 3, GL_FLOAT, 0, s, BUFFER_OFFSET(ptrtangents));
    glNormalPointer      (      GL_FLOAT,    s, BUFFER_OFFSET(ptrnormals));
    glTexCoordPointer    (2,    GL_FLOAT,    s, BUFFER_OFFSET(ptrtexcoords));
    glVertexPointer      (3,    GL_FLOAT,    s, BUFFER_OFFSET(ptrvertices));
}

void Mesh3D::drawMtrl(int mi)
{
    GLint   GL_max_texture_image_units;
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &GL_max_texture_image_units);

    /* Bind as many of the texture maps as the GL implementation will allow. */
    if (GL_max_texture_image_units > 3 && materials[mi].texturemap[OBJ_NS])
    {
        glActiveTexture(GL_TEXTURE3);
        drawProp(mi, OBJ_NS);
    }
    if (GL_max_texture_image_units > 2 && materials[mi].texturemap[OBJ_KS])
    {
        glActiveTexture(GL_TEXTURE2);
        drawProp(mi, OBJ_KS);
    }
    if (GL_max_texture_image_units > 1 && materials[mi].texturemap[OBJ_KA])
    {
        glActiveTexture(GL_TEXTURE1);
        drawProp(mi, OBJ_KA);
    }
    if (GL_max_texture_image_units > 0 && materials[mi].texturemap[OBJ_KD])
    {
        glActiveTexture(GL_TEXTURE0);
        drawProp(mi, OBJ_KD);
    }

    // apply the material properties.
	materials[mi].draw ();
}

void Mesh3D::drawSurf(unsigned long si)
{
    PrimitiveGroup3D *sp = &(groups[si]);

    if (0 < sp->numfaces)
    {
        /* Apply this surface's material. */
        if (0 <= sp->mtrlIdx && sp->mtrlIdx < nummaterials)
            drawMtrl(sp->mtrlIdx);

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

void Mesh3D::drawObject(void)
{
    unsigned long si;

    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_ENABLE_BIT);
    {
        glDisable(GL_COLOR_MATERIAL);

        /* Load the vertex buffer. */
        drawVert();

        /* Render each surface. */
        for (si = 0; si < numgroups; ++si)
            drawSurf(si);
    }
    glPopAttrib();
    glPopClientAttrib();
}

// from obj_init(void)
void Mesh3D::bufferObject(void)
{
	if (bobject_opaque)
	{
        glDeleteBuffers(1, &bobject_opaque);
        bobject_opaque = 0;
	}
	
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
	
	if (dlist_opaque)
	{
		glDeleteLists (dlist_opaque, 1);
		dlist_opaque = 0;
	}
	
	dlist_opaque=glGenLists(1);
	glNewList(dlist_opaque,GL_COMPILE);
	drawObject();
	glEndList();
	
	if (dlist_transparent)
	{
		glDeleteLists (dlist_transparent, 1);
		dlist_transparent = 0;
	}
	
	dlist_transparent=glGenLists(1);
	glNewList(dlist_transparent,GL_COMPILE);
	// drawObject();
	glEndList();
}
#endif

void Mesh3D::draw()
{
	if (double_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);

	glCallList(dlist_opaque);
	glCallList(dlist_transparent);
}

void Mesh3D::drawOpaque()
{
	if (double_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);

	glCallList(dlist_opaque);
}

void Mesh3D::drawTransparent()
{
	if (double_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
	glCallList(dlist_transparent);
}

void Mesh3D::addSearchPaths(vector<char *>addpaths)
{
	for (unsigned int i=0; i<addpaths.size();i++)
		paths.push_back(STR_DUP(addpaths.at(i)));
}

void Mesh3D::addSearchPath(char * addpath)
{
	paths.push_back(STR_DUP(addpath));
}

char *Mesh3D::getFullPath(const char * file)
{
	char * final = NULL;
	for (unsigned int i=0; i<paths.size();i++)
	{
		char * dir = paths.at(i);
		int len = strlen(dir)+strlen(file)+2; // 1 for delimiter & 1 for null
		final = (char *)malloc(len*sizeof(char));
		sprintf(final,"%s%c%s",dir,DIR_DELIMITER,file);
		ifstream test;
		test.open(final);
		if (!test.fail())
		{
			test.close();
			return final;
		}
		free (final);
	}
	return NULL;
}

#if 0
///////////////////////////////////////////// INTERSECTION METHODS
int MyMesh::FindIntersection(Ray &r)//Vector v1, Vector v2, Vector& new_normal,MyMesh *mesh, Vector& intersection_point, int* group, int* face, Vector& barycentric, double* t)
// This is a generic method to compute an intersection between a semi-infinite line beginning at v1 and passing
// through v2 and the mesh. If the intersection between the line and the mesh is located before v1, then the
// intersection point is discarded. See the overloaded member below for argument details.
{
	int i, j, hits;
	Vector int_point, new_norm, bar_coords;
	r.t = 10000000.0;
	Ray r2 ;
	CopyVector(r2.origin, r.origin);
	CopyVector(r2.end, r.end);
	CopyVector(r2.dir, r.dir);
	r2.group_index = r.group_index;
	r2.face_index=r.face_index;
	r2.mesh = r.mesh;

	int_point[X] = int_point[Y] = int_point[Z] = 0;
	bar_coords[X] = 0; bar_coords[Y] = 0; bar_coords[Z] = 0;

	hits = 0;

	for(i=0;i<numgroups;i++){
		for(j=0;j< groups[i].numfaces;j++){
			if(FindIntersection( faces[ groups[i].faces[j]], r2)){//r.origin, r.end, new_norm, intersection_point, t_check)){
				if(r2.group_index == i && r2.face_index==j && this==r2.mesh)
					continue;

				hits++;

				if(r2.t<r.t){
					CopyVector(r.isect_point, r2.isect_point);
					CopyVector(r.barycentric, r2.barycentric);
					CopyVector(r.normal_isect, r2.normal_isect);
					r.t = r2.t;//*t = *t_check;

					r.group_index = i;
					r.face_index = j;
					r.mesh = this;
				}//if t
			}
		}//for j
	}//for i

	return hits;
}
#endif

