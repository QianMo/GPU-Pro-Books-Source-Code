
#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	#pragma warning (disable : 4996)
#endif

#include <string.h>

#include "obj.h"

/* Find group id NAME in MODEL */
int objMesh3D::findGroup( char *name)
{
    GLuint i;

    for (i = 0; i < numgroups; i++)
        if (STR_EQUAL(name, groups[i].name))
		    return i;

    return -1;
}

/* Add group NAME to the MODEL structure */
unsigned int objMesh3D::addGroup( char *name)
{
   int find = findGroup( name);
    if (find==-1)
    {
        strcpy(groups[numgroups].name,name);
        numgroups++;
	    return numgroups-1;
    }
    else
	    return find;
}

void objMesh3D::readMTL(char *filename)
{
	FILE *file;
	char buf[256];
	char buf1[256];
	char *fname;
	unsigned int mat_num = 1;
	float   r,g,b;

	fname = getFullPath(filename);
	if(!fname)
	{
		EAZD_TRACE ("objMesh3D::readMTL() : ERROR - File \"" << filename << "\" is corrupt or does not exist.");
		return;
	}
    file = fopen(fname,"rt");
    rewind(file);

	//1st pass - count materials
	while(!feof(file))
	{
		buf[0] = NULL;
		fscanf(file,"%s", buf);

		if (STR_EQUAL(buf,"newmtl"))
		{
			mat_num ++;
		}
		else
		{
			fgets(buf, sizeof(buf), file);
		}
	}
	materials=(Material3D*)malloc(sizeof(Material3D)*mat_num);
	for (unsigned int i = 0; i< mat_num; i++)
		materials[i] = Material3D();
    this->nummaterials = mat_num;
	rewind(file);
	mat_num = 0;

	while(!feof(file))
	{
		buf[0] = NULL;
		fscanf(file,"%s", buf);

		if (STR_EQUAL(buf,"newmtl"))
		{
			fscanf(file,"%s",buf1);
			mat_num ++;
			strcpy(materials[mat_num].name, buf1);
			materials[mat_num].has_texture[MATERIAL_MAP_DIFFUSE0] = false;
			materials[mat_num].has_texture[MATERIAL_MAP_DIFFUSE1] = false;
			materials[mat_num].has_texture[MATERIAL_MAP_BUMP] = false;
			materials[mat_num].has_texture[MATERIAL_MAP_SPECULAR] = false;
			materials[mat_num].has_texture[MATERIAL_MAP_EMISSION] = false;
		}
		else if (STR_EQUAL(buf,"Ka"))
		{
			fscanf(file,"%f %f %f", &r, &g, &b );
			materials[mat_num].ambient = Vector3D(r,g,b);
		}
		else if (STR_EQUAL(buf,"Ke"))
		{
			fscanf(file,"%f %f %f", &r, &g, &b );
			materials[mat_num].emission = Vector3D(r,g,b);
		}
		else if (STR_EQUAL(buf,"Kd"))
		{
			fscanf(file,"%f %f %f",&r,&g,&b);
			materials[mat_num].diffuse = Vector3D(r,g,b);
		}
		else if (STR_EQUAL(buf,"Ks"))
		{
			fscanf(file,"%f %f %f",&r,&g,&b);
			materials[mat_num].specular = Vector3D(r,g,b);
		}
		else if (STR_EQUAL(buf,"Ns"))
		{
			fscanf(file,"%f",&r);
			materials[mat_num].shininess =  (int)floor(r);
		}
		else if (STR_EQUAL(buf,"d"))
		{
			fscanf(file,"%f",&r);
			materials[mat_num].alpha = r;
		}
		else if (STR_EQUAL(buf,"Tt"))
		{
			fscanf(file,"%f",&r);
			materials[mat_num].translucency = r;
		}
		else if (STR_EQUAL(buf, "map_Kd"))
		{
			char texname[256];
			int ret = fscanf(file,"%s",texname);
			if (ret!=EOF || ret!=0)
			{
				materials[mat_num].has_texture[MATERIAL_MAP_DIFFUSE0] = true;
				materials[mat_num].texturestr[MATERIAL_MAP_DIFFUSE0] = STR_DUP(texname);
				materials[mat_num].texturemap[MATERIAL_MAP_DIFFUSE0] = loadTexture(texname);
			}
		}
		else if (STR_EQUAL(buf, "map_Kd2"))
		{
			char texname[256];
			int ret = fscanf(file,"%s",texname);
			if (ret!=EOF || ret!=0)
			{
				materials[mat_num].has_texture[MATERIAL_MAP_DIFFUSE1] = true;
				materials[mat_num].texturestr[MATERIAL_MAP_DIFFUSE1] = STR_DUP(texname);
				materials[mat_num].texturemap[MATERIAL_MAP_DIFFUSE1] = loadTexture(texname);
			}
		}
/*
		else if (STR_EQUAL(buf, "map_bump"))
		{
			char texname[256];
			int ret = fscanf(file,"%s",texname);
			if (ret!=EOF || ret!=0)
			{
				materials[mat_num].has_texture[MATERIAL_MAP_BUMP] = true;
				materials[mat_num].texturestr[MATERIAL_MAP_BUMP] = STR_DUP(texname);
				materials[mat_num].texturemap[MATERIAL_MAP_BUMP] = loadTexture(texname);
			}
		}
*/
		else if (STR_EQUAL(buf, "map_Ks"))
		{
			char texname[256];
			int ret = fscanf(file,"%s",texname);
			if (ret!=EOF || ret!=0)
			{
				materials[mat_num].has_texture[MATERIAL_MAP_SPECULAR] = true;
				materials[mat_num].texturestr[MATERIAL_MAP_SPECULAR] = STR_DUP(texname);
				materials[mat_num].texturemap[MATERIAL_MAP_SPECULAR] = loadTexture(texname);
			}
		}
		else if (STR_EQUAL(buf, "map_Ke"))
		{
			char texname[256];
			int ret = fscanf(file,"%s",texname);
			if (ret!=EOF || ret!=0)
			{
				materials[mat_num].has_texture[MATERIAL_MAP_EMISSION] = true;
				materials[mat_num].texturestr[MATERIAL_MAP_EMISSION] = STR_DUP(texname);
				materials[mat_num].texturemap[MATERIAL_MAP_EMISSION] = loadTexture(texname);
			}
		}
		else if (STR_EQUAL(buf,"#"))
			fgets(buf,100,file);
	}

	if (file) fclose(file);
    free(fname);
}

void objMesh3D::readFormat(const char * filename)
{
	FILE *file;
	char buf[256];
	char buf1[256];
	long unsigned int    numvertices;
	long unsigned int    numnormals;
	long unsigned int    numtexs;
	long unsigned int    numfaces;
	long unsigned int    numgroups;
	unsigned int  material;
	long unsigned int v,n,t,i;
	long unsigned int grp;

	if(!filename)
	{
		EAZD_TRACE ("objMesh3D::readFormat() : ERROR - File \"" << filename << "\" is corrupt or does not exist.");
		return;
	}
    file = fopen(filename,"rt");

	strcpy(matlib,"");

	// Discover groups
	rewind(file);
    numgroups = 1;
	fgets(buf, sizeof(buf), file);
	while(fscanf(file, "%s", buf) != EOF)
	{
		if(buf[0]=='g')
			numgroups++;
		else
			fgets(buf, sizeof(buf), file);
	}
	if(numgroups==0)
		numgroups=1;
	this->numgroups = 0;
	groups = (PrimitiveGroup3D*)malloc(sizeof(PrimitiveGroup3D)*numgroups);
	for (grp = 0; grp<numgroups; grp++)
		groups[grp] = PrimitiveGroup3D();
	this->numfaces =0;

	// 1st Pass - counting
	rewind(file);
	numtexs = numvertices = numnormals = numfaces = 0;
	grp = addGroup((char *) "default");
	groups[0].mtrlIdx=0;
	while(fscanf(file, "%s", buf) != EOF)
	{
		if(STR_EQUAL(buf, "mtllib"))
		{
			fscanf(file, "%s", buf1);
			strcpy(matlib, buf1);
			readMTL( buf1);
		}
	    switch(buf[0])
		{
		case '#':
            fgets(buf, sizeof(buf), file);
			break;
		case 'v': // v[?]
			switch(buf[1])
			{
			case '\0': // v
				fgets(buf, sizeof(buf), file);
				numvertices++;
				break;
			case 'n': // vn
				fgets(buf, sizeof(buf), file);
				numnormals++;
				break;
			case 't': // vt
				fgets(buf, sizeof(buf), file);
				numtexs ++;
				break;
			default:
				EAZD_TRACE ("objMesh3D::readFormat() : Warning - Unknown token \"" << buf << "\".");
			}
			break;
		case 'm':
			fgets(buf, sizeof(buf), file);
			break;
		case 'u':
			fgets(buf, sizeof(buf), file);
			break;
		case 'g':
			fgets(buf, sizeof(buf), file);
			sscanf(buf, "%s", buf);
			grp = addGroup( buf);
			break;
		case 'f':
			v = n = t = 0;
			fscanf(file, "%s", buf);
			if (strstr(buf, "//"))
			{
				sscanf(buf, "%lu//%lu", &v, &n);
				fscanf(file, "%lu//%lu", &v, &n);
				fscanf(file, "%lu//%lu", &v, &n);
				groups[grp].numfaces++;
				while(fscanf(file, "%lu//%lu", &v, &n) > 0)
					groups[grp].numfaces++;
			}
			else if (sscanf(buf, "%lu/%lu/%lu", &v, &t, &n) == 3)
			{
				fscanf(file, "%lu/%lu/%lu", &v, &t, &n);
				fscanf(file, "%lu/%lu/%lu", &v, &t, &n);
				groups[grp].numfaces++;
				while(fscanf(file, "%lu/%lu/%lu", &v, &t, &n) > 0)
					groups[grp].numfaces++;
			}
			else if (sscanf(buf, "%lu/%lu", &v, &t) == 2)
			{
				fscanf(file, "%lu/%lu", &v, &t);
				fscanf(file, "%lu/%lu", &v, &t);
				groups[grp].numfaces++;
				while(fscanf(file, "%lu/%lu", &v, &t) > 0)
					groups[grp].numfaces++;
			}
			else
			{
				fscanf(file, "%lu", &v);
				fscanf(file, "%lu", &v);
				groups[grp].numfaces++;
				while(fscanf(file, "%lu", &v) > 0)
					groups[grp].numfaces++;
			}
			break;
		default:
			fgets(buf, sizeof(buf), file);
	    }
	}

	numtexs++;
	this->numvertices  = numvertices;
	this->numnormals   = numnormals;
	this->numtexcoords[0] = numtexs;

	

	for(i=0;i<this->numgroups;i++)
    {
#ifdef BUFFER_OBJECT
		groups[i].faces=(Face3D*)malloc(sizeof(Face3D)*groups[i].numfaces);
#else // COMPILE_LIST
		groups[i].faces=(Triangle3D*)malloc(sizeof(Triangle3D)*groups[i].numfaces);
#endif
		groups[i].numfaces=0;
    }

	// Allocating memory
	vertices=(Vector3D*)malloc(sizeof(Vector3D)*(this->numvertices));
	if(this->numnormals!=0)
		normals=(Vector3D*)malloc(sizeof(Vector3D)*(this->numnormals));
	if(this->numtexcoords[0]!=0)
		texcoords[0]=(Vector2D*)malloc(sizeof(Vector2D)*(this->numtexcoords[0]));

	// GEPAP: Add a dummy tex coord set to avoid problems with untextured faces.
	texcoords[0][0] = Vector2D(0,0);

	// Second Pass - assignment
	rewind(file);
	numvertices = numnormals = numtexs = 0;
	material = 0;
	grp = 0;
	// GEPAP:
	numtexs++;

	while(!feof(file))
	{
		fscanf(file, "%s", buf);

		if(STR_EQUAL(buf, "usemtl"))
		{
			fscanf(file, "%s", buf1);
			material = findMaterial( buf1);
			groups[grp].mtrlIdx = material;
			strcpy(matlib, buf1);
	    }

		switch(buf[0])
		{
		case '#':
			fgets(buf, sizeof(buf), file);
			break;
		case 'v': // v[?]
			switch(buf[1])
			{
			case '\0': // v
				fscanf(file, "%f %f %f",
				&(vertices[numvertices].x),
				&(vertices[numvertices].y),
				&(vertices[numvertices].z));
				numvertices++;
				break;
			case 'n': // vn
				fscanf(file, "%f %f %f",
				&(normals[numnormals].x),
				&(normals[numnormals].y),
				&(normals[numnormals].z));
				numnormals++;
				break;
			case 't': // vt
				fscanf(file, "%f %f",
				&(texcoords[0][numtexs].x),
				&(texcoords[0][numtexs].y));
				numtexs++;
				break;
			}
			break;
		case 'u':
			fgets(buf, sizeof(buf), file);
			break;
		case 'g':
			fgets(buf, sizeof(buf), file);
			sscanf(buf, "%s", buf);
			grp = findGroup(buf);
			groups[grp].mtrlIdx = material;

			numfaces = 0;
			groups[grp].numfaces = 0;
			break;
		case 'f':
			v = n = t = 0;
			fscanf(file, "%s", buf);
			if (strstr(buf, "//"))
			{
				t=0;
				groups[grp].faces[numfaces].texcIdx[0] = t;
				groups[grp].faces[numfaces].texcIdx[1] = t;
				groups[grp].faces[numfaces].texcIdx[2] = t;
				sscanf(buf, "%lu//%lu", &v, &n);
				v--, n--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[0] = v;
#ifndef BUFFER_OBJECT
				groups[grp].faces[numfaces].normIdx[0] = n;
#endif
				fscanf(file, "%lu//%lu", &v, &n);
				v--, n--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[1] = v;
#ifndef BUFFER_OBJECT
				groups[grp].faces[numfaces].normIdx[1] = n;
#endif
				fscanf(file, "%lu//%lu", &v, &n);
				v--, n--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[2] = v;
#ifndef BUFFER_OBJECT
				groups[grp].faces[numfaces].normIdx[2] = n;
#endif

				numfaces++;
				groups[grp].numfaces++;

				groups[grp].has_normals = true;
				groups[grp].has_texcoords[0] = false;
				while(fscanf(file, "%lu//%lu", &v, &n) > 0)
				{
					t=0;
					groups[grp].faces[numfaces-1].texcIdx[0] = t;
					groups[grp].faces[numfaces-1].texcIdx[1] = t;
					groups[grp].faces[numfaces-1].texcIdx[2] = t;
				
					v--, n--;   // because the format is 1 based
					groups[grp].faces[numfaces].vertIdx[0] = groups[grp].faces[numfaces-1].vertIdx[0];
#ifndef BUFFER_OBJECT
					groups[grp].faces[numfaces].normIdx[0] = groups[grp].faces[numfaces-1].normIdx[0];
#endif
					groups[grp].faces[numfaces].vertIdx[1] = groups[grp].faces[numfaces-1].vertIdx[2];
#ifndef BUFFER_OBJECT
					groups[grp].faces[numfaces].normIdx[1] = groups[grp].faces[numfaces-1].normIdx[2];
#endif
					groups[grp].faces[numfaces].vertIdx[2] = v;
#ifndef BUFFER_OBJECT
					groups[grp].faces[numfaces].normIdx[2] = n;
#endif

					numfaces++;
					groups[grp].numfaces++;
				}
			}
			else if (sscanf(buf, "%lu/%lu/%lu", &v, &t, &n) == 3)
			{
				// GEPAP: Add a dummy tex coord set to avoid problems with untextured faces.
				// v--, t--, n--;   // because the format is 1 based
				v--, n--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[0] = v;
#ifndef BUFFER_OBJECT
				groups[grp].faces[numfaces].normIdx[0] = n;
				groups[grp].faces[numfaces].texcIdx[0] = t;
#endif
				fscanf(file, "%lu/%lu/%lu", &v, &t, &n);
				// v--, t--, n--;   // because the format is 1 based
				v--, n--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[1] = v;
#ifndef BUFFER_OBJECT
				groups[grp].faces[numfaces].normIdx[1] = n;
				groups[grp].faces[numfaces].texcIdx[1] = t;
#endif
				fscanf(file, "%lu/%lu/%lu", &v, &t, &n);
				// v--, t--, n--;   // because the format is 1 based
				v--, n--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[2] = v;
#ifndef BUFFER_OBJECT
				groups[grp].faces[numfaces].normIdx[2] = n;
				groups[grp].faces[numfaces].texcIdx[2] = t;
#endif

				numfaces++;
				groups[grp].numfaces++;

				groups[grp].has_normals = true;
				groups[grp].has_texcoords[0] = true;

				while(fscanf(file, "%lu/%lu/%lu", &v, &t, &n) > 0)
				{
					//v--, t--, n--;   // because the format is 1 based
					v--, n--;   // because the format is 1 based
					groups[grp].faces[numfaces].vertIdx[0] = groups[grp].faces[numfaces-1].vertIdx[0];
#ifndef BUFFER_OBJECT
					groups[grp].faces[numfaces].normIdx[0] = groups[grp].faces[numfaces-1].normIdx[0];
					groups[grp].faces[numfaces].texcIdx[0] = groups[grp].faces[numfaces-1].texcIdx[0];
#endif
					groups[grp].faces[numfaces].vertIdx[1] = groups[grp].faces[numfaces-1].vertIdx[2];
#ifndef BUFFER_OBJECT
					groups[grp].faces[numfaces].normIdx[1] = groups[grp].faces[numfaces-1].normIdx[2];
					groups[grp].faces[numfaces].texcIdx[1] = groups[grp].faces[numfaces-1].texcIdx[2];
#endif
					groups[grp].faces[numfaces].vertIdx[2] = v;
#ifndef BUFFER_OBJECT
					groups[grp].faces[numfaces].normIdx[2] = n;
					groups[grp].faces[numfaces].texcIdx[2] = t;
#endif

					numfaces++;
					groups[grp].numfaces++;
				}
			}
			else if (sscanf(buf, "%lu/%lu", &v, &t) == 2)
			{
				//v--, t--;   // because the format is 1 based
				v--;   // because the format is 1 based
				groups[grp].has_texcoords[0] = true;
				groups[grp].faces[numfaces].vertIdx[0] = v;
#ifndef BUFFER_OBJECT
				groups[grp].faces[numfaces].texcIdx[0] = t;
#endif
				fscanf(file, "%lu/%lu", &v, &t);
				//v--, t--;   // because the format is 1 based
				v--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[1] = v;
#ifndef BUFFER_OBJECT
				groups[grp].faces[numfaces].texcIdx[1] = t;
#endif
				fscanf(file, "%lu/%lu", &v, &t);
				//v--, t--;   // because the format is 1 based
				v--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[2] = v;
#ifndef BUFFER_OBJECT
				groups[grp].faces[numfaces].texcIdx[2] = t;
#endif

				numfaces++;
				groups[grp].numfaces++;
				while(fscanf(file, "%lu/%lu", &v, &t) > 0)
				{
					v--;   // because the format is 1 based
					//v--, t--;   // because the format is 1 based
					groups[grp].faces[numfaces].vertIdx[0] = groups[grp].faces[numfaces-1].vertIdx[0];
#ifndef BUFFER_OBJECT
					groups[grp].faces[numfaces].texcIdx[0] = groups[grp].faces[numfaces-1].texcIdx[0];
#endif
					groups[grp].faces[numfaces].vertIdx[1] = groups[grp].faces[numfaces-1].vertIdx[2];
#ifndef BUFFER_OBJECT
					groups[grp].faces[numfaces].texcIdx[1] = groups[grp].faces[numfaces-1].texcIdx[2];
#endif
					groups[grp].faces[numfaces].vertIdx[2] = v;
#ifndef BUFFER_OBJECT
					groups[grp].faces[numfaces].texcIdx[2] = t;
#endif

					numfaces++;
					groups[grp].numfaces++;
				}
			}
			else
			{
				sscanf(buf, "%lu", &v);
				v--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[0] = v;
				fscanf(file, "%lu", &v);
				v--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[1] = v;
				fscanf(file, "%lu", &v);
				v--;   // because the format is 1 based
				groups[grp].faces[numfaces].vertIdx[2] = v;

				groups[grp].has_normals = false;
				groups[grp].has_texcoords[0] = false;

				numfaces++;
				groups[grp].numfaces++;
				while(fscanf(file, "%lu", &v) == 1)
				{
					v--;   // because the format is 1 based
					groups[grp].faces[numfaces].vertIdx[0] = groups[grp].faces[numfaces-1].vertIdx[0];
					groups[grp].faces[numfaces].vertIdx[1] = groups[grp].faces[numfaces-1].vertIdx[2];
					groups[grp].faces[numfaces].vertIdx[2] = v;

					numfaces++;
					groups[grp].numfaces++;
				}
			}
			break;
		default:
			fgets(buf, sizeof(buf), file);
		}
	}
	
	if (file) fclose(file);
	
}

void objMesh3D::writeFormat(const char * filename)
{
    EAZD_ASSERTALWAYS (0);
}

