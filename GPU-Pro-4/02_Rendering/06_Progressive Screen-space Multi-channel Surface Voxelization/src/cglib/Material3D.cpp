
#include <stdlib.h>

#ifdef INTEL_COMPILER
	#include <dvec.h>
	#include <mathimf.h>
#else
	#include <math.h>
#endif

#include <string.h>

#include "Material3D.h"
#include "Texture2D.h"

#ifdef WIN32
#pragma warning (disable : 4996)
#endif

void Material3D::generateAutoNormalMap (Texture2D &normal_map, Texture2D color_map)
{
  // TODO
}

void Material3D::generateNormalMapFromBumpMap (Texture2D &normal_map, Texture2D bump_map)
{
  // TODO
}

Material3D::Material3D()
{
	setAmbient  (Vector3D (0.2f,0.2f,0.2f));
	setDiffuse  (Vector3D (0.8f,0.8f,0.8f));
	setSpecular (Vector3D (0.0f,0.0f,0.0f));
	setEmission (Vector3D (0.0f,0.0f,0.0f));
	setShininess (0);
	setMetallicShine (0.0f);
	setRoughness (0.0f);
	setAlpha (1.0f);
	type = MATERIAL_BLINN;
	setIndexOfRefraction (1.0);
	strcpy(name,"_Default_Material");
	for (unsigned int i=0; i < MATERIAL_MAP_COUNT; i++)
	{
	    has_texture[i] = false;
	    texturestr[i] = NULL;
	    texturemap[i] = 0;
	}
    has_auto_normal_map = false;
}

Material3D::~Material3D()
{
	for (unsigned int i=0; i < MATERIAL_MAP_COUNT; i++)
        if (texturestr[i])
			free (texturestr[i]);

	glDeleteTextures (MATERIAL_MAP_COUNT, texturemap);
}

char * Material3D::dumpType()
{
         if (type == MATERIAL_NONE)  return (char *) "none";
    else if (type == MATERIAL_BLINN) return (char *) "blinn";
    else if (type == MATERIAL_METAL) return (char *) "metal";
    else if (type == MATERIAL_ROUGH) return (char *) "rough";
    else                             return (char *) "none";
}

void Material3D::dump()
{
    fprintf (stdout, "\tMaterial \"%s\" info:\n", name);
    fprintf (stdout, "\t\tambient  : % f % f % f\n", ambient [0], ambient [1], ambient [2]);
    fprintf (stdout, "\t\tdiffuse  : % f % f % f\n", diffuse [0], diffuse [1], diffuse [2]);
    fprintf (stdout, "\t\tspecular : % f % f % f\n", specular[0], specular[1], specular[2]);
    fprintf (stdout, "\t\temission : % f % f % f\n", emission[0], emission[1], emission[2]);
    fprintf (stdout, "\t\tshininess: %d \talpha : % f\n", shininess, alpha);
    fprintf (stdout, "\t\tmetallic_shine: % f \troughness : % f\n", metallic_shine, roughness);
    fprintf (stdout, "\t\tindex_of_refraction : % f \ttype : %s\n", ior, dumpType ());
    for (unsigned int i=0; i < MATERIAL_MAP_COUNT; i++)
        fprintf (stdout, "\t\thas_texture[%d] : %s \ttexture_map_id[%d] : %d \ttexturestr[%d] : %s\n", i, has_texture[i] ? "yes" : "no ", i, texturemap[i], i, texturestr[i]);
    fflush  (stdout);
}

void Material3D::draw()
{
	glDisable (GL_COLOR_MATERIAL);

    float attrib[4];

    attrib[0] = ambient[0];
    attrib[1] = ambient[1];
    attrib[2] = ambient[2];
    attrib[3] = alpha;
    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT,  attrib);

    attrib[0] = diffuse[0];
    attrib[1] = diffuse[1];
    attrib[2] = diffuse[2];
    attrib[3] = alpha;
    glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, attrib);

    attrib[0] = specular[0];
    attrib[1] = specular[1];
    attrib[2] = specular[2];
    attrib[3] = 1;
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, attrib);

    attrib[0] = emission[0];
    attrib[1] = emission[1];
    attrib[2] = emission[2];
    attrib[3] = 1;
    glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, attrib);

    glMateriali (GL_FRONT_AND_BACK, GL_SHININESS, shininess);
}

