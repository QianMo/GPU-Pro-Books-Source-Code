/******************************************************************/
/* ImportedSceneComponent.cpp                                     */
/* -----------------------                                        */
/*                                                                */
/* The file defines an container that stores a scene component,   */
/*    namely a set of triangles or other primitves that all have  */
/*    the same material propoerty.  This is not really different  */
/*    then the "mesh" object, except it requires different        */
/*    initialization techniques.                                  */
/*                                                                */
/* Eventually, "Imported Scene" and "Mesh" might want to be       */
/*    merged back into one construct, for user clarity.           */
/*                                                                */
/* Chris Wyman (04/17/2009)                                       */
/******************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include "ImportedSceneComponent.h"
#include "Materials/Material.h"
#include "Utils/ImageIO/imageIO.h"
#include "Utils/ModelIO/glm.h"
#include "Scene/Scene.h"
#include "Utils/ModelIO/SimpleModelLib.h"

#define BUFFER_OFFSET(x)   ((GLubyte*) NULL + (x))




ImportedSceneComponent::~ImportedSceneComponent()
{
	glDeleteBuffers( 1, &interleavedVertDataVBO );
}



void ImportedSceneComponent::Draw( Scene *s, unsigned int matlFlags, unsigned int optionFlags )
{
	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		matl->Enable( s, matlFlags );
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		matl->EnableOnlyTextures( s );

	glPushMatrix();
	if (objMove) objMove->ApplyCurrentFrameMovementMatrix();
	if (ball) ball->MultiplyTrackballMatrix();

	if (renderMode == MESH_RENDER_AS_DISPLAY_LIST)
		glCallList( displayListID );
	else if (renderMode == MESH_RENDER_AS_VBO_VERTEX_ARRAY)	
	{
		glBindBuffer( GL_ARRAY_BUFFER, interleavedVertDataVBO );
		glInterleavedArrays( interleavedFormat, 0, BUFFER_OFFSET(0) );
		glDrawArrays( GL_TRIANGLES, 0, elementCount );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
		glDisableClientState( GL_TEXTURE_COORD_ARRAY );
		glDisableClientState( GL_VERTEX_ARRAY );
		glDisableClientState( GL_NORMAL_ARRAY );
	}

	glPopMatrix();

	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		matl->Disable();
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		matl->DisableOnlyTextures();
}


// Draw this object (or it's sub-objects only if they have some property)
void ImportedSceneComponent::DrawOnly( Scene *s, unsigned int propertyFlags, unsigned int matlFlags, unsigned int optionFlags )
{
	bool validToDraw = ((flags & propertyFlags) == propertyFlags);
	if (validToDraw) this->Draw( s, matlFlags, optionFlags );
}


ImportedSceneComponent::ImportedSceneComponent( Scene *s, _GLMmodel *glm, _GLMgroup *component ) :
    Object(0), displayListID(0), interleavedVertDataVBO(0), elementCount(0), renderMode( MESH_RENDER_AS_DISPLAY_LIST )
{
	// This is perfectly fine, since it is either a valid material or NULL (the default)
	matl = component->fwMaterial;
	displayListID = glmGroupList( glm, GLM_SMOOTH, component );
}

ImportedSceneComponent::ImportedSceneComponent( Scene *s, _GLMmodel *glm, _GLMgroup *component, GLenum interleavedFormat ) : 
    Object(0), elementCount(0), interleavedFormat( interleavedFormat ),
	interleavedVertDataVBO( 0 ), renderMode( MESH_RENDER_AS_VBO_VERTEX_ARRAY )
{
	// This is perfectly fine, since it is either a valid material or NULL (the default)
	matl = component->fwMaterial;

	// Setup an array of vertices
	unsigned int dataSize = component->numtriangles*3*sizeof(float);
	dataSize *= ( interleavedFormat==GL_V3F ? 3 :
		( interleavedFormat==GL_N3F_V3F ? 6 : 8 /* for GL_T2F_N3F_V3F */ ) );
	float *floatData = (float *)malloc( dataSize );
	float *fptr = floatData;
	//for (unsigned int i=0; i<(glm->numvertices+1)*3; i+=3)
	int count=0;
	for (unsigned int i=0; i < component->numtriangles; i++)
	{
		GLMtriangle *tri = &(glm->triangles[component->triangles[i]]);

		for (unsigned int j=0; j < 3; j++)
		{
			if (glm->texcoords)
			{
				*(fptr++) = glm->texcoords[2*tri->tindices[j]];
				*(fptr++) = glm->texcoords[2*tri->tindices[j]+1];
				count+=2;
			}
			if (glm->normals)
			{
				*(fptr++) = glm->normals[3*tri->nindices[j]];
				*(fptr++) = glm->normals[3*tri->nindices[j]+1];
				*(fptr++) = glm->normals[3*tri->nindices[j]+2];
				count+=3;
			}
			*(fptr++) = glm->vertices[3*tri->vindices[j]];
			*(fptr++) = glm->vertices[3*tri->vindices[j]+1];
			*(fptr++) = glm->vertices[3*tri->vindices[j]+2];
			count+=3;
		}
	}
	glGenBuffers( 1, &interleavedVertDataVBO );
	glBindBuffer( GL_ARRAY_BUFFER, interleavedVertDataVBO );
	glBufferData( GL_ARRAY_BUFFER, dataSize, floatData, GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	free( floatData );
	elementCount = component->numtriangles * 3;
}

