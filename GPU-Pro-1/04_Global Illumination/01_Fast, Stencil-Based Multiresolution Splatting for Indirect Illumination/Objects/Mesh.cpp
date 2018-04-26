/******************************************************************/
/* Mesh.cpp                                                       */
/* -----------------------                                        */
/*                                                                */
/* The file defines an object type that contains a triangle mesh  */
/*    read in from a single .obj, .m, .smf, or .hem file.         */
/*                                                                */
/* Chris Wyman (02/04/2008)                                       */
/******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "Mesh.h"
#include "Materials/Material.h"
#include "Utils/ImageIO/imageIO.h"
#include "Utils/ModelIO/glm.h"
#include "Scene/Scene.h"
#include "Utils/ModelIO/SimpleModelLib.h"

#define BUFFER_OFFSET(x)   ((GLubyte*) NULL + (x))


Mesh::Mesh( Material *matl ) : Object(matl), displayListID(0), hem(0), glm(0),
	meshXForm( Matrix4x4::Identity() ), modelType(-1), elementVBO(0), lowResFile(0),
	interleavedVertDataVBO(0), renderMode( MESH_RENDER_AS_DISPLAY_LIST ),
	glm_lowRes(0), hem_lowRes(0)
{
}

Mesh::~Mesh()
{
	if (filename) free( filename );
	if (lowResFile) free( lowResFile );
}

void Mesh::Preprocess( Scene *s ) 
{ 
	if (displayListID>0) return;

	if (modelType == TYPE_HEM_FILE)
	{
		if (renderMode == MESH_RENDER_AS_DISPLAY_LIST)
		{
			displayListID = hem->CreateOpenGLDisplayList( WITH_NORMALS );
			if (objectOptionFlags & OBJECT_OPTION_USE_LOWRES)
				displayListID_low = hem_lowRes->CreateOpenGLDisplayList( WITH_NORMALS );
			if (flags & OBJECT_FLAGS_ALLOWDRAWEDGESONLY)
				hem->CreateOpenGLDisplayList( USE_LINES | WITH_NORMALS | WITH_ADJACENT_FACE_NORMS );
			if ( (flags & OBJECT_FLAGS_ALLOWDRAWEDGESONLY) && (objectOptionFlags & OBJECT_OPTION_USE_LOWRES) )
				hem_lowRes->CreateOpenGLDisplayList( USE_LINES | WITH_NORMALS | WITH_ADJACENT_FACE_NORMS );
		}
		else if (renderMode == MESH_RENDER_AS_VBO_VERTEX_ARRAY)
		{
			interleavedVertDataVBO = hem->CreateOpenGLVBO( WITH_NORMALS );
			if (flags & OBJECT_FLAGS_ALLOWDRAWEDGESONLY)
				hem->CreateOpenGLVBO( USE_LINES | WITH_NORMALS | WITH_ADJACENT_FACE_NORMS );
			if (objectOptionFlags & OBJECT_OPTION_USE_LOWRES)
				interleavedVertDataVBO_low = hem_lowRes->CreateOpenGLVBO( WITH_NORMALS );
			if ( (flags & OBJECT_FLAGS_ALLOWDRAWEDGESONLY) && (objectOptionFlags & OBJECT_OPTION_USE_LOWRES) )
				hem_lowRes->CreateOpenGLVBO( USE_LINES | WITH_NORMALS | WITH_ADJACENT_FACE_NORMS );
		}
			
	}
	else if (modelType == TYPE_OBJ_FILE || modelType == TYPE_SMF_FILE)
	{
		if (renderMode == MESH_RENDER_AS_DISPLAY_LIST)
		{
			displayListID = glmList( glm, GLM_SMOOTH );
			if (objectOptionFlags & OBJECT_OPTION_USE_LOWRES)
				displayListID_low = glmList( glm_lowRes, GLM_SMOOTH );
		}
		else if (renderMode == MESH_RENDER_AS_VBO_VERTEX_ARRAY)
		{
			// Setup element vertex array for vertex buffer objects
			unsigned int *arrayIndices = (unsigned int *)malloc( glm->numtriangles * 3 * sizeof( unsigned int ) );
			for (unsigned int i=0; i<glm->numtriangles; i++)
			{
				arrayIndices[3*i+0] = glm->triangles[i].vindices[0];
				arrayIndices[3*i+1] = glm->triangles[i].vindices[1];
				arrayIndices[3*i+2] = glm->triangles[i].vindices[2];
			}
			glGenBuffers( 1, &elementVBO );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, elementVBO );
			glBufferData( GL_ELEMENT_ARRAY_BUFFER, glm->numtriangles*3*sizeof(unsigned int), 
						  arrayIndices, GL_STATIC_DRAW );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
			free( arrayIndices );

			// Setup data array for vertex buffer objects
			unsigned int dataSize = (glm->numvertices+1)*6*sizeof(float);
			float *floatData = (float *)malloc( dataSize );
			float *fptr = floatData;
			for (unsigned int i=0; i<(glm->numvertices+1)*3; i+=3)
			{
				*(fptr++) = glm->normArray[i];
				*(fptr++) = glm->normArray[i+1];
				*(fptr++) = glm->normArray[i+2];
				*(fptr++) = glm->vertices[i];
				*(fptr++) = glm->vertices[i+1];
				*(fptr++) = glm->vertices[i+2];
			}
			glGenBuffers( 1, &interleavedVertDataVBO );
			glBindBuffer( GL_ARRAY_BUFFER, interleavedVertDataVBO );
			glBufferData( GL_ARRAY_BUFFER, dataSize, floatData, GL_STATIC_DRAW );
			glBindBuffer( GL_ARRAY_BUFFER, 0 );
			free( floatData );
			elementCount = glm->numtriangles*3;

			if (objectOptionFlags & OBJECT_OPTION_USE_LOWRES)
			{
				// Setup element vertex array for vertex buffer objects
				arrayIndices = (unsigned int *)malloc( glm_lowRes->numtriangles * 3 * sizeof( unsigned int ) );
				for (unsigned int i=0; i<glm_lowRes->numtriangles; i++)
				{
					arrayIndices[3*i+0] = glm_lowRes->triangles[i].vindices[0];
					arrayIndices[3*i+1] = glm_lowRes->triangles[i].vindices[1];
					arrayIndices[3*i+2] = glm_lowRes->triangles[i].vindices[2];
				}
				glGenBuffers( 1, &elementVBO_low );
				glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, elementVBO_low );
				glBufferData( GL_ELEMENT_ARRAY_BUFFER, glm_lowRes->numtriangles*3*sizeof(unsigned int), 
							  arrayIndices, GL_STATIC_DRAW );
				glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
				free( arrayIndices );

				// Setup data array for vertex buffer objects
				dataSize = (glm_lowRes->numvertices+1)*6*sizeof(float);
				floatData = (float *)malloc( dataSize );
				fptr = floatData;
				for (unsigned int i=0; i<(glm_lowRes->numvertices+1)*3; i+=3)
				{
					*(fptr++) = glm_lowRes->normArray[i];
					*(fptr++) = glm_lowRes->normArray[i+1];
					*(fptr++) = glm_lowRes->normArray[i+2];
					*(fptr++) = glm_lowRes->vertices[i];
					*(fptr++) = glm_lowRes->vertices[i+1];
					*(fptr++) = glm_lowRes->vertices[i+2];
				}
				glGenBuffers( 1, &interleavedVertDataVBO_low );
				glBindBuffer( GL_ARRAY_BUFFER, interleavedVertDataVBO_low );
				glBufferData( GL_ARRAY_BUFFER, dataSize, floatData, GL_STATIC_DRAW );
				glBindBuffer( GL_ARRAY_BUFFER, 0 );
				free( floatData );
				elementCount_low = glm_lowRes->numtriangles*3;
			}
		}
	}

	if (glm) glmDelete( glm );
	//if (hem) hem->FreeNonGLMemory();
}



void Mesh::Draw( Scene *s, unsigned int matlFlags, unsigned int optionFlags )
{
	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		matl->Enable( s, matlFlags );
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		matl->EnableOnlyTextures( s );

	glPushMatrix();
	if (objMove) objMove->ApplyCurrentFrameMovementMatrix();
	if (ball) ball->MultiplyTrackballMatrix();
	glMultMatrixf( meshXForm.GetDataPtr() );

	if (renderMode == MESH_RENDER_AS_DISPLAY_LIST)
		glCallList( optionFlags & OBJECT_OPTION_USE_LOWRES ?
                    displayListID_low :
	                displayListID );
	else if (renderMode == MESH_RENDER_AS_VBO_VERTEX_ARRAY)	
	{
		if (hem && !(optionFlags & OBJECT_OPTION_USE_LOWRES) )
			hem->CallVBO( USE_TRIANGLES );
		else if (hem_lowRes && (optionFlags & OBJECT_OPTION_USE_LOWRES))
			hem_lowRes->CallVBO( USE_TRIANGLES );
		else
		{
			glBindBuffer( GL_ARRAY_BUFFER, 
				(optionFlags & OBJECT_OPTION_USE_LOWRES) ? interleavedVertDataVBO_low : interleavedVertDataVBO );
			glInterleavedArrays( GL_N3F_V3F, 0, BUFFER_OFFSET(0) );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 
				(optionFlags & OBJECT_OPTION_USE_LOWRES) ? elementVBO_low : elementVBO );
			glDrawElements( GL_TRIANGLES, 
				(optionFlags & OBJECT_OPTION_USE_LOWRES) ? elementCount_low : elementCount, 
				GL_UNSIGNED_INT, BUFFER_OFFSET(0) );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
			glBindBuffer( GL_ARRAY_BUFFER, 0 );
			glDisableClientState( GL_VERTEX_ARRAY );
			glDisableClientState( GL_NORMAL_ARRAY );
		}
	}

	glPopMatrix();

	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		matl->Disable();
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		matl->DisableOnlyTextures();
}


// Draw this object (or it's sub-objects only if they have some property)
void Mesh::DrawOnly( Scene *s, unsigned int propertyFlags, unsigned int matlFlags, unsigned int optionFlags )
{
	bool validToDraw = ((flags & propertyFlags) == propertyFlags);
	bool useLowRes =   (objectOptionFlags & optionFlags & OBJECT_OPTION_USE_LOWRES);

	if ( (propertyFlags & OBJECT_FLAGS_ALLOWDRAWEDGESONLY) && validToDraw )
	{
		if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
			matl->Enable( s, matlFlags );
		else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
			matl->EnableOnlyTextures( s );
		glPushMatrix();
		if (objMove) objMove->ApplyCurrentFrameMovementMatrix();
		if (ball) ball->MultiplyTrackballMatrix();
		glMultMatrixf( meshXForm.GetDataPtr() );
		if (renderMode == MESH_RENDER_AS_DISPLAY_LIST)
			( useLowRes ? hem_lowRes : hem )->CallList( USE_LINES );
		else if (renderMode == MESH_RENDER_AS_VBO_VERTEX_ARRAY)	
			( useLowRes ? hem_lowRes : hem )->CallVBO( USE_LINES );
		glPopMatrix();
		if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
			matl->Disable();
		else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
			matl->DisableOnlyTextures();
	}
	else if (validToDraw) 
		this->Draw( s, matlFlags, optionFlags );
}





Mesh::Mesh( char *linePtr, FILE *f, Scene *s ) : Object(0), hem(0), glm(0),
	displayListID(0), meshXForm( Matrix4x4::Identity() ), modelType(-1),
	elementVBO(0), interleavedVertDataVBO(0), renderMode( MESH_RENDER_AS_VBO_VERTEX_ARRAY ),
	lowResFile(0), glm_lowRes(0), hem_lowRes(0)
{
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	char file[ MAXLINELENGTH ] = { "No mesh file specified!" };
	char lowResFileName[ MAXLINELENGTH ] = { "No mesh file specified!" };

	// Find out what type of object we've got...
	ptr = StripLeadingTokenToBuffer( linePtr, token );
	MakeLower( token );
	if (!strcmp(token,"obj")) modelType = TYPE_OBJ_FILE;
	else if (!strcmp(token,"m") || !strcmp(token,"dotm")) modelType = TYPE_M_FILE;
	else if (!strcmp(token,"hem")) modelType = TYPE_HEM_FILE;
	else FatalError("Unknown mesh type '%s'!", token);

	// Now find out the other model parameters
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;
		if (TestCommonObjectProperties( token, ptr, s, f ))
			continue;
		else if (!strcmp(token, "file")) // The filename of the mesh!
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			strncpy( file, token, MAXLINELENGTH );
		}
		else if (!strcmp(token, "lowres")) // The filename of the low resolution mesh!
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			strncpy( lowResFileName, token, MAXLINELENGTH );
			objectOptionFlags |= OBJECT_OPTION_USE_LOWRES;
		}
		else if (!strcmp(token, "displaylist"))
			renderMode = MESH_RENDER_AS_DISPLAY_LIST;
		else if (!strcmp(token, "vbo") || !strcmp(token,"vertexarray") ||
			     !strcmp(token, "vertexbuffer"))
		    renderMode = MESH_RENDER_AS_VBO_VERTEX_ARRAY;
		else if (!strcmp(token, "trackball"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			int id = atoi(token);
			ball = new Trackball( s->GetWidth(), s->GetHeight() );
			s->SetupObjectTrackball( id, ball );
		}
		else if (!strcmp(token,"edges") || !strcmp(token,"enableedges") || !strcmp(token,"edgeenable"))
			flags |= OBJECT_FLAGS_ALLOWDRAWEDGESONLY;
		else if (!strcmp(token, "scale") || !strcmp(token,"center"))
			Warning("Mesh keyword '%s' not currently supported in OpenGL!", token);
		else if (!strcmp(token, "naivebvh") || !strcmp(token, "iterativebvh") || !strcmp(token, "arraybvh") ||
				 !strcmp(token, "skiptreebvh") || !strcmp(token, "kdtree"))
			Warning("Mesh acceleration structures (%s) not supported in OpenGL!", token);
		else if (!strcmp(token, "matrix")) // Transform the mesh?
		{
			Matrix4x4 mat( f, ptr );
			meshXForm *= mat;
		}
		else
			Error("Unknown command '%s' when loading Mesh!", token);
	}

	// Now load the mesh file(s)...  First get the filename
	filename = s->paths->GetModelPath( file );
	if (!filename) FatalError("Unable to open mesh '%s'!", file);

	// If we'll load a high and low res model, get the name of the low res version
	if ( objectOptionFlags & OBJECT_OPTION_USE_LOWRES )
	{
		lowResFile = s->paths->GetModelPath( lowResFileName );
		if (!lowResFile) FatalError("Unable to open mesh '%s'!", file);
	}

	// Load model.  This varies depending on the input type
	if (modelType == TYPE_HEM_FILE)
	{
		// Get the full res model
		hem     = new HalfEdgeModel( filename, TYPE_HEM_FILE   );

		// Check if we need a low res version
		if ( objectOptionFlags & OBJECT_OPTION_USE_LOWRES )
		{
			hem_lowRes = new HalfEdgeModel( lowResFile, TYPE_HEM_FILE );
			if (!hem_lowRes) flags &= ~OBJECT_OPTION_USE_LOWRES;
		}
	}
	else if (modelType == TYPE_OBJ_FILE || modelType == TYPE_SMF_FILE)
	{
		// OBJ files do not allow edge-only drawing!
		flags &= ~OBJECT_FLAGS_ALLOWDRAWEDGESONLY;

		// Get the full res model
		glm = glmReadOBJ( filename, s );
		glmUnitize( glm );
		glmFacetNormals( glm );
		glmVertexNormals( glm, 180 );

		// Check if we need a low res version
		if ( objectOptionFlags & OBJECT_OPTION_USE_LOWRES )
		{
			glm_lowRes = glmReadOBJ( lowResFile, s );
			if (!glm_lowRes) 
				objectOptionFlags &= ~OBJECT_OPTION_USE_LOWRES;
			else 
			{
				glmUnitize( glm_lowRes );
				glmFacetNormals( glm_lowRes );
				glmVertexNormals( glm_lowRes, 180 );
			}
		}
	}
	else
		FatalError("Curently unhandled mesh type: '.m'!");

}

