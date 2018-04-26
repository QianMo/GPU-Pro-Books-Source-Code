/******************************************************************/
/* ImportedScene.h                                                */
/* -----------------------                                        */
/*                                                                */
/* The file defines an container that contains an imported scene. */
/*    An imported "Scene" is conceptually different than a "Mesh" */
/*    in the sense that a scene may have multiple objects or      */
/*    components, each with a separate material type.  While this */
/*    distinction is somewhat arbitrary, and thus classifies many */
/*    true "objects" as scenes, it simplifies mesh construction   */
/*    somewhat.                                                   */
/*                                                                */
/* Eventually, "Imported Scene" and "Mesh" might want to be       */
/*    merged back into one construct, for user clarity.           */
/*                                                                */
/* Chris Wyman (04/17/2009)                                       */
/******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "ImportedScene.h"
#include "ImportedSceneComponent.h"
#include "Materials/Material.h"
#include "Utils/ImageIO/imageIO.h"
#include "Utils/ModelIO/glm.h"
#include "Scene/Scene.h"
#include "Utils/ModelIO/SimpleModelLib.h"

#define BUFFER_OFFSET(x)   ((GLubyte*) NULL + (x))


ImportedScene::~ImportedScene()
{
	if (filename) free( filename );
}

void ImportedScene::Preprocess( Scene *s ) 
{ 
	printf("    (-) Converting scene '%s' to internal format...\n", glm->pathname);

	if (modelType == TYPE_OBJ_FILE || modelType == TYPE_SMF_FILE)
	{
		// What format data do we have?
		interleavedFormat = GL_V3F;

		// Check if we have surface normals, if so, change the interleaved format
		if (glm->normals) 
			interleavedFormat = GL_N3F_V3F;

		// Check if we have texture coordinates, if so, change the interleaved format
		if (glm->texcoords && interleavedFormat == GL_V3F)
			interleavedFormat = GL_T2F_V3F;
		else if (glm->texcoords && interleavedFormat == GL_N3F_V3F)
			interleavedFormat = GL_T2F_N3F_V3F;


		// Now actually populate our child components
		for ( GLMgroup *currentGroup = glm->groups; 
			  currentGroup; 
			  currentGroup = currentGroup->next )
		{
			//printf("       (*) Processing component '%s'...\n", currentGroup->name );
			if (renderMode == MESH_RENDER_AS_DISPLAY_LIST)
				objs.Add( new ImportedSceneComponent( s, glm, currentGroup ) );
			else if (renderMode == MESH_RENDER_AS_VBO_VERTEX_ARRAY)
				objs.Add( new ImportedSceneComponent( s, glm, currentGroup, interleavedFormat ) );
		}

	}

	if (glm) glmDelete( glm );
}



void ImportedScene::Draw( Scene *s, unsigned int matlFlags, unsigned int optionFlags )
{
	unsigned int currentFlags = matlFlags;
	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
	{
		matl->Enable( s, matlFlags );
		currentFlags |= MATL_FLAGS_NOSCENEMATERIALS;
	}
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
	{
		matl->EnableOnlyTextures( s );
		currentFlags |= MATL_FLAGS_NOSCENEMATERIALS;
		currentFlags &= ~MATL_FLAGS_ENABLEONLYTEXTURES;
	}

	glPushMatrix();
	if (objMove) objMove->ApplyCurrentFrameMovementMatrix();
	if (ball) ball->MultiplyTrackballMatrix();
	glMultMatrixf( sceneXForm.GetDataPtr() );

	for (unsigned int i=0;i<objs.Size();i++) 
		objs[i]->Draw( s, currentFlags, optionFlags ); 

	glPopMatrix();

	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		matl->Disable();
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		matl->DisableOnlyTextures();
}


// Draw this object (or it's sub-objects only if they have some property)
void ImportedScene::DrawOnly( Scene *s, unsigned int propertyFlags, unsigned int matlFlags, unsigned int optionFlags )
{
	bool meetsReqs       = ((propertyFlags & flags) == propertyFlags);
	bool someReqsNotMet  = ((propertyFlags & ~flags) > 0);
	unsigned int missingFlags = propertyFlags ^ (propertyFlags & flags);
	if (meetsReqs && !someReqsNotMet) 
		this->Draw( s, matlFlags, optionFlags );
	else
	{
		unsigned int currentFlags = matlFlags;
		if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		{
			matl->Enable( s, matlFlags );
			currentFlags |= MATL_FLAGS_NOSCENEMATERIALS;
		}
		else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		{
			matl->EnableOnlyTextures( s );
			currentFlags |= MATL_FLAGS_NOSCENEMATERIALS;
			currentFlags &= ~MATL_FLAGS_ENABLEONLYTEXTURES;
		}

		glPushMatrix();
		if (ball) ball->MultiplyTrackballMatrix();
		glMultMatrixf( sceneXForm.GetDataPtr() );

		for (unsigned int i=0;i<objs.Size();i++) 
			objs[i]->DrawOnly( s, missingFlags, currentFlags, optionFlags ); 

		glPopMatrix();

		if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
			matl->Disable();
		else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
			matl->DisableOnlyTextures();
	}
}





ImportedScene::ImportedScene( char *linePtr, FILE *f, Scene *s ) : Object(0), glm(0),
	sceneXForm( Matrix4x4::Identity() ), modelType(-1),
	renderMode( MESH_RENDER_AS_VBO_VERTEX_ARRAY )
{
	float facetSmoothingAngle = 180;
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	char file[ MAXLINELENGTH ] = { "No mesh file specified!" };
	char lowResFileName[ MAXLINELENGTH ] = { "No mesh file specified!" };

	// Find out what type of object we've got...
	ptr = StripLeadingTokenToBuffer( linePtr, token );
	MakeLower( token );
	if (!strcmp(token,"obj")) modelType = TYPE_OBJ_FILE;
	else FatalError("Scene type '%s' not supported for importing!", token);

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
		else if (!strcmp(token, "file")) // The filename of the scene!
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			strncpy( file, token, MAXLINELENGTH );
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
		else if (!strcmp(token, "smoothing") || !strcmp(token,"smooth") || !strcmp(token,"smoothangle"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			facetSmoothingAngle = atof(token);
		}
		else if (!strcmp(token, "scale") || !strcmp(token,"center"))
			Warning("Import keyword '%s' not currently supported in OpenGL!", token);
		else if (!strcmp(token, "naivebvh") || !strcmp(token, "iterativebvh") || !strcmp(token, "arraybvh") ||
				 !strcmp(token, "skiptreebvh") || !strcmp(token, "kdtree"))
			Warning("Import acceleration structures (%s) not supported in OpenGL!", token);
		else if (!strcmp(token, "matrix")) // Transform the mesh?
		{
			Matrix4x4 mat( f, ptr );
			sceneXForm *= mat;
		}
		else
			Error("Unknown command '%s' when importing scene!", token);
	}

	// Now load the mesh file(s)...  First get the filename
	filename = s->paths->GetModelPath( file );
	if (!filename) FatalError("Unable to open file '%s' to import!", file);

	// Load model.  This varies depending on the input type
	if (modelType == TYPE_OBJ_FILE || modelType == TYPE_SMF_FILE)
	{
		// OBJ files do not allow edge-only drawing!
		flags &= ~OBJECT_FLAGS_ALLOWDRAWEDGESONLY;

		// Get the full res model
		glm = glmReadOBJ( filename, s );
		glmUnitize( glm );
		glmFacetNormals( glm );
		glmVertexNormals( glm, facetSmoothingAngle );
	}
	else
		FatalError("An unhandled scene type specified during ImportedScene constructor!?");

}

