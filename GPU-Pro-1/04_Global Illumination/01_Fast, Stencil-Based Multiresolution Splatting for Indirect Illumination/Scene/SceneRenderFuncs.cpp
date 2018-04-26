/******************************************************************/
/* SceneRenderFuncs.cpp                                           */
/* -----------------------                                        */
/*                                                                */
/* The file defines part of a scene class that encapsulates the   */
/*     scene information.  In particular, this file includes      */
/*     common rendering commands that might be used either by     */
/*     applications or other scene methods.                       */
/*                                                                */
/* Chris Wyman (02/07/2008)                                       */
/******************************************************************/

#include "Utils/ImageIO/imageIO.h"
#include "DataTypes/Color.h"
#include "DataTypes/glTexture.h"
#include "DataTypes/Matrix4x4.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Scene/glLight.h"
#include "Objects/Group.h"
#include "Objects/Triangle.h"
#include "Objects/Sphere.h"
#include "Objects/Quad.h"
#include "Objects/Cylinder.h"
#include "Objects/Mesh.h"
#include "Materials/Material.h"
#include "Materials/GLMaterial.h"
#include "Materials/GLLambertianMaterial.h"
#include "Materials/GLLambertianTexMaterial.h"
#include "Materials/GLConstantMaterial.h"
#include "Materials/GLSLShaderMaterial.h"
#include "Utils/ProgramPathLists.h"
#include "Interface/SceneFileDefinedInteraction.h"
#include "Utils/Trackball.h"
#include "Utils/framebufferObject.h"

Matrix4x4 Scene::GetLightLookAtMatrix( int i )
{
	Point lightPos( light[i]->GetCurrentPos() );
	Point at( light[i]->GetLookAt() );
	//Point at( camera->GetAt() );
	Vector view = at-lightPos;
	view.Normalize();
	Vector perp = abs( view.Dot( Vector::YAxis() ) ) < 0.95 ? view.Cross( Vector::YAxis() ) : view.Cross( Vector::XAxis() );
	perp.Normalize();
	Vector up = perp.Cross( view );
	return Matrix4x4::LookAt( lightPos, at, up );
	//return Matrix4x4::Identity();
}

Matrix4x4 Scene::GetLightPerspectiveMatrix( int i, float aspect )
{
	return Matrix4x4::Perspective( light[i]->GetLightFovy(), aspect, 
		light[i]->GetLightNearPlane(), light[i]->GetLightFarPlane() );
}

void Scene::LightPerspectiveMatrix( int i, float aspect )
{
	gluPerspective( light[i]->GetLightFovy(), aspect, 
					light[i]->GetLightNearPlane(), light[i]->GetLightFarPlane() );
}

void Scene::LightPerspectiveInverseMatrix( int i, float aspect )
{
	Matrix4x4 mat = Matrix4x4::Perspective( light[i]->GetLightFovy(), aspect, 
					light[i]->GetLightNearPlane(), light[i]->GetLightFarPlane() ).Invert();
	glMultMatrixf( mat.GetDataPtr() );
}

void Scene::LightLookAtMatrix( int i )
{
	Point lightPos( light[i]->GetCurrentPos() );
	Point at( light[i]->GetLookAt() );
	//Point at( camera->GetAt() );
	Vector view = at-lightPos;
	view.Normalize();
	Vector perp = abs( view.Dot( Vector::YAxis() ) ) < 0.95 ? view.Cross( Vector::YAxis() ) : view.Cross( Vector::XAxis() );
	perp.Normalize();
	Vector up = perp.Cross( view );
	gluLookAt( lightPos.X(), lightPos.Y(), lightPos.Z(), 
		       at.X(), at.Y(), at.Z(), 
			   up.X(), up.Y(), up.Z() );
}

void Scene::LightLookAtInverseMatrix( int i )
{
	Point lightPos( light[i]->GetCurrentPos() );
	Vector view = light[i]->GetLookAt()-lightPos;
	//Vector view = camera->GetAt()-lightPos;
	view.Normalize();
	Vector perp = abs( view.Dot( Vector::YAxis() ) ) < 0.95 ? view.Cross( Vector::YAxis() ) : view.Cross( Vector::XAxis() );
	perp.Normalize();
	Vector up = perp.Cross( view );
	Matrix4x4 mat = Matrix4x4::LookAt( lightPos, light[i]->GetLookAt(), up ).Invert();
	//Matrix4x4 mat = Matrix4x4::LookAt( lightPos, camera->GetAt(), up ).Invert();
	glMultMatrixf( mat.GetDataPtr() );
}


void Scene::CreateShadowMap( FrameBuffer *shadMapBuf, float *shadMapMatrixTranspose, int lightNum, float shadowMapBias )
{
	// Go ahead and render the shadow map
	glPushAttrib( GL_COLOR_BUFFER_BIT );
	glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
	shadMapBuf->BindBuffer();
	shadMapBuf->ClearBuffers();
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	LightPerspectiveMatrix( lightNum, shadMapBuf->GetAspectRatio() );
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
		LightLookAtMatrix( lightNum );
		this->Draw( MATL_FLAGS_NOSCENEMATERIALS, OBJECT_OPTION_USE_LOWRES );
	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	shadMapBuf->UnbindBuffer();
	glPopAttrib();

	// Now compute the shadow map matrix for use by the current eye/camera view
	glPushMatrix();
	glLoadIdentity();
	glTranslatef( 0.5f, 0.5f, 0.5f + shadowMapBias );
	glScalef( 0.5f, 0.5f, 0.5f );
	LightPerspectiveMatrix( lightNum, shadMapBuf->GetAspectRatio() );
	LightLookAtMatrix( lightNum );
	camera->InverseLookAtMatrix();
	glGetFloatv(GL_TRANSPOSE_MODELVIEW_MATRIX, shadMapMatrixTranspose);
	glGetFloatv(GL_TRANSPOSE_MODELVIEW_MATRIX, shadMapTransMatrix[lightNum]);  // The scene's internal storage
	glGetFloatv(GL_MODELVIEW_MATRIX, shadMapMatrix[lightNum]);                 // The scene's internal storage
	glPopMatrix(); 

	shadowMapTexID[lightNum] = shadMapBuf->GetDepthTextureID();
}
