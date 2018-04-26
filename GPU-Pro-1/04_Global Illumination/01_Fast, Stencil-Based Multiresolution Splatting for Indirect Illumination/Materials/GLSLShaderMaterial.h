/******************************************************************/
/* GLSLShaderMaterial.h                                           */
/* -----------------------                                        */
/*                                                                */
/* The file defines a complex OpenGL material defined by a GLSL   */
/*    shader.  This shader may consist of a vertex, geometry,     */
/*    and/or fragment shader.                                     */
/*                                                                */
/* Chris Wyman (02/01/2008)                                       */
/******************************************************************/

#ifndef __GLSLSHADERMATERIAL_H__
#define __GLSLSHADERMATERIAL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Material.h"
#include "DataTypes/Array1D.h"
#include "DataTypes/Color.h"
#include "Interface/UIVars/UIInt.h"
#include "Interface/UIVars/UIFloat.h"
#include "Interface/UIVars/UIBool.h"

class GLTexture;
class Scene;

#pragma warning( disable: 4996 )

#define SHADERMATL_NO_SPECIAL_BITS			0x00000000
#define SHADERMATL_IS_GLOSSY				0x00000001
#define SHADERMATL_IS_REFLECTIVE			0x00000002
#define SHADERMATL_IS_REFRACTIVE			0x00000004
#define SHADERMATL_USES_TEXTURE				0x00000008
#define SHADERMATL_USES_ALPHA_BLEND			0x00000010
#define SHADERMATL_ALLOWS_SHADOWMAPUSE		0x00000020
#define SHADERMATL_ALLOWS_CAUSTICMAPUSE		0x00000040

class PathList;

class GLSLShaderMaterial : public Material
{
private:
	GLSLProgram *shader;
	GLSLProgram *glslTemp;  // Used for temporary storage in specialized rendering modes
	unsigned int propertyFlags;

	Array1D<char *>  bindNames;
	Array1D<float *> bindPtrs;
	Array1D<char *>      bindTexNames;
	Array1D<GLTexture *> bindTexs;
	Array1D<char *>	 bindConstNames;
	Array1D<Color *> bindConstColors;

	unsigned int enables, disables;
	bool geomSettingsUpdated;
	GLenum geomInputType, geomOutputType;
	int geomMaxEmittedVerts;
	char *vertFile, *geomFile, *fragFile;

	bool usingShadows, usingCaustics, noSceneAmbient;

	void SetupShadowMap( int lightNum, GLuint texID, float *matrix );
	void DisableShadowMap( int lightNum );
	void SetupCausticMap( int lightNum, GLuint causticID, GLuint causticDepthID );
	void DisableCausticMap( int lightNum );

	Scene *inputScene;
	PathList *shaderPath;
public:
	// A constructor for the base material class
	GLSLShaderMaterial( char *matlName="<Unnamed GLSL Shader Material>" );
	GLSLShaderMaterial( FILE *f, Scene *s );
	~GLSLShaderMaterial();

	// All materials must be able to be Enable()'d or Disable()'d
	virtual void Enable( Scene *s, unsigned int flags=MATL_FLAGS_NONE );
	virtual void Disable( void );

	virtual void EnableOnlyTextures( Scene *s );
	virtual void DisableOnlyTextures( void );

	// There are a number of questions one might wish to know about a material
	//   Feel free to add more, but make sure there is a default method, do not
	//   have any abstract methods in this category!
	virtual bool UsesTexture( void )					{ return (propertyFlags & SHADERMATL_USES_TEXTURE)>0; }
	virtual bool UsesAlpha( void )						{ return (propertyFlags & SHADERMATL_USES_ALPHA_BLEND)>0; }
	virtual bool UsesLighting( void )					{ return true; }
	virtual bool UsesShaders( void )					{ return true; }
	virtual bool HandlesShadowMap( void )				{ return (propertyFlags & SHADERMATL_ALLOWS_SHADOWMAPUSE)>0; }

	// Perhaps some sorts of materials need to be handled specially...
	virtual bool IsReflective( void )					{ return (propertyFlags & SHADERMATL_IS_REFLECTIVE)>0; }
	virtual bool IsRefractive( void )					{ return (propertyFlags & SHADERMATL_IS_REFRACTIVE)>0; }
	virtual bool IsGlossy( void )						{ return (propertyFlags & SHADERMATL_IS_GLOSSY)>0; }

	// Get some data about the material
	virtual GLTexture *GetMaterialTexture( void )		{ return NULL; }
	virtual GLSLProgram *GetMaterialShader( void )		{ return shader; }

	virtual bool NeedsPreprocessing( void )				{ return shader==NULL; }
	virtual void Preprocess( Scene *s );

};




#endif

