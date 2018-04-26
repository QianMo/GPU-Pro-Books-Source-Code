
#include "stencilMultiResSplatting.h"


GLSLProgram *displayTexLayer = 0;

void SetupDisplayLayerOfTextureArray( void )
{
	displayTexLayer = new GLSLProgram( NULL, NULL, "displayTexArrayLayer.frag.glsl", true, new PathList( "shaders/utilityShaders/" ) );
	displayTexLayer->SetTextureBinding( "textureArray", GL_TEXTURE0 );
}

void DisplayLayerOfTextureArray( GLuint texArrayID, int layerID )
{
	// Sadly, this can only be done with a shader (not the fixed function pipe)...
	//    If we don't have the shader, load it.
	if (!displayTexLayer)  SetupDisplayLayerOfTextureArray();

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0,1,0,1);
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glPushAttrib( GL_ALL_ATTRIB_BITS );
	glDisable( GL_LIGHTING );
	displayTexLayer->EnableShader();
	displayTexLayer->SetParameter( "layerID", layerID );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, texArrayID );
	glEnable( GL_TEXTURE_2D_ARRAY_EXT );
	glColor3f( 1, 1, 1 );
	glBegin( GL_QUADS );
		glTexCoord2f(0,0);	glVertex2f(0,0);
		glTexCoord2f(1,0);	glVertex2f(1,0);
		glTexCoord2f(1,1);	glVertex2f(1,1);
		glTexCoord2f(0,1);	glVertex2f(0,1);
	glEnd();
	glDisable( GL_TEXTURE_2D_ARRAY_EXT );
	glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, 0 );
	displayTexLayer->DisableShader();
	glPopAttrib();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

}
