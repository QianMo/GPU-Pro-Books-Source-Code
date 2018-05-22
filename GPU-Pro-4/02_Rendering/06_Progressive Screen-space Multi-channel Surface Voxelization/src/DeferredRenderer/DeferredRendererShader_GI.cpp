#include "shaders/DeferredRendererShader_GI.h"

DRShaderGI::DRShaderGI()
{
	initialized = false;
}

//----------------- Shader text ----------------------------

char DRShaderGI::DRSH_Vertex[] = "\n\
void main(void) \n\
{ \n\
   gl_Position = ftransform(); \n\
   gl_TexCoord[0] = gl_TextureMatrix[0]*gl_MultiTexCoord0; \n\
}";
