#ifndef _DR_SHADER_GI_
#define _DR_SHADER_GI_

#include "DeferredRendererShader.h"

// All GI shaders must be derived from this class.
class DRShaderGI : public DRShader
{
protected:
	static char DRSH_Vertex[];
	bool initialized;

	float fmat_MVP[16], fmat_P[16], fmat_MVP_inv[16], fmat_P_inv[16];
	
public:
	DRShaderGI();
	void setModelViewProjectionMatrix(const GLdouble* mat){for (int j=0;j<16;j++) fmat_MVP[j] = (float)mat[j];}
	void setModelViewProjectionMatrixInverse(const GLdouble* mat){for (int j=0;j<16;j++) fmat_MVP_inv[j] = (float)mat[j];}
	void setProjectionMatrix(const GLdouble* mat){for (int j=0;j<16;j++) fmat_P[j] = (float)mat[j];}
	void setProjectionMatrixInverse(const GLdouble* mat){for (int j=0;j<16;j++) fmat_P_inv[j] = (float)mat[j];}
};


#endif