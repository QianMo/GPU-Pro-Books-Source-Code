#ifndef _DR_SHADER_SHADOWFUNCTIONS_
#define _DR_SHADER_SHADOWFUNCTIONS_

class DRShader_ShadowFunctions
{
public:
	// 16-tap Gauss sample distribution with variable size kernel and position-dependent
	// jittering. Good for soft shadows (penumbra depends on distance from point of contact).
	static char DRSH_Shadow_Gaussian[];
};

#endif
