#ifndef _DR_SHADER_SPHERICALHARMONICS_
#define _DR_SHADER_SPHERICALHARMONICS_

class DRShader_SH
{
public:
    // 4-band spherical harmonics base creation and transformation functions
	static char DRSH_SH_Basis[];

	// Functions for the projection of scalar and vector values to 4-band SH basis
	static char DRSH_SH_Projection[];

	// Functions for the un-projection (reconstruction) of scalar and vector 
	// values from a 4-band SH basis
	static char DRSH_SH_Unprojection[];
};

#endif