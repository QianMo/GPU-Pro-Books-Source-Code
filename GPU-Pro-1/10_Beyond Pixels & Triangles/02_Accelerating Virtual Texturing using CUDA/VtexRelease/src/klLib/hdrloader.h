
/***********************************************************************************
	Created:	17:9:2002
	FileName: 	hdrloader.h
	Author:		Igor Kravtchenko
	
	Info:		Load HDR image and convert to a set of float32 RGB triplet.
************************************************************************************/
#include "vectors.h"
#include "FileSystem.h"

class HDRLoaderResult {
public:
	int width, height;
	// each pixel takes 3 float32, each component can be of any value...
	klVec4 *pixels; // Free with "delete []" after use
};

class HDRLoader {
public:
    static bool load(std::istream &stream, HDRLoaderResult &res);
};

