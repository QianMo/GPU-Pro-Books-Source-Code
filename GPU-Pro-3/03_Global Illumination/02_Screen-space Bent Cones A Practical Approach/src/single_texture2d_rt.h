#pragma once

#include "multiple_texture2d_rt.h"

class SingleTexture2DRT : public MultipleTexture2DRT {
public:
	SingleTexture2DRT(GLenum glTextureFormat) : MultipleTexture2DRT(glTextureFormat) {};

	inline GLuint rtId() const { return MultipleTexture2DRT::rtId(0); };
	inline void bindTexture(unsigned int offset = 0) const { MultipleTexture2DRT::bindTexture(0, offset); };
};

