#pragma once

#include <gl/glew.h>
#include "program.h"
#include <vector>
#include "framebuffer.h"

class EnvMapConvolution {
public:
	EnvMapConvolution();
	~EnvMapConvolution();

	struct InputParameters {
		int sampleCount;
	};

	void resize(unsigned int size, unsigned int layerCount);

	void render(const InputParameters& params, GLuint envMapCube);

	GLuint& output() { return cubeMapArrayOutput_; };

private:
	void resizeFrameBuffers(size_t newSize);

private:
	Program program_;

	std::vector<FrameBuffer*> frameBuffers_;
	GLuint cubeMapArrayOutput_;
	unsigned int cubeSize_;
	unsigned int layerCount_;
};