#pragma once

#include "program.h"
#include <glm/glm.hpp>

class SkyRenderer {
public:
	SkyRenderer();

	void render(const glm::mat4& invViewProjection, GLuint envMap, unsigned int width, unsigned int height);

private:
	Program program_;
};