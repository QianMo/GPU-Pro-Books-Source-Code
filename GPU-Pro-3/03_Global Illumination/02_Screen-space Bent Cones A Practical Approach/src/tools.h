#pragma once

#include <time.h>
#include <glm/glm.hpp>
#include <cmath>
#include <string>
#include <sstream>


class Tools {
public:
	static std::string intToString(const int i) {
		std::stringstream ss;
		ss << i;
		return ss.str();
	}

	static void inline srandTimeSeed() {
		srand(time(NULL));
	}

	static float inline frand() {
		return (float(rand()) / float(RAND_MAX));
	}

	static void unitSphericalToCarthesian(const glm::vec2& spherical, glm::vec3& result) {
		const float phi = spherical.x;
		const float theta = spherical.y;
		result.x = sinf(phi) * sinf(theta);
		result.y = cosf(phi) * sinf(theta);
		result.z = cosf(theta);
	}

	static void sequentialDrawBuffers(const int count) {
		GLenum buffers[16] = {
			GL_COLOR_ATTACHMENT0,
			GL_COLOR_ATTACHMENT1,
			GL_COLOR_ATTACHMENT2,
			GL_COLOR_ATTACHMENT3,
			GL_COLOR_ATTACHMENT4,
			GL_COLOR_ATTACHMENT5,
			GL_COLOR_ATTACHMENT6,
			GL_COLOR_ATTACHMENT7,
			GL_COLOR_ATTACHMENT8,
			GL_COLOR_ATTACHMENT9,
			GL_COLOR_ATTACHMENT10,
			GL_COLOR_ATTACHMENT11,
			GL_COLOR_ATTACHMENT12,
			GL_COLOR_ATTACHMENT13,
			GL_COLOR_ATTACHMENT14,
			GL_COLOR_ATTACHMENT15,
		};
		glDrawBuffers(count, buffers);
	}
};