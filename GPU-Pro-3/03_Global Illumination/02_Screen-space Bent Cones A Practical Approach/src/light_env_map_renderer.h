#pragma once

#include "program.h"
#include <glm/glm.hpp>
#include "framebuffer.h"

class LightEnvMapRenderer {
public:
	LightEnvMapRenderer();
	~LightEnvMapRenderer();

	struct InputParameters {
		struct Light {
			Light(const glm::vec3& _color, glm::vec3 _direction, float _innerAngle, float _outerAngle) :
			color(_color), direction(_direction), innerAngle(_innerAngle), outerAngle(_outerAngle) {}

			glm::vec3 color;
			glm::vec3 direction;
			float innerAngle;
			float outerAngle;
		};
		std::vector<Light> lights;
	};

	void resize(unsigned int size);

	void render(const InputParameters& params);

	GLuint& output() { return cubeMapOutput_; };

private:
	Program program_;

	FrameBuffer fbo_;
	GLuint cubeMapOutput_;
	unsigned int cubeSize_;
};