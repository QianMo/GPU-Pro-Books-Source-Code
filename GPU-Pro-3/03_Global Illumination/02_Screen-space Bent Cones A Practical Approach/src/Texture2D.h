#pragma once

#include "glhelper.h"

class Texture2D {
public:
	Texture2D() {
		glGenTextures(1, &id_);
	};

	~Texture2D() {
		if(id_) {
			glBindTexture(GL_TEXTURE_2D, 0);
			glDeleteTextures(1, &id_);
		}
	};

	inline void bind(unsigned int offset = 0) {
		GLCHECK(glActiveTexture(GL_TEXTURE0 + offset));
		GLCHECK(glBindTexture(GL_TEXTURE_2D, id_));
	}
	inline void unbind(unsigned int offset = 0) {
		GLCHECK(glActiveTexture(GL_TEXTURE0 + offset));
		GLCHECK(glBindTexture(GL_TEXTURE_2D, 0));
	}

	inline GLuint id() { return id_; };
private:
	GLuint id_;
};