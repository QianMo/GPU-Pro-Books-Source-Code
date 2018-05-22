#pragma once

#include <gl/glew.h>

class FrameBuffer {
public:
	FrameBuffer();
	~FrameBuffer();

	void bindRead() const { glBindFramebuffer(GL_READ_FRAMEBUFFER, id_); };
	static void unbindRead() { glBindFramebuffer(GL_READ_FRAMEBUFFER, 0); };

	void bindDraw() const { glBindFramebuffer(GL_DRAW_FRAMEBUFFER, id_); };
	static void unbindDraw() { glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); };

	inline GLuint id() { return id_; };

	static bool checkStatus();

private:
	GLuint id_;
};