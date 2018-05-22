#include "framebuffer.h"
#include "glhelper.h"

FrameBuffer::FrameBuffer() {
	glGenFramebuffers(1, &id_);
}

FrameBuffer::~FrameBuffer() {
	glDeleteFramebuffers(1, &id_);
}

bool FrameBuffer::checkStatus() {
	return checkFramebufferStatus(__LINE__, __FILE__, __FUNCTION__);
}
