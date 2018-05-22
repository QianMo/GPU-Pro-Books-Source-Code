#include "GLError.h"
#include <sstream>
using std::stringstream;

void checkOpenGLError(char* file, int line)
{
 GLenum error = glGetError(); 

 if (error != GL_NO_ERROR){
	 std::cout << "[OpenGL ERROR]: " << error << " " << gluErrorString(error) << " in  line " << line << " in file " << file << std::endl;
	 assert(error != GL_NO_ERROR);
  }

}


std::string checkFramebufferStatus()
{
    switch (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT))
    {
    case GL_FRAMEBUFFER_COMPLETE_EXT:
        return "okay";

    case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
        return "Setup FBO failed. Unsupported framebuffer format.";

    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
         return "Setup FBO failed. Missing attachment.";

    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
         return "Setup FBO failed. Incomplete attachment.";

    case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
         return "Setup FBO failed. Attached images must have the same dimensions.";

    case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
         return "Setup FBO failed. Attached images must have the same format.";

    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
         return "Setup FBO failed. Missing draw buffer.";

    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
         return "Setup FBO failed. Missing read buffer.";

    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_EXT:
       return "Setup FBO failed. Attached images must have the same number of samples.";

    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT:
       return "Setup FBO failed. Incomplete layer targets.";

    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_COUNT_EXT:
       return "Setup FBO failed. Incomplete layer count.";

    default:
       {
          stringstream str;
          str << "Setup FBO failed. Fatal error. ";
          str << std::hex << (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT));
        return str.str();//("Setup FBO failed. Fatal error. " + int(glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)) );
       }
    }
}
