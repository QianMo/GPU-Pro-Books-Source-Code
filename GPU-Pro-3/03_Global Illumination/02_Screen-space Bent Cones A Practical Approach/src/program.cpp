#include "program.h"

Program::~Program() {
	cleanup();
}

void Program::loadFiles(const std::string& VS, const std::string& FS, const std::vector<std::string>& defines) {
    cleanup();
	program_ = createProgram(createShaderWithDefines(GL_VERTEX_SHADER, VS, defines), createShaderWithDefines(GL_FRAGMENT_SHADER, FS, defines));
	//program_ = createProgram(VS, FS);
}

void Program::cleanup() {
    if(program_) {
        unUse();
        GLCHECK(glDeleteProgram(program_));
        program_ = 0;
    }
}

void Program::setProgramId(GLuint id) {
    if(id != program_) cleanup();
    program_ = id;
}
