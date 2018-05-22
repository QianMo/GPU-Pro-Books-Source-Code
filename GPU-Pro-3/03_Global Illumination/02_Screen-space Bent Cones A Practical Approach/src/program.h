#pragma once

#include "glhelper.h"

class Program {
public:
    Program() : program_(0) {};
	Program(GLuint program) : program_(program) {};

	~Program();

    void loadFiles(const std::string& VS, const std::string& FS, const std::vector<std::string>& defines = std::vector<std::string>());

    void cleanup();

    void setProgramId(GLuint id);
    GLuint id() const { return program_; };

    inline void use() { GLCHECK(glUseProgram(program_)); };
    inline void unUse() { GLCHECK(glUseProgram(0)); };

private:
    GLuint program_;
};