#include "glhelper.h"

#include <sstream>

#include "shader_loader.h"

using namespace std;

namespace {
    void printStringWithLineNumbers(const std::string& output) {
        std::ostringstream oss;
        size_t lastStartPos = 0;
        size_t endPos = 0;
        for(unsigned int i=1;; ++i) {
            oss << i << ":\t";
            endPos = output.find(10, lastStartPos);
            oss << output.substr(lastStartPos, endPos-lastStartPos) << std::endl;
            lastStartPos = endPos+1;
            if(endPos == std::string::npos) break;
        }
        std::cout << oss.str() << std::endl;
    }

    GLuint createShader(GLenum type, char const* source, const std::string& output) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);

        if(!checkShader(shader, output)) {
            //printStringWithLineNumbers(source);
            return 0;
        }
        return shader;
    }

    void prependDefines(const std::vector<std::string>& defines, const std::string& source, std::string& output) {
        size_t versionPos = source.find("#version");
        if(versionPos != std::string::npos) {
            versionPos = source.find(10, versionPos);
            output = source.substr(0, versionPos+1);
        }
        for(size_t i=0; i<defines.size(); ++i) {
            output += "#define ";
            output += defines[i];
            output += " \n";
        }
        if(versionPos != std::string::npos) {
            output += source.substr(versionPos+1);
        }
        else {
            output += source;
        }
    }

    void replaceMacros(const std::vector<std::pair<std::string, std::string>>& macros, std::string& source) {
        for(size_t i=0; i<macros.size(); ++i) {
            size_t pos = source.find(macros[i].first);
            if(std::string::npos != pos) {
                source.replace(pos, macros[i].first.length(), macros[i].second);
            }
        }
    }
}

bool checkFramebufferStatus(unsigned int lineNbr, const std::string& file, const std::string& func) {
#ifdef _DEBUG
    std::string fbocheck;
    if(!checkFramebufferStatus(fbocheck)) {
        std::cerr << "Framebuffer error: " << fbocheck << std::endl << func << " in " << file << ", line: " << lineNbr << std::endl;
        return false;
    }
    return true;
#else
    return true;
#endif
}

/*inline*/ bool checkError(const char* Title) {
	int Error;
	if((Error = glGetError()) != GL_NO_ERROR)
	{
		std::string ErrorString;
		switch(Error)
		{
		case GL_INVALID_ENUM:
			ErrorString = "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:
			ErrorString = "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:
			ErrorString = "GL_INVALID_OPERATION";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			ErrorString = "GL_INVALID_FRAMEBUFFER_OPERATION";
			break;
		case GL_OUT_OF_MEMORY:
			ErrorString = "GL_OUT_OF_MEMORY";
			break;
		default:
			ErrorString = "UNKNOWN";
			break;
		}
		fprintf(stdout, "OpenGL Error(%s): %s\n", ErrorString.c_str(), Title);
	}
	return Error == GL_NO_ERROR;
}

/*inline*/ bool validateProgram(GLuint ProgramName) {
	if(!ProgramName)
		return false;

	glValidateProgram(ProgramName);
	GLint Result = GL_FALSE;
	glGetProgramiv(ProgramName, GL_VALIDATE_STATUS, &Result);

	if(Result == GL_FALSE)
	{
		fprintf(stdout, "Validate program\n");
		int InfoLogLength;
		glGetProgramiv(ProgramName, GL_INFO_LOG_LENGTH, &InfoLogLength);
		std::vector<char> Buffer(InfoLogLength);
		glGetProgramInfoLog(ProgramName, InfoLogLength, NULL, &Buffer[0]);
		fprintf(stdout, "%s\n", &Buffer[0]);
	}

	return Result == GL_TRUE;
}

/*inline*/ bool checkProgram(GLuint ProgramName, const std::string& output) {
	if(!ProgramName)
		return false;

	GLint Result = GL_FALSE;
	glGetProgramiv(ProgramName, GL_LINK_STATUS, &Result);

	int InfoLogLength;
	glGetProgramiv(ProgramName, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> Buffer(InfoLogLength);
	glGetProgramInfoLog(ProgramName, InfoLogLength, NULL, &Buffer[0]);
    if(InfoLogLength > 1)
	    fprintf(stdout, "linker error (%s): %s\n", output.c_str(), &Buffer[0]);

	return Result == GL_TRUE;
}

/*inline*/ bool checkShader(GLuint ShaderName, const std::string& output = std::string()) {
	if(!ShaderName)
		return false;

	GLint Result = GL_FALSE;
	glGetShaderiv(ShaderName, GL_COMPILE_STATUS, &Result);

    int InfoLogLength;
    glGetShaderiv(ShaderName, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> Buffer(InfoLogLength);
    glGetShaderInfoLog(ShaderName, InfoLogLength, NULL, &Buffer[0]);

    if(InfoLogLength > 1) {
        fprintf(stdout, "shader error (%s):\n%s\n", output.c_str(), &Buffer[0]);
    }

	return Result == GL_TRUE;
}

/*inline*/ GLuint createProgram(const std::string& vertShader, const std::string& fragShader) {
	bool valid = true;
	GLuint program = 0;

	// Compile a shader
	GLuint VertexShaderName = 0;
	if(valid && !vertShader.empty()) {
		VertexShaderName = createShaderFromFile(GL_VERTEX_SHADER, vertShader);
        if(!VertexShaderName) valid = false;
	}

	// Compile a shader
	GLuint FragmentShaderName = 0;
	if(valid && !fragShader.empty()) {
        FragmentShaderName = createShaderFromFile(GL_FRAGMENT_SHADER, fragShader);
        if(!FragmentShaderName) valid = false;
	}

	// Link a program
	if(valid) {
		program = glCreateProgram();
		if(VertexShaderName != 0)
			glAttachShader(program, VertexShaderName);
		if(FragmentShaderName != 0)
			glAttachShader(program, FragmentShaderName);
		glDeleteShader(VertexShaderName);
		glDeleteShader(FragmentShaderName);
		glLinkProgram(program);
		valid = checkProgram(program);
	}

	return program;
}

GLuint createShaderFromSource(GLenum type, const std::string& source, const std::string& output) {
    return createShader(type, source.c_str(), output);
}

GLuint createShaderFromSourceWithMacros(GLenum type, const std::string& source, const std::vector<std::pair<std::string, std::string>>& macroReplacements, const std::string& output) {
    std::string sourceCopy = source;
    replaceMacros(macroReplacements, sourceCopy);
    return createShader(type, sourceCopy.c_str(), output);
}

GLuint createShaderFromFile(GLenum type, const std::string& file) {
    std::string source = ShaderLoader::loadShader(file);
    return createShader(type, source.c_str(), file);
}

GLuint createShaderWithDefines(GLenum type, const std::string& file, const std::vector<std::string>& defines) {
    std::string source = ShaderLoader::loadShader(file);

    std::string newSource;
    prependDefines(defines, source, newSource);
    return createShader(type, newSource.c_str(), file);
}

GLuint createShaderWithLib(GLenum type, const std::string& shaderFile, const std::string& libFile, const std::vector<std::string>& defines /*= std::vector<std::string>()*/) {
    std::string source = ShaderLoader::loadShader(shaderFile);
    std::string lib = ShaderLoader::loadShader(libFile);
    source.append(lib);

    std::string newSource;
    prependDefines(defines, source, newSource);
    return createShader(type, newSource.c_str(), shaderFile);
}

GLuint createShaderWithMacro(GLenum type, const std::string& shaderFile, std::pair<std::string, std::string>& macroReplacement, const std::vector<std::string>& defines) {
    std::string source = ShaderLoader::loadShader(shaderFile);
    replaceMacros(std::vector<std::pair<std::string, std::string>>(1, macroReplacement), source);

    std::string newSource;
    prependDefines(defines, source, newSource);
    return createShader(type, newSource.c_str(), shaderFile);
}

GLuint createShaderWithMacros(GLenum type, const std::string& shaderFile, const std::vector<std::pair<std::string, std::string>>& macroReplacements, const std::vector<std::string>& defines) {
    std::string source = ShaderLoader::loadShader(shaderFile);
    replaceMacros(macroReplacements, source);

    std::string newSource;
    prependDefines(defines, source, newSource);
    return createShader(type, newSource.c_str(), shaderFile);
}

GLuint createShaderWithLibMacros(GLenum type, const std::string& shaderFile, const std::string& libFile, const std::vector<std::pair<std::string, std::string>>& macroReplacements, const std::vector<std::string>& defines) {
    std::string source = ShaderLoader::loadShader(shaderFile);
    std::string lib = ShaderLoader::loadShader(libFile);
    source.append(lib);

    replaceMacros(macroReplacements, source);

    std::string newSource;
    prependDefines(defines, source, newSource);
    return createShader(type, newSource.c_str(), shaderFile);
}

GLuint createProgram(const GLuint vertShader, const GLuint fragShader, const std::string& output) {
    GLuint program = glCreateProgram();
    if(vertShader != 0)
        glAttachShader(program, vertShader);
    if(fragShader != 0)
        glAttachShader(program, fragShader);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    glLinkProgram(program);

    if(!checkProgram(program, output)) return 0;
    return program;
}

std::string floatToString(float value) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss.precision(5);
    oss << value;
    return oss.str();
}

GLenum internalFormatToFormat(GLint internalFormat) {
	switch(internalFormat) {
		case GL_ALPHA:
		case GL_ALPHA8:
		case GL_ALPHA16:
		case GL_ALPHA16F_ARB:
		case GL_ALPHA32F_ARB:
			return GL_ALPHA;
		case GL_ALPHA16I_EXT:
		case GL_ALPHA16UI_EXT:
		case GL_ALPHA32I_EXT:
		case GL_ALPHA32UI_EXT:
			return GL_ALPHA_INTEGER_EXT;

		case GL_LUMINANCE:
		case GL_LUMINANCE8:
		case GL_LUMINANCE16:
		case GL_LUMINANCE16F_ARB:
		case GL_LUMINANCE32F_ARB:
			return GL_LUMINANCE;
		case GL_LUMINANCE16I_EXT:
		case GL_LUMINANCE16UI_EXT:
		case GL_LUMINANCE32I_EXT:
		case GL_LUMINANCE32UI_EXT:
			return GL_LUMINANCE_INTEGER_EXT;

		case GL_LUMINANCE_ALPHA:
		case GL_LUMINANCE8_ALPHA8:
		case GL_LUMINANCE16_ALPHA16:
		case GL_LUMINANCE_ALPHA16F_ARB:
		case GL_LUMINANCE_ALPHA32F_ARB:
			return GL_LUMINANCE_ALPHA;
		case GL_LUMINANCE_ALPHA16I_EXT:
		case GL_LUMINANCE_ALPHA16UI_EXT:
		case GL_LUMINANCE_ALPHA32I_EXT:
		case GL_LUMINANCE_ALPHA32UI_EXT:
			return GL_LUMINANCE_ALPHA_INTEGER_EXT;

		case GL_DEPTH_COMPONENT:
		case GL_DEPTH_COMPONENT16:
		case GL_DEPTH_COMPONENT24:
		case GL_DEPTH_COMPONENT32:
		case GL_DEPTH_COMPONENT32F:
			return GL_DEPTH_COMPONENT;

		case GL_DEPTH32F_STENCIL8:
		case GL_DEPTH24_STENCIL8:
			return GL_DEPTH_STENCIL;

		case GL_R8:
		case GL_R16:
		case GL_R16F:
		case GL_R32F:
			return GL_RED;
		case GL_R8I:
		case GL_R8UI:
		case GL_R16I:
		case GL_R16UI:
		case GL_R32I:
		case GL_R32UI:
			return GL_RED_INTEGER;

		case GL_RG8:
		case GL_RG16:
		case GL_RG16F:
		case GL_RG32F:
			return GL_RG;
		case GL_RG8I:
		case GL_RG8UI:
		case GL_RG16I:
		case GL_RG16UI:
		case GL_RG32I:
		case GL_RG32UI:
			return GL_RG_INTEGER;

		case GL_RGB:
		case GL_RGB8:
		case GL_RGB16F:
		case GL_RGB32F:
			return GL_RGB;
		case GL_RGB8I:
		case GL_RGB8UI:
		case GL_RGB16I:
		case GL_RGB16UI:
		case GL_RGB32I:
		case GL_RGB32UI:
			return GL_RGB_INTEGER;

		case GL_RGBA:
		case GL_RGBA8:
		case GL_RGBA16F:
		case GL_RGBA32F:
			return GL_RGBA;
		case GL_RGBA8I:
		case GL_RGBA8UI:
		case GL_RGBA16I:
		case GL_RGBA16UI:
		case GL_RGBA32I:
		case GL_RGBA32UI:
			return GL_RGBA_INTEGER;
	}
}

GLenum internalFormatToType(GLint internalFormat) {
	switch(internalFormat) {

		case GL_LUMINANCE:
		case GL_ALPHA:
		case GL_LUMINANCE_ALPHA:
		case GL_RGB:
		case GL_RGBA:

		case GL_LUMINANCE8:
		case GL_ALPHA8:
		case GL_LUMINANCE8_ALPHA8:
		case GL_R8:
		case GL_RG8:
		case GL_RGB8:		
		case GL_RGBA8:
			return GL_UNSIGNED_BYTE;

		case GL_R16:// right?
		case GL_RG16:// right?
		case GL_LUMINANCE16:
		case GL_ALPHA16:
		case GL_LUMINANCE16_ALPHA16:
		case GL_ALPHA16UI_EXT:
		case GL_LUMINANCE16UI_EXT:
		case GL_LUMINANCE_ALPHA16UI_EXT:
		case GL_R16UI:
		case GL_RG16UI:
		case GL_RGB16UI:
		case GL_RGBA16UI:
			return GL_UNSIGNED_SHORT;

		case GL_LUMINANCE16F_ARB:
		case GL_ALPHA16F_ARB:
		case GL_LUMINANCE_ALPHA16F_ARB:
		case GL_R16F:
		case GL_RG16F:
		case GL_RGB16F:
		case GL_RGBA16F:

		case GL_ALPHA32F_ARB:
		case GL_LUMINANCE32F_ARB:
		case GL_LUMINANCE_ALPHA32F_ARB:
		case GL_R32F:
		case GL_RG32F:
		case GL_RGB32F:
		case GL_RGBA32F:
			return GL_FLOAT;

		case GL_ALPHA16I_EXT:
		case GL_LUMINANCE16I_EXT:
		case GL_LUMINANCE_ALPHA16I_EXT:
		case GL_R16I:
		case GL_RG16I:
		case GL_RGB16I:
		case GL_RGBA16I:
			return GL_SHORT;

		case GL_ALPHA32I_EXT:
		case GL_LUMINANCE32I_EXT:
		case GL_LUMINANCE_ALPHA32I_EXT:
		case GL_R32I:
		case GL_RG32I:
		case GL_RGB32I:
		case GL_RGBA32I:
			return GL_INT;

		case GL_ALPHA32UI_EXT:
		case GL_LUMINANCE32UI_EXT:
		case GL_LUMINANCE_ALPHA32UI_EXT:
		case GL_RGB32UI:
		case GL_RGBA32UI:
			return GL_UNSIGNED_INT;

		case GL_DEPTH_COMPONENT:
		case GL_DEPTH_COMPONENT16:
		case GL_DEPTH_COMPONENT24:
		case GL_DEPTH_COMPONENT32:
		case GL_DEPTH_COMPONENT32F:
			return GL_FLOAT;

		case GL_DEPTH32F_STENCIL8:
			return GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
		case GL_DEPTH24_STENCIL8:
			return GL_UNSIGNED_INT_24_8;
	}
}
