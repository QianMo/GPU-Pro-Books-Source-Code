/**
*	EWA filtering on the GPU
*	(original version by Cyril Crassin, adapted by Pavlos Mavridis)
*/

#include "ShadersManagment.h"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

///////////////////////////////////////////

void defineMacro(std::string &shaderSource, const char *macro, const char *value);
void defineMacro(std::string &shaderSource, const char *macro, int val);

///////////////////////////////////////////

//Size of the string, the shorter is better
#define STRING_BUFFER_SIZE 2048
char stringBuffer[STRING_BUFFER_SIZE];


struct ShaderMacroStruct{
	std::string macro;
	std::string value;
};
std::vector<ShaderMacroStruct>	shadersMacroList;

void resetShadersGlobalMacros(){
	shadersMacroList.clear();
}

void setShadersGlobalMacro(const char *macro, int val){
	ShaderMacroStruct ms;
	ms.macro=std::string(macro);

	char buff[128];
	sprintf(buff, "%d", val);
	ms.value=std::string(buff);

	shadersMacroList.push_back(ms);
}
void setShadersGlobalMacro(const char *macro, float val){
	ShaderMacroStruct ms;
	ms.macro=std::string(macro);

	char buff[128];
	sprintf(buff, "%ff", val);

	ms.value=std::string(buff);

	shadersMacroList.push_back(ms);
}

//GLSL shader program creation
GLuint createShaderProgram(const char *fileNameVS, const char *fileNameFS, GLuint programID){
	bool reload=programID!=0;

	GLuint vertexShaderID=0;
	GLuint fragmentShaderID=0;

	if(!reload){
		// Create GLSL program
		programID=glCreateProgram();
	}else{
		GLsizei count;
		GLuint shaders[2];
		glGetAttachedShaders(programID, 2, &count, shaders);

		GLint shadertype;
		glGetShaderiv(	shaders[0], GL_SHADER_TYPE, &shadertype);
		if(shadertype == GL_VERTEX_SHADER){
			vertexShaderID=shaders[0];
			fragmentShaderID=shaders[1];
		}else{
			vertexShaderID=shaders[1];
			fragmentShaderID=shaders[0];
		}
	}


	// Create vertex shader
	if(!reload){
		vertexShaderID=createShader(fileNameVS, GL_VERTEX_SHADER, 0);

		// Attach vertex shader to program object
		glAttachShader(programID, vertexShaderID);
	}else{
		createShader(fileNameVS, GL_VERTEX_SHADER, vertexShaderID);
	}

	// Create fragment shader
	if(!reload){
		fragmentShaderID=createShader(fileNameFS, GL_FRAGMENT_SHADER, 0);
		// Attach fragment shader to program object
		glAttachShader(programID, fragmentShaderID);
	}else{
		createShader(fileNameFS, GL_FRAGMENT_SHADER, fragmentShaderID);
	}
	


	// Link all shaders togethers into the GLSL program
	glLinkProgram(programID);
	checkProgramInfos(programID, GL_LINK_STATUS);

	// Validate program executability giving current OpenGL states
	glValidateProgram(programID);
	checkProgramInfos(programID, GL_VALIDATE_STATUS);

	return programID;
}


//GLSL shader creation (of a certain type, vertex shader, fragment shader oe geometry shader)
GLuint createShader(const char *fileName, GLuint shaderType, GLuint shaderID){
	if(shaderID==0){
		shaderID=glCreateShader(shaderType);
	}
	
	std::string shaderSource=loadTextFile(fileName);

	for(int i=0; i<shadersMacroList.size(); i++){
		defineMacro(shaderSource, shadersMacroList[i].macro.c_str(), shadersMacroList[i].value.c_str());
		//std::cout<<"["<<fileName<<"] Define: "<<shadersMacroList[i].macro<<"="<<shadersMacroList[i].value<<"\n";
	}
	//Passing shader source code to GL
	//Source used for "shaderID" shader, there is only "1" source code and the string is NULL terminated (no sizes passed)
	const char *src=shaderSource.c_str();
	glShaderSource(shaderID, 1, &src, NULL);

	//Compile shader object
	glCompileShader(shaderID);

	//Check compilation status
	GLint ok;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &ok);
	if (!ok){
		int ilength;
		glGetShaderInfoLog(shaderID, STRING_BUFFER_SIZE, &ilength, stringBuffer);
		std::cout<<"Compilation error ("<<fileName<<") : "<<stringBuffer; 
	}

	return shaderID;
}


//Text file loading for shaders sources
std::string loadTextFile(const char *name){
	//Source file reading
	std::string buff("");
	std::ifstream file;
	file.open(name);

	if(file.fail())
		std::cout<<"loadFile: unable to open file: "<<name;
	
	buff.reserve(1024*1024);

	std::string line;
	while(std::getline(file, line)){
		buff += line + "\n";
	}

	const char *txt=buff.c_str();

	return std::string(txt);
}

void checkProgramInfos(GLuint programID, GLuint stat){
	GLint ok = 0;
	glGetProgramiv(programID, stat, &ok);
	if (!ok){
		int ilength;
		glGetProgramInfoLog(programID, STRING_BUFFER_SIZE, &ilength, stringBuffer);
		std::cout<<"Program error :\n"<<stringBuffer<<"\n"; 
	}
}

void defineMacro(std::string &shaderSource, const char *macro, const char *value){
	char buff[512];


	sprintf(buff, "#define %s", macro);

	int mstart = (int)shaderSource.find(buff);
	sprintf(buff, "#define %s %s\n", macro, value);
	if(mstart>=0){
		//std::cout<<"Found at "<<mstart<<"\n";
		int mlen = (int)shaderSource.find("\n", mstart)-mstart+1 ;
		std::string prevval=shaderSource.substr(mstart, mlen);
		if( strcmp(prevval.c_str(), buff ) ){
			shaderSource.replace(mstart, mlen, buff);
		}
	}else{
		shaderSource.insert(0, buff);
	}

}

