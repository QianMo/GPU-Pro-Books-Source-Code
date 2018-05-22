#include "ShaderProgram.h"
#include "GLError.h"

//###########################################
//# constructor
//###########################################


ShaderProgram::ShaderProgram(string v, string f)
{
	V(createVertexShader(v));
	V(createFragmentShader(f));

	V(createProgram());
}

ShaderProgram::ShaderProgram(string v, string g,
                             GLenum inputType, GLenum outputType, GLuint maxPointsToEmit,
                             string f)
{
	V(createVertexShader(v));
	V(createGeometryShader(g));
	V(createFragmentShader(f));

	V(createProgram(inputType, outputType, maxPointsToEmit));
}



//###########################################
//# file input
//###########################################


// Reads a file and returns the content as a string
string ShaderProgram::readFile(string fileName)
{
	string fileContent;
	string line;

	ifstream file(fileName.c_str());
	if (file.is_open()) {
		while (!file.eof()){
			getline (file,line);
			line += "\n";
			fileContent += line;					
		}
		file.close();
	}
	else
		cout << "ERROR: Unable to open file " << fileName << endl;

	return fileContent;
}

// Print information about the compiling step
void ShaderProgram::printShaderInfoLog(GLuint shader)
{
    GLint infologLength = 0;
    GLsizei charsWritten  = 0;
    char *infoLog;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH,&infologLength);		
	infoLog = (char *)malloc(infologLength);
	glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
	printf("%s\n",infoLog);
	free(infoLog);
}

// Print information about the linking step
void ShaderProgram::printProgramInfoLog(GLuint program)
{
	GLint infoLogLength = 0;
	GLsizei charsWritten  = 0;
	char *infoLog;

	glGetProgramiv(program, GL_INFO_LOG_LENGTH,&infoLogLength);
   if(infoLogLength > 1)
      cout << endl << "  ====== SHADER PROGRAM INFO ====== " << endl;
	infoLog = (char *)malloc(infoLogLength);
	glGetProgramInfoLog(program, infoLogLength, &charsWritten, infoLog);
	printf("%s\n",infoLog);
	free(infoLog);
}


//###########################################
//# Create Shader Objects
//###########################################


// creates and compiles a Vertex Shader
void ShaderProgram::createVertexShader(string filename)
{
	// Create empty shader object (vertex shader)
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	string shaderSource = readFile(filename);
	const char* sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShader, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShader);

	cout << "[Vertex Shader]   " << filename << endl;
	//printShaderInfoLog(vertexShader);

	mVertexShader = vertexShader;

}

// creates and compiles a Fragment Shader
void ShaderProgram::createFragmentShader(string filename)
{
	// Create empty shader object (fragment shader)
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	string shaderSource = readFile(filename);
	const char* sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShader, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShader);

	cout << "[Fragment Shader] " << filename << endl;
	//printShaderInfoLog(fragmentShader);
	
	mFragmentShader = fragmentShader;
}

// creates and compiles a Geometry Shader
void ShaderProgram::createGeometryShader(string filename)
{
	// Create empty shader object (vertex shader)
	GLuint geometryShader = glCreateShader(GL_GEOMETRY_SHADER_EXT);

	// Read vertex shader source 
	string shaderSource = readFile(filename);
	const char* sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(geometryShader, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(geometryShader);

	cout << "[Geometry Shader] " << filename << endl;
	//printShaderInfoLog(geometryShader);

	mGeometryShader = geometryShader;

}


//###########################################
//# Create Program
//###########################################

// creates and links a shader program with vertex and fragment shader
void ShaderProgram::createProgram()
{
	// Create shader program
	GLuint shaderProgram = glCreateProgram();	

	// Attach shader
	glAttachShader(shaderProgram, mVertexShader);
	glAttachShader(shaderProgram, mFragmentShader);

	// Link program
	glLinkProgram(shaderProgram);
	printProgramInfoLog(shaderProgram);

	mProgram = shaderProgram;

}


// creates and links a shader program with vertex, geometry and fragment shader
void ShaderProgram::createProgram(GLenum inputType, GLenum outputType, GLuint maxPointsToEmit)
{
	// Create shader program
	GLuint shaderProgram = glCreateProgram();	

	// Attach shader
	glAttachShader(shaderProgram, mVertexShader);
   glAttachShader(shaderProgram, mGeometryShader);
	glAttachShader(shaderProgram, mFragmentShader);

   // set glProgramParameter (before linking)

   // max output vertices
   V(glProgramParameteriEXT(shaderProgram, GL_GEOMETRY_VERTICES_OUT_EXT, maxPointsToEmit));
   // Primitive Input Type (optional, default: GL_TRIANGLES)
   V(glProgramParameteriEXT(shaderProgram, GL_GEOMETRY_INPUT_TYPE_EXT, inputType));
   // Primitive Output Type
   V(glProgramParameteriEXT(shaderProgram, GL_GEOMETRY_OUTPUT_TYPE_EXT, outputType));


	// Link program
	V(glLinkProgram(shaderProgram));
	printProgramInfoLog(shaderProgram);

	mProgram = shaderProgram;
}



void ShaderProgram::useProgram()
{
	GLint activeProgram;
	glGetIntegerv(GL_CURRENT_PROGRAM, &activeProgram);

	if(activeProgram != static_cast<int>(mProgram))		
		glUseProgram(mProgram);
}


GLint ShaderProgram::getUniformLocation(string uniformName)
{
	GLint location = 0;

	map<string, GLint>::iterator uniformLocationsIterator;

	uniformLocationsIterator = mUniformLocations.find(uniformName);

	if(uniformLocationsIterator != mUniformLocations.end()) {
		location = uniformLocationsIterator->second;	
		//if(location<0) cout << uniformName << "  " << location<< endl;
		//assert(location >= 0);
	}
	else {
		location = glGetUniformLocation(mProgram, uniformName.c_str());
		//if(location<0) cout << uniformName << "  " << location<< endl;
		//assert(location >= 0);
		mUniformLocations[uniformName] = location;		
	}
	return location;
}

