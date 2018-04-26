
#include	<assert.h>
#include	<stdio.h>
#include	<stdlib.h>

#include "stb_image/stb_image.h"

#include	"glUtils.h"

bool getLastGLError()
{
   GLenum  errorCode = glGetError ();

   if ( errorCode == GL_NO_ERROR ) {
      return true;
   }

   printf ( "GL Error %d: %s\n", errorCode, gluErrorString(errorCode) );

   return false;
}

bool printInfo(GLhandleARB handle, const char* shaderName)
{
   int infoLength = 0;  

   glGetObjectParameterivARB ( handle, GL_OBJECT_INFO_LOG_LENGTH_ARB, &infoLength );

   if (!getLastGLError()) {
      return false;
   }

   if ( infoLength > 1 ) {
      
      GLcharARB * info = (GLcharARB*)malloc(infoLength);
      
      int charsWritten = 0;
      glGetInfoLogARB(handle, infoLength, &charsWritten, info);

      printf ( "ARB message (%s): %s\n", shaderName, info );

      free(info);
   }

   return true;
}

bool loadShader(GLhandleARB shader, const char * fileName)
{
   FILE * file = fopen(fileName, "rb");

   if (!file) {
      printf("Failed to load %s\n", fileName);
      assert(0);
      return false;
   }

   fseek(file, 0, SEEK_END);	
   int size = ftell(file);

   if ( size < 1 ) {
      printf("Failed to load %s\n", fileName);
      fclose(file);		
      return false;
   }

   char *buf = (char*)malloc(size);

   fseek(file, 0, SEEK_SET);

   if (fread(buf, 1, size, file) != size ) {
      fclose(file);
      printf("Failed to load %s\n", fileName);
      return false;
   }

   fclose(file);

   GLint compileStatus;

   glShaderSourceARB(shader, 1, (const char **)&buf, &size );

   glCompileShaderARB (shader);

   if (!getLastGLError()) {
      return false;
   }

   glGetObjectParameterivARB(shader, GL_OBJECT_COMPILE_STATUS_ARB, &compileStatus);

   printInfo(shader, fileName);

   free(buf);

   return compileStatus != 0;

}

bool setUniformVector ( GLhandleARB program, const char * name, const float * value )
{
   int location = glGetUniformLocationARB(program, name);

   if (location < 0) {
      return false;
   }

   glUniform4fvARB ( location, 1, value );   

   return true;
}

bool setUniformFloat ( GLhandleARB program, const char * name, float value )
{
   int location = glGetUniformLocationARB(program, name);

   if (location < 0) {
      return false;
   }

   glUniform1fARB ( location, value );

   return true;
}

bool setUniformTexture(GLhandleARB location, GLuint textureObject, int type, int unitNumber)
{
   if ( !( type == GL_TEXTURE_2D || type == GL_TEXTURE_CUBE_MAP) ) {
      return false;
   }

   glActiveTexture(GL_TEXTURE0 + unitNumber);
   glBindTexture(type, textureObject);

   glUniform1iARB(location, unitNumber);

   return true;
}

bool checkGL()
{
   if ( glewInit () != GLEW_OK ) {
      getLastGLError();
		printf("GLEW initialization failed\n");		
		return false;
	}
	
   if (!glewIsSupported("GL_VERSION_2_0 ")) {
      printf("OpenGL 2.0 not supported.");
      return false;
   }

   return true;
}

bool createShaders(const char* vshName, GLhandleARB* vertexShader, const char* fshName, GLhandleARB* fragmentShader, GLhandleARB* program)
{  
   if (!vshName || !vertexShader || !fshName || !fragmentShader || !program) {
      assert(0);
      printf("Wrong parameters provided while creating shaders.");
      return false;
   }

   *vertexShader   = glCreateShaderObjectARB( GL_VERTEX_SHADER_ARB );
   *fragmentShader = glCreateShaderObjectARB( GL_FRAGMENT_SHADER_ARB );
   
   if (!loadShader( *vertexShader, vshName )) {      
      return false;
   }

   if (!loadShader( *fragmentShader, fshName)) {
      return false;
   }
   
   *program = glCreateProgramObjectARB();

   glAttachObjectARB(*program, *vertexShader);
   glAttachObjectARB(*program, *fragmentShader);
   
   glLinkProgramARB (*program);

   if (!getLastGLError()) {
      return false;
   }

   GLint linked;
   glGetObjectParameterivARB(*program, GL_OBJECT_LINK_STATUS_ARB, &linked);

   if (!linked) {
      getLastGLError();
      return false;
   }

   return true;
}

bool createGLTexture(const char* name, GLuint* handle)
{
	if (handle == NULL || name == NULL) {
		return false;
	}

	int width, height, components;
	stbi_uc* bitmap = stbi_load(name, &width, &height, &components, 3);

	glGenTextures(1, handle);

	glBindTexture(GL_TEXTURE_2D, *handle);

	glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, bitmap);

	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	// free the memory allocated by the image library
	stbi_image_free(bitmap);

	return true;
}


bool createGLCubeMap(const std::string& name, GLuint* handle)
{
	if (handle == NULL) {
		return false;
	}

	const char* postfixes[] = {"+x", "-x", "+y", "-y", "+z", "-z"};

	glGenTextures(1, handle);
	glBindTexture(GL_TEXTURE_CUBE_MAP, *handle);

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 

	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_REPLACE); 

	std::string finalName;

	for (int i=0; i < 6; i++) {

		finalName = name;
		finalName.insert(name.rfind('.'), postfixes[i]);

		int width, height, components;
		stbi_uc* bitmap = stbi_load(finalName.c_str(), &width, &height, &components, 3);

		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 3, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, bitmap);

		// free the memory allocated by the image library
		stbi_image_free(bitmap);
	}

	return true;
}

bool createGLTexture3D(const char* name, GLuint* handle)
{
   if (handle == NULL || name == NULL) {
		return false;
	}

   FILE* f = fopen(name, "rb");

   if (!f) {
      return false;
   }

	int width, height, depth, components;

	fread(&width, sizeof(width), 1, f);
   fread(&height, sizeof(height), 1, f);
   fread(&depth, sizeof(depth), 1, f);
   fread(&components, sizeof(components), 1, f);

   int fullSize = width * height * depth * components;

   unsigned char *volume = new unsigned char[fullSize];

   fread(volume, fullSize, 1, f);

   fclose(f);
   
	glGenTextures(1, handle);		
	glBindTexture(GL_TEXTURE_3D, *handle);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);	
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);	
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, width, height, depth, 0, GL_RGB, GL_UNSIGNED_BYTE, volume);

   // can now deallocate the memory
	delete volume;

	return true;
}

bool createVBO(GLuint* vbo, unsigned int size)
{
   // create buffer object
   glGenBuffers(1, vbo);
   glBindBuffer(GL_ARRAY_BUFFER, *vbo);

   // initialize buffer object
   glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   glutReportErrors();

   return true;
}

void deleteVBO(GLuint* vbo)
{
   glBindBuffer(1, *vbo);
   glDeleteBuffers(1, vbo);
   //cutilSafeCall(cudaGLUnregisterBufferObject(*vbo));

   *vbo = 0;
}

void renderBuffers(GLuint posVBO, GLuint normalVBO, int vertexNumber)
{
   glBindBufferARB(GL_ARRAY_BUFFER_ARB, posVBO);
   glVertexPointer(4, GL_FLOAT, 0, 0);
   glEnableClientState(GL_VERTEX_ARRAY);

   glBindBuffer(GL_ARRAY_BUFFER, normalVBO);   
   glNormalPointer(GL_FLOAT, sizeof(float)*4, 0);
   glEnableClientState(GL_NORMAL_ARRAY);

   //glColor3f(1.0, 0.0, 0.0);
   glDrawArrays(GL_TRIANGLES, 0, vertexNumber);
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);

   glBindBuffer(GL_ARRAY_BUFFER, 0);
}