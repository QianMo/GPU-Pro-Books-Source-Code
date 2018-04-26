// ***********************************************
// *** Screen-Space Directional Occlusion Demo ***
// *** by Thorsten Grosch and Tobias Ritschel  ***
// ***********************************************

#include <glew.h>
#include <stdlib.h> 

#include <glut.h>

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cassert>

#include "Vector.h"
#include "Triangle.h"

using namespace std;


// variables for geometry

vector<int> indices;
vector<Vector> positions;
vector<Vector> normals;
vector<Triangle> triangles;


// variables for camera control 

float center[3]; 

#define PI 3.141592

#define ROTATE 1
#define MOVE 2

float thetaStart = PI / 2.0f - 0.5;
float phiStart = PI / 2.0f;
float rStart = 1.5f;

float theta = thetaStart;
float phi = phiStart;
float r = rStart;

float oldX, oldY;
int motionState;


// Window size

int width = 512;       
int height = 512;

// view parameters

GLfloat viewPosition[4] = {0.0, 5.0, 8.0, 1.0};  
GLfloat viewDirection[4] = {-0.0, -5.0, -8.0, 0.0};  
GLfloat viewAngle = 40.0;
GLfloat viewNear = 0.2;
GLfloat viewFar = 100000.0;

// light parameters

GLfloat lightOrigPosition[4] = {-1258.34, 1109.91, -312.386 ,1};  
GLfloat lightPosition[4] = {-1258.34, 1109.91, -312.386 ,1};  
GLfloat lightDirection[4] = {-1.0, -1.0, -1.0 ,0.0};  



// SSDO variables

// SSDO strength
float strength = 4.0f;

// SSDO singularity for distance
float singularity = 50.0f;

// SSDO depth bias to avoid self-shadowing
float depthBias = 1.0f;

// Strength of the indirect bounce
float bounceStrength = 10700.0f;

// SSDO form factor singularity
float bounceSingularity = 2700.0f;

// hemisphere radius for SSDO samples
float sampleRadius = 50.0f;

// user rotation of the light source
float lightRotationAngle = 0.0f;

// number of samples for SSDO
int sampleCount = 16;

// maximum number samples for SSDO
int maxSampleCount = 128;

// pattern size for interleaved sampling (patternSize x patternSize pixel block)
int patternSize = 4;

// blur filter size 
int kernelSize = 4;

// blur filter position and normal threshold
float positionThreshold = 30.0f;
float normalThreshold = 0.3f;

// maximum radiance for tone mapper
float maxRadiance = 4.5f;


// texture and frambuffer object IDs

GLuint viewerPositionTextureId = 0;
GLuint viewerNormalTextureId = 0;
GLuint viewerColorTextureId = 0;

GLuint directLightTextureId = 0;
GLuint directLightFB = 0;

GLuint ssdoTextureId = 0;
GLuint ssdoFB = 0;

GLuint geometryAwareBlurVerticalTextureId = 0;
GLuint geometryAwareBlurVerticalFB = 0;

GLuint mrtFB = 0;

GLuint sampleDirectionsTextureId = 0;
GLuint blurredEnvMapTextureId = 0;

GLuint shadow_texture_id = 0;



// GLSL related variables

GLuint vertexShaderCreateDirectLightBuffer;	
GLuint fragmentShaderCreateDirectLightBuffer;
GLuint shaderProgramCreateDirectLightBuffer;

GLuint vertexShaderGeometryAwareBlur;	
GLuint fragmentShaderGeometryAwareBlur;
GLuint shaderProgramGeometryAwareBlur;

GLuint vertexShaderGeometryAwareBlurVertical;	
GLuint fragmentShaderGeometryAwareBlurVertical;
GLuint shaderProgramGeometryAwareBlurVertical;

GLuint vertexShaderGeometryAwareBlurHorizontal;	
GLuint fragmentShaderGeometryAwareBlurHorizontal;
GLuint shaderProgramGeometryAwareBlurHorizontal;

GLuint vertexShaderSSDO;	
GLuint fragmentShaderSSDO;
GLuint shaderProgramSSDO;

GLuint vertexShaderMRT;	
GLuint fragmentShaderMRT;
GLuint shaderProgramMRT;


// uniform location variables

GLint lightPositionLocation;              
             
GLint sampleDirectionsTextureLocation;
GLint blurredEnvMapTextureLocation;
GLint sampleAreaLocation;
GLint rotationAngleLocation;
GLint maxLuminanceLocation;

GLint sampleRadiusLocation;
GLint strengthLocation;
GLint singularityLocation;
GLint depthBiasLocation;
GLint bounceStrengthLocation;
GLint bounceSingularityLocation;

GLint positionTextureLocation;
GLint normalTextureLocation;
GLint colorTextureLocation;
GLint directRadianceTextureLocation;

GLint sampleCountLocation;
GLint patternSizeLocation;
GLint envMapTextureLocation;
GLint seedTextureLocation;

GLint modelviewMatrixLocation;
GLint projectionMatrixLocation;

GLint positionThresholdLocation;
GLint normalThresholdLocation;
GLint kernelSizeLocation;
GLint maxRadianceLocation;

GLint positionThresholdLocationVertical;
GLint normalThresholdLocationVertical;
GLint kernelSizeLocationVertical;

GLint positionThresholdLocationHorizontal;
GLint normalThresholdLocationHorizontal;
GLint kernelSizeLocationHorizontal;
GLint maxRadianceLocationHorizontal;

GLint lightRotationAngleLocation;
GLint blurPositionTextureLocation;
GLint blurNormalTextureLocation;
GLint blurColorTextureLocation;
GLint ssdoTextureLocation;
GLint directLightTextureLocation;

GLint blurVerticalPositionTextureLocation;
GLint blurVerticalNormalTextureLocation;
GLint blurVerticalSSDOTextureLocation;

GLint blurHorizontalPositionTextureLocation;
GLint blurHorizontalNormalTextureLocation;
GLint blurHorizontalColorTextureLocation;
GLint blurHorizontalRadianceTextureLocation;
GLint blurHorizontalDirectLightTextureLocation;

GLint directLightPositionTextureLocation;
GLint directLightNormalTextureLocation;   
GLint directLightReflectanceTextureLocation;




// user toggles

bool useSeparatedFilter = true;
bool animate = true;

// timer for quad aniamtion
int timer = 0;

float blockScale = 2.0;


// variables for blurred envmap

#define MAX_LINE 1024 
float *envMapPixels;
int envMapWidth;
int envMapHeight;
float revGamma;
float LMax;
float rotationAngle = 0.0;     // envmap rotation around y axis            



// forward declarations of some functions

void drawScene();
void loadOFF();
void drawGeometryVertexBuffer();
void generateGeometryVertexBuffer();
void calcViewerCamera(float theta, float phi, float r);
void drawGroundPlane();
void drawQuadField(int numQuads, float baseHeight, float size);
void generateSampleDirections();


// load hdr image in PFM format
bool loadPFM( const char* filename )  
{ 
  // init some variables 
  char imageformat[ MAX_LINE ]; 
  float f[1]; 
	 
  // open the file handle  
  FILE* infile = fopen( filename, "rb" ); 
 
  if ( infile == NULL ) { 
    printf("Error loading %s !\n",filename); 
    exit(-1); 
  } 
 
  // read the header  
  fscanf( infile," %s %d %d ", &imageformat, &envMapWidth, &envMapHeight ); 
	 
  // set member variables 
  // assert( width > 0 && height > 0 ); 
  printf("Image format %s Width %d Height %d\n",imageformat, envMapWidth, envMapHeight ); 
   
  envMapPixels = (float*) (malloc(envMapWidth * envMapHeight * 3 * sizeof(float))); 

  // go ahead with the data 
    fscanf( infile,"%f", &f[0] ); 
    fgetc( infile ); 
 
    float red, green, blue; 
     
    float *p = envMapPixels; 
 
    // read the values and store them 
    for ( int j = 0; j < envMapHeight ; j++ )  { 
		for ( int i = 0; i < envMapWidth ; i++ )  { 
	     
			fread( f, 4, 1, infile ); 
			red = f[0]; 
		     
			fread( f, 4, 1, infile ); 
			green = f[0]; 
		     
			fread( f, 4, 1, infile ); 
			blue = f[0]; 
		     
			*p++ = red; 
			*p++ = green; 
			*p++ = blue; 
	 
			float L = (red + green + blue) / 3.0; 
			if (L > LMax) 
			  LMax = L; 
		} 
    } 
    printf("Loading Envmap finished\n"); 

    revGamma = 1.0 / 2.2; 

	return true;
} 




// Print information about the compiling step
void printShaderInfoLog(GLuint shader)
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
void printProgramInfoLog(GLuint program)
{
	GLint infoLogLength = 0;
	GLsizei charsWritten  = 0;
	char *infoLog;

	glGetProgramiv(program, GL_INFO_LOG_LENGTH,&infoLogLength);
	infoLog = (char *)malloc(infoLogLength);
	glGetProgramInfoLog(program, infoLogLength, &charsWritten, infoLog);
	printf("%s\n",infoLog);
	free(infoLog);
}


// Reads a file and returns the content as a string
string readFile(string fileName)
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

void printKeys()
{
	printf(" SSDO demo by Thorsten Grosch and Tobias Ritschel\n");
	printf("--------------------------------------------------\n");
	printf("Use mouse to navigate\n");
	printf("Use left/right arrow keys to rotate light\n");
	printf("Use s/S keys to control directional occlusion strength\n");
	printf("Use i/I keys to control indirect light strength\n");
	printf("Use t/T keys to control tone mapping\n");
	printf("Use k/K keys to control blur filter size\n");
	printf("Use p/P keys to control blur filter position threshold\n");
	printf("Use n/N keys to control blur filter normal threshold\n");
	printf("Use r/R keys to control sample radius\n");
	printf("Use +/- keys to control number of samples\n");
	printf("Use f key to toggle separated blur filter\n");
	printf("Use space key to toggle animation\n");
	printf("--------------------------------------------------\n");
}


void initGL()
{
	// init some GL state variables
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);               
	glEnable(GL_LIGHT0);                 
	glEnable(GL_TEXTURE_2D);

	glViewport(0,0,width,height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    gluPerspective(viewAngle, 1.0f, viewNear, viewFar);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}


// load and compile all shaders
void initGLSL()
{
	string shaderSource;
	const char* sourcePtr;


	// Create multiple render target shader for position/normal/color
	vertexShaderMRT = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "multiple_render_target.vert" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShaderMRT, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShaderMRT);
	printShaderInfoLog(vertexShaderMRT);

	// Create empty shader object (fragment shader)
	fragmentShaderMRT = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "multiple_render_target.frag" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShaderMRT, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShaderMRT);
	printShaderInfoLog(fragmentShaderMRT);

	// Create shader program
	shaderProgramMRT = glCreateProgram();	

	// Attach shader
	glAttachShader(shaderProgramMRT, vertexShaderMRT);
	glAttachShader(shaderProgramMRT, fragmentShaderMRT);

	// Link program
	glLinkProgram(shaderProgramMRT);
	printProgramInfoLog(shaderProgramMRT);

	// glUseProgram( shaderProgramMRT );



	// Create empty shader object (vertex shader)
	vertexShaderCreateDirectLightBuffer = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "create_direct_light_buffer.vert" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShaderCreateDirectLightBuffer, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShaderCreateDirectLightBuffer);
	printShaderInfoLog(vertexShaderCreateDirectLightBuffer);

	// Create empty shader object (fragment shader)
	fragmentShaderCreateDirectLightBuffer = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "create_direct_light_buffer.frag" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShaderCreateDirectLightBuffer, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShaderCreateDirectLightBuffer);
	printShaderInfoLog(fragmentShaderCreateDirectLightBuffer);

	// Create shader program
	shaderProgramCreateDirectLightBuffer = glCreateProgram();	

	// Attach shader
	glAttachShader(shaderProgramCreateDirectLightBuffer, vertexShaderCreateDirectLightBuffer);
	glAttachShader(shaderProgramCreateDirectLightBuffer, fragmentShaderCreateDirectLightBuffer);

	// Link program
	glLinkProgram(shaderProgramCreateDirectLightBuffer);
	printProgramInfoLog(shaderProgramCreateDirectLightBuffer);

	glUseProgram( shaderProgramCreateDirectLightBuffer );

	lightPositionLocation = glGetUniformLocation( shaderProgramCreateDirectLightBuffer, "lightPosition" );
	glUniform4fv(lightPositionLocation, 4, lightPosition);
	
	directLightPositionTextureLocation = glGetUniformLocation( shaderProgramCreateDirectLightBuffer, "positionTexture" );
	glUniform1i(directLightPositionTextureLocation, 0);   
	directLightNormalTextureLocation = glGetUniformLocation( shaderProgramCreateDirectLightBuffer, "normalTexture" );
	glUniform1i(directLightNormalTextureLocation, 1);   
	directLightReflectanceTextureLocation = glGetUniformLocation( shaderProgramCreateDirectLightBuffer, "reflectanceTexture" );
	glUniform1i(directLightReflectanceTextureLocation, 2);   



	// Create empty shader object (vertex shader)
	vertexShaderGeometryAwareBlur = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "geometry_aware_blur.vert" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShaderGeometryAwareBlur, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShaderGeometryAwareBlur);
	printShaderInfoLog(vertexShaderGeometryAwareBlur);

	// Create empty shader object (fragment shader)
	fragmentShaderGeometryAwareBlur = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "geometry_aware_blur.frag" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShaderGeometryAwareBlur, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShaderGeometryAwareBlur);
	printShaderInfoLog(fragmentShaderGeometryAwareBlur);

	// Create shader program
	shaderProgramGeometryAwareBlur = glCreateProgram();	

	// Attach shader
	glAttachShader(shaderProgramGeometryAwareBlur, vertexShaderGeometryAwareBlur);
	glAttachShader(shaderProgramGeometryAwareBlur, fragmentShaderGeometryAwareBlur);

	// Link program
	glLinkProgram(shaderProgramGeometryAwareBlur);
	printProgramInfoLog(shaderProgramGeometryAwareBlur);

	glUseProgram( shaderProgramGeometryAwareBlur );

	positionThresholdLocation = glGetUniformLocation( shaderProgramGeometryAwareBlur, "positionThreshold" );
	glUniform1f(positionThresholdLocation, positionThreshold);
	normalThresholdLocation = glGetUniformLocation( shaderProgramGeometryAwareBlur, "normalThreshold" );
	glUniform1f(normalThresholdLocation, normalThreshold);
	kernelSizeLocation = glGetUniformLocation( shaderProgramGeometryAwareBlur, "kernelSize" );
	glUniform1i(kernelSizeLocation, kernelSize);
	maxRadianceLocation = glGetUniformLocation( shaderProgramGeometryAwareBlur, "maxRadiance" );
	glUniform1f(maxRadianceLocation, maxRadiance);

	blurPositionTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlur, "positionTexture" );
	glUniform1i(blurPositionTextureLocation, 0);   
	blurNormalTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlur, "normalTexture" );
	glUniform1i(blurNormalTextureLocation, 1);   
	blurColorTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlur, "colorTexture" );
	glUniform1i(blurColorTextureLocation, 2);   
	ssdoTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlur, "radianceTexture" );
	glUniform1i(ssdoTextureLocation, 5);   
	directLightTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlur, "directLightTexture" );
	glUniform1i(directLightTextureLocation, 6);   



	// Create empty shader object (vertex shader)
	vertexShaderGeometryAwareBlurVertical = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "geometry_aware_blur_vertical.vert" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShaderGeometryAwareBlurVertical, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShaderGeometryAwareBlurVertical);
	printShaderInfoLog(vertexShaderGeometryAwareBlurVertical);

	// Create empty shader object (fragment shader)
	fragmentShaderGeometryAwareBlurVertical = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "geometry_aware_blur_vertical.frag" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShaderGeometryAwareBlurVertical, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShaderGeometryAwareBlurVertical);
	printShaderInfoLog(fragmentShaderGeometryAwareBlurVertical);

	// Create shader program
	shaderProgramGeometryAwareBlurVertical = glCreateProgram();	

	// Attach shader
	glAttachShader(shaderProgramGeometryAwareBlurVertical, vertexShaderGeometryAwareBlurVertical);
	glAttachShader(shaderProgramGeometryAwareBlurVertical, fragmentShaderGeometryAwareBlurVertical);

	// Link program
	glLinkProgram(shaderProgramGeometryAwareBlurVertical);
	printProgramInfoLog(shaderProgramGeometryAwareBlurVertical);

	glUseProgram( shaderProgramGeometryAwareBlurVertical );

	positionThresholdLocationVertical = glGetUniformLocation( shaderProgramGeometryAwareBlurVertical, "positionThreshold" );
	glUniform1f(positionThresholdLocationVertical, positionThreshold);
	normalThresholdLocationVertical = glGetUniformLocation( shaderProgramGeometryAwareBlurVertical, "normalThreshold" );
	glUniform1f(normalThresholdLocationVertical, normalThreshold);
	kernelSizeLocationVertical = glGetUniformLocation( shaderProgramGeometryAwareBlurVertical, "kernelSize" );
	glUniform1i(kernelSizeLocationVertical, kernelSize);

	blurVerticalPositionTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlurVertical, "positionTexture" );
	glUniform1i(blurVerticalPositionTextureLocation, 0);   
	blurVerticalNormalTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlurVertical, "normalTexture" );
	glUniform1i(blurVerticalNormalTextureLocation, 1);   
	ssdoTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlurVertical, "radianceTexture" );
	glUniform1i(ssdoTextureLocation, 5);   



	// Create empty shader object (vertex shader)
	vertexShaderGeometryAwareBlurHorizontal = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "geometry_aware_blur_horizontal.vert" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShaderGeometryAwareBlurHorizontal, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShaderGeometryAwareBlurHorizontal);
	printShaderInfoLog(vertexShaderGeometryAwareBlurHorizontal);

	// Create empty shader object (fragment shader)
	fragmentShaderGeometryAwareBlurHorizontal = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "geometry_aware_blur_horizontal.frag" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShaderGeometryAwareBlurHorizontal, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShaderGeometryAwareBlurHorizontal);
	printShaderInfoLog(fragmentShaderGeometryAwareBlurHorizontal);

	// Create shader program
	shaderProgramGeometryAwareBlurHorizontal = glCreateProgram();	

	// Attach shader
	glAttachShader(shaderProgramGeometryAwareBlurHorizontal, vertexShaderGeometryAwareBlurHorizontal);
	glAttachShader(shaderProgramGeometryAwareBlurHorizontal, fragmentShaderGeometryAwareBlurHorizontal);

	// Link program
	glLinkProgram(shaderProgramGeometryAwareBlurHorizontal);
	printProgramInfoLog(shaderProgramGeometryAwareBlurHorizontal);

	glUseProgram( shaderProgramGeometryAwareBlurHorizontal );

	positionThresholdLocationHorizontal = glGetUniformLocation( shaderProgramGeometryAwareBlurHorizontal, "positionThreshold" );
	glUniform1f(positionThresholdLocationHorizontal, positionThreshold);
	normalThresholdLocationHorizontal = glGetUniformLocation( shaderProgramGeometryAwareBlurHorizontal, "normalThreshold" );
	glUniform1f(normalThresholdLocationHorizontal, normalThreshold);
	kernelSizeLocationHorizontal = glGetUniformLocation( shaderProgramGeometryAwareBlurHorizontal, "kernelSize" );
	glUniform1i(kernelSizeLocationHorizontal, kernelSize);
	maxRadianceLocationHorizontal = glGetUniformLocation( shaderProgramGeometryAwareBlurHorizontal, "maxRadiance" );
	glUniform1f(maxRadianceLocationHorizontal, maxRadiance);

	blurHorizontalPositionTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlurHorizontal, "positionTexture" );
	glUniform1i(blurHorizontalPositionTextureLocation, 0);   
	blurHorizontalNormalTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlurHorizontal, "normalTexture" );
	glUniform1i(blurHorizontalNormalTextureLocation, 1);   
	blurHorizontalColorTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlurHorizontal, "colorTexture" );
	glUniform1i(blurHorizontalColorTextureLocation, 2);   
	blurHorizontalRadianceTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlurHorizontal, "radianceTexture" );
	glUniform1i(blurHorizontalRadianceTextureLocation, 7);   
	blurHorizontalDirectLightTextureLocation = glGetUniformLocation( shaderProgramGeometryAwareBlurHorizontal, "directLightTexture" );
	glUniform1i(blurHorizontalDirectLightTextureLocation, 6);   




	// Create empty shader object (vertex shader)
	vertexShaderSSDO = glCreateShader(GL_VERTEX_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "SSDO.vert" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(vertexShaderSSDO, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(vertexShaderSSDO);
	printShaderInfoLog(vertexShaderSSDO);

	// Create empty shader object (fragment shader)
	fragmentShaderSSDO = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile( "SSDO.frag" );
	sourcePtr = shaderSource.c_str();

	// Attach shader code
	glShaderSource(fragmentShaderSSDO, 1, &sourcePtr, NULL);	

	// Compile
	glCompileShader(fragmentShaderSSDO);
	printShaderInfoLog(fragmentShaderSSDO);

	// Create shader program
	shaderProgramSSDO = glCreateProgram();	

	// Attach shader
	glAttachShader(shaderProgramSSDO, vertexShaderSSDO);
	glAttachShader(shaderProgramSSDO, fragmentShaderSSDO);

	// Link program
	glLinkProgram(shaderProgramSSDO);
	printProgramInfoLog(shaderProgramSSDO);

	glUseProgram( shaderProgramSSDO );

	sampleRadiusLocation = glGetUniformLocation( shaderProgramSSDO, "sampleRadius" );
	glUniform1f(sampleRadiusLocation, sampleRadius);
	strengthLocation = glGetUniformLocation( shaderProgramSSDO, "strength" );
	glUniform1f(strengthLocation, strength);
	singularityLocation = glGetUniformLocation( shaderProgramSSDO, "singularity" );
	glUniform1f(singularityLocation, singularity);
	depthBiasLocation = glGetUniformLocation( shaderProgramSSDO, "depthBias" );
	glUniform1f(depthBiasLocation, depthBias);
	bounceStrengthLocation = glGetUniformLocation( shaderProgramSSDO, "bounceStrength" );
	glUniform1f(bounceStrengthLocation, bounceStrength);
	bounceSingularityLocation = glGetUniformLocation( shaderProgramSSDO, "bounceSingularity" );
	glUniform1f(bounceSingularityLocation, bounceSingularity);
	lightRotationAngleLocation = glGetUniformLocation( shaderProgramSSDO, "lightRotationAngle" );
	glUniform1f(lightRotationAngleLocation, lightRotationAngle);

	sampleCountLocation = glGetUniformLocation( shaderProgramSSDO, "sampleCount" );
	glUniform1i(sampleCountLocation, sampleCount);
	patternSizeLocation = glGetUniformLocation( shaderProgramSSDO, "patternSize" );
	glUniform1i(patternSizeLocation, sampleCount);

	seedTextureLocation = glGetUniformLocation( shaderProgramSSDO, "seedTexture" );
	glUniform1i(seedTextureLocation, 3);         
	envMapTextureLocation = glGetUniformLocation( shaderProgramSSDO, "envmapTexture" );
	glUniform1i(envMapTextureLocation, 4);        


	positionTextureLocation = glGetUniformLocation( shaderProgramSSDO, "positionTexture" );
	glUniform1i(positionTextureLocation, 0);         
	normalTextureLocation = glGetUniformLocation( shaderProgramSSDO, "normalTexture" );
	glUniform1i(normalTextureLocation, 1);         
	colorTextureLocation = glGetUniformLocation( shaderProgramSSDO, "colorTexture" );
	glUniform1i(colorTextureLocation, 2);        
	directRadianceTextureLocation = glGetUniformLocation( shaderProgramSSDO, "directRadianceTexture" );
	glUniform1i(directRadianceTextureLocation, 6);        


	calcViewerCamera(theta, phi, r);

	GLfloat modelviewValues[16];
	GLfloat projectionValues[16];

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(viewAngle, 1.0f, viewNear, viewFar);	
				
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(viewPosition[0], viewPosition[1], viewPosition[2],
				viewPosition[0] + viewDirection[0], viewPosition[1] + viewDirection[1], viewPosition[2] + viewDirection[2], 
				0, 1, 0);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelviewValues);
	glGetFloatv(GL_PROJECTION_MATRIX, projectionValues);


	modelviewMatrixLocation = glGetUniformLocation( shaderProgramSSDO, "modelviewMatrix" );
	glUniformMatrix4fv(modelviewMatrixLocation, 1, false, modelviewValues);
	projectionMatrixLocation = glGetUniformLocation( shaderProgramSSDO, "projectionMatrix" );
	glUniformMatrix4fv(projectionMatrixLocation, 1, false, projectionValues);

	printProgramInfoLog(shaderProgramSSDO);

	glUseProgram( 0 );

	printKeys();

}


// calc the view position and direction from theta/phi coordinates
void calcViewerCamera(float theta, float phi, float r)
{
    float x = r * sin(theta) * cos(phi);
    float y = r * cos(theta);
    float z = r * sin(theta) * sin(phi);
 
	viewPosition[0] = center[0] + x;
	viewPosition[1] = center[1] + y;
	viewPosition[2] = center[2] + z;
	viewDirection[0] = -x;
	viewDirection[1] = -y;
	viewDirection[2] = -z;
}



// draw the ground plane
void drawGroundPlane()
{
	float size = 2500;
	float y = -360;

	glNormal3f(0,1,0);
	glBegin(GL_QUADS);
	glVertex3f(-size + center[0], y + center[1], size + center[2]); 
	glVertex3f( size + center[0], y + center[1], size + center[2]); 
	glVertex3f( size + center[0], y + center[1], -size + center[2]); 
	glVertex3f(-size + center[0], y + center[1], -size + center[2]); 
	glEnd();

}

// draw a screen filling quad (for fragment programs) 
void drawScreenFillingQuad()
{
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glColor3f(1,1,1);

	glBegin(GL_QUADS);

	glTexCoord2f(0, 0);
	glVertex2f(-1, -1);
	glTexCoord2f(1, 0);
	glVertex2f( 1, -1);
	glTexCoord2f(1, 1);
	glVertex2f( 1,  1);
	glTexCoord2f(0, 1);
	glVertex2f(-1,  1);

	glEnd();

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
}

// draw a quad on the groud
void drawQuad(Vector pos, float size, float height)
{
	Vector p1(pos[0]-size, pos[1], pos[2]+size);
	Vector p2(pos[0]+size, pos[1], pos[2]+size);
	Vector p3(pos[0]+size, pos[1], pos[2]-size);
	Vector p4(pos[0]-size, pos[1], pos[2]-size);
	Vector p5(pos[0]-size, pos[1]+height, pos[2]+size);
	Vector p6(pos[0]+size, pos[1]+height, pos[2]+size);
	Vector p7(pos[0]+size, pos[1]+height, pos[2]-size);
	Vector p8(pos[0]-size, pos[1]+height, pos[2]-size);


	glNormal3f(0,0,1);
	glVertex3f(p1[0], p1[1], p1[2]);
	glVertex3f(p2[0], p2[1], p2[2]);
	glVertex3f(p6[0], p6[1], p6[2]);
	glVertex3f(p5[0], p5[1], p5[2]);

	glNormal3f(1,0,0);
	glVertex3f(p2[0], p2[1], p2[2]);
	glVertex3f(p3[0], p3[1], p3[2]);
	glVertex3f(p7[0], p7[1], p7[2]);
	glVertex3f(p6[0], p6[1], p6[2]);

	glNormal3f(0,0,-1);
	glVertex3f(p4[0], p4[1], p4[2]);
	glVertex3f(p8[0], p8[1], p8[2]);
	glVertex3f(p7[0], p7[1], p7[2]);
	glVertex3f(p3[0], p3[1], p3[2]);

	glNormal3f(-1,0,0);
	glVertex3f(p1[0], p1[1], p1[2]);
	glVertex3f(p5[0], p5[1], p5[2]);
	glVertex3f(p8[0], p8[1], p8[2]);
	glVertex3f(p4[0], p4[1], p4[2]);

	glNormal3f(0,1,0);
	glVertex3f(p5[0], p5[1], p5[2]);
	glVertex3f(p6[0], p6[1], p6[2]);
	glVertex3f(p7[0], p7[1], p7[2]);
	glVertex3f(p8[0], p8[1], p8[2]);

}

// draw a field of colored quads
void drawQuadField(int numQuads, float baseHeight, float size)
{
	float stepSize = 2*size / numQuads;

	glBegin(GL_QUADS);

	for (int i = 0 ; i < numQuads ; i++) {

		for (int j = 0 ; j < numQuads ; j++) {
	
			Vector pos(center[0] + i*stepSize - size, baseHeight + center[1], center[2] + j*stepSize - size);
			float blockSize = blockScale/0.1; 
			float blockHeight = 1.2 * blockScale * (i + j) * abs(sin((i + j)/3.0 + timer/15.0)) + 0.1;

			glColor3f(abs(sin(float(i))), abs(sin(float(j))), abs(sin(float(i+j))));
			GLfloat diffuseCol[3] = {abs(sin(float(i))), abs(sin(float(j))), abs(sin(float(i+j)))};
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseCol);
			drawQuad(pos, blockSize, blockHeight);
		}
	}

	glEnd();

}

// draw the whole scene (neptune, quads and ground plane)
void drawScene()
{
	drawGroundPlane();
	drawQuadField(20, -360, 2500);
	glColor3f(1.0, 1.0, 1.0);
	drawGeometryVertexBuffer();
}

void display()
{
	int timeStart = glutGet(GLUT_ELAPSED_TIME);

	// clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	calcViewerCamera(theta, phi, r);

	// now render from camera view
	glLoadIdentity();
	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
   
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(viewAngle, 1.0f, viewNear, viewFar);	
				
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(viewPosition[0], viewPosition[1], viewPosition[2],
			  viewPosition[0] + viewDirection[0], viewPosition[1] + viewDirection[1], viewPosition[2] + viewDirection[2], 
			  0, 1, 0);


	// activate shaders for the creation of position/normal/color buffer from viewer
	glUseProgram( shaderProgramMRT );

	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, mrtFB);      // activate fbo

	glClearColor(0.0, 0.0, 0.0, 0.0);                       // clear with alpha = 0 for undefined pixels
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawScene();

	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);       // deactivate fbo


	// activate shaders for creation of direct light buffer (from view position)
	glUseProgram( shaderProgramCreateDirectLightBuffer);

	glUniform4f(lightPositionLocation, lightPosition[0], lightPosition[1], lightPosition[2], 1.0);

	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, directLightFB);      // activate fbo

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawScreenFillingQuad();

	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);       // deactivate fbo


	// start SSDO 

	calcViewerCamera(theta, phi, r);

	GLfloat modelviewValues[16];
	GLfloat projectionValues[16];

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(viewAngle, 1.0f, viewNear, viewFar);	
			
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(viewPosition[0], viewPosition[1], viewPosition[2],
			viewPosition[0] + viewDirection[0], viewPosition[1] + viewDirection[1], viewPosition[2] + viewDirection[2], 
			0, 1, 0);

	// remember modelview & projection matrix values for deferred shading later
	glGetFloatv(GL_MODELVIEW_MATRIX, modelviewValues);
	glGetFloatv(GL_PROJECTION_MATRIX, projectionValues);


	glUseProgram( shaderProgramSSDO );

	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, ssdoFB);      // activate fbo

	glUniform1f(sampleRadiusLocation, sampleRadius);
	glUniform1f(strengthLocation, strength);
	glUniform1f(singularityLocation, singularity);
	glUniform1f(depthBiasLocation, depthBias);
	glUniform1f(bounceStrengthLocation, bounceStrength);
	glUniform1f(bounceSingularityLocation, bounceSingularity);
	glUniform1f(lightRotationAngleLocation, lightRotationAngle);

	glUniform1i(sampleCountLocation, sampleCount);
	glUniform1i(patternSizeLocation, patternSize);

	glUniformMatrix4fv(modelviewMatrixLocation, 1, false, modelviewValues);
	glUniformMatrix4fv(projectionMatrixLocation, 1, false, projectionValues);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawScreenFillingQuad();

	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);       // deactivate fbo
	glUseProgram( 0 );



	if (!useSeparatedFilter) {

		// non-separated geometry aware blur

		glUseProgram( shaderProgramGeometryAwareBlur );

		glUniform1f(positionThresholdLocation, positionThreshold);
		glUniform1f(normalThresholdLocation, normalThreshold);
		glUniform1i(kernelSizeLocation, kernelSize);
		glUniform1f(maxRadianceLocation, maxRadiance);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();

		glUseProgram( 0 );
	}
	else {

		// separated geometry-aware blur
		// do a vertical blur first

		glUseProgram( shaderProgramGeometryAwareBlurVertical );

		glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, geometryAwareBlurVerticalFB);      // activate fbo

		glUniform1f(positionThresholdLocationVertical, positionThreshold);
		glUniform1f(normalThresholdLocationVertical, normalThreshold);
		glUniform1i(kernelSizeLocationVertical, kernelSize);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();

		glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);       // deactivate fbo
		glUseProgram( 0 );


		// do a horizontal blur (with the vertical blur as input)

		glUseProgram( shaderProgramGeometryAwareBlurHorizontal );

		glUniform1f(positionThresholdLocationHorizontal, positionThreshold);
		glUniform1f(normalThresholdLocationHorizontal, normalThreshold);
		glUniform1i(kernelSizeLocationHorizontal, kernelSize);
		glUniform1f(maxRadianceLocationHorizontal, maxRadiance);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();

		glUseProgram( 0 );
	}

	// swap display buffers
	glutSwapBuffers();
	glFinish();

	// measure frame time in milliseconds
	int timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Delay %d     \r",timeEnd - timeStart);

	if (animate)
		timer++;
}

void special(int key, int x, int y)
{
		switch (key) {                                     
		case GLUT_KEY_UP :
			break;
		case GLUT_KEY_DOWN :
			break;
		case GLUT_KEY_LEFT :
			lightRotationAngle -= 0.1;
			if (lightRotationAngle < 0) lightRotationAngle += 2*PI;

			lightPosition[0] = lightOrigPosition[0] * cos(-lightRotationAngle) - lightOrigPosition[2] * sin(-lightRotationAngle);
			lightPosition[2] = lightOrigPosition[0] * sin(-lightRotationAngle) + lightOrigPosition[2] * cos(-lightRotationAngle);

			break;

		case GLUT_KEY_RIGHT :
			lightRotationAngle += 0.1;
			if (lightRotationAngle > 2*PI) lightRotationAngle -= 2*PI;

			lightPosition[0] = lightOrigPosition[0] * cos(-lightRotationAngle) - lightOrigPosition[2] * sin(-lightRotationAngle);
			lightPosition[2] = lightOrigPosition[0] * sin(-lightRotationAngle) + lightOrigPosition[2] * cos(-lightRotationAngle);

			break;

		case GLUT_KEY_PAGE_UP :
			break;
		case GLUT_KEY_PAGE_DOWN :
			break;
		}
}

void keyboard(unsigned char key, int x, int y)
{
	// set parameters
	switch (key) 
	{                                 
		case 'p':
			positionThreshold += 1.0;
			break;
		case 'P':
			if (positionThreshold > 1.5)
				positionThreshold -= 1.0;
			break;
		case 'n':
			normalThreshold += 0.01;
			break;
		case 'N':
			if (normalThreshold > 0.015)
				normalThreshold -= 0.01;
			break;
		case 'i':
			bounceStrength *= 1.2;
			break;
		case 'I':
			bounceStrength /= 1.2;
			break;
		case 'f':
			useSeparatedFilter = !useSeparatedFilter;
			break;
		case 't':
			maxRadiance *= 1.2;
			break;
		case 'T':
			maxRadiance /= 1.2;
			break;
		case 'd':
			depthBias *= 1.2;
			break;
		case 'D':
			depthBias /= 1.2;
			break;
		case 'r':
			sampleRadius *= 1.1;
			break;
		case 'R':
			sampleRadius /= 1.1;
			break;
		case 's':
			strength *= 1.2;
			break;
		case 'S':
			strength /= 1.2;
			break;
		case 'k':
			kernelSize++;
			break;
		case 'K':
			if (kernelSize > 0) kernelSize--;
			break;
		case '+':
			if (sampleCount < maxSampleCount)
				sampleCount++;
			break;
		case '-':
			if (sampleCount > 1) 
				sampleCount--;
			break;
		case ' ':
			animate = !animate;
			break;
	}
	printf("SSDO Strength %f\nIndirect Light Strength %f\nNum Samples %d\nKernel Size %d\nPosition Threshold %f\nNormal Threshold %f\n",strength,bounceStrength,sampleCount,kernelSize,positionThreshold,normalThreshold);
	if (useSeparatedFilter)
		printf("Separated Blur Filter ON\n");
	else
		printf("Separated Blur Filter OFF\n");
	printf("\n");

	printKeys();
}

// use a virtual trackball as mouse control
void mouseMotion(int x, int y)
{
	float deltaX = x - oldX;
	float deltaY = y - oldY;
	
	if (motionState == ROTATE) {
		theta -= 0.001 * deltaY;

		if (theta < 0.001) theta = 0.001;
		else if (theta > PI - 0.001) theta = PI - 0.001;

		phi += 0.001 * deltaX;	
		if (phi < 0) phi += 2*PI;
		else if (phi > 2*PI) phi -= 2*PI;
	}
	else if (motionState == MOVE) {
		r += 1.0 * deltaY;
	}

	oldX = x;
	oldY = y;

	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
	oldX = x;
	oldY = y;

	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			motionState = ROTATE;
		}
	}
	else if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) {
			motionState = MOVE;
		}
	}
}


int main(int argc, char** argv)
{
	// Initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(width, height);

	glutCreateWindow("Screen-Space Directional Occlusion");


	// Init glew so that the GLSL functionality will be available
	if(glewInit() != GLEW_OK)
	   cout << "GLEW init failed!" << endl;

	// Check for GLSL availability
	if(!GLEW_VERSION_1_5)
	   cout << "OpenGL 1.5 not supported!" << endl;

	// verify that FBOs are supported 
	if (!glutExtensionSupported ("GL_EXT_framebuffer_object") )
	{
		cout << "FBO extension unsupported" << endl;
		exit (1);
	}
	else {
		// cout << "found FBO extension" << endl;
	}

	// create depth buffer 
	glGenTextures (1, &shadow_texture_id);
	glBindTexture (GL_TEXTURE_2D, shadow_texture_id);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


	// load blurred envmap
	loadPFM("KitchenMediumBlurred.pfm");

	// texture for blurred envmap
	glGenTextures (1, &blurredEnvMapTextureId);
	glBindTexture (GL_TEXTURE_2D, blurredEnvMapTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB32F_ARB, envMapWidth, envMapHeight, 0, GL_RGB, GL_FLOAT, envMapPixels);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);


	// texture for viewer position
	glGenTextures (1, &viewerPositionTextureId);
	glBindTexture (GL_TEXTURE_2D, viewerPositionTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// texture for viewer normal
	glGenTextures (1, &viewerNormalTextureId);
	glBindTexture (GL_TEXTURE_2D, viewerNormalTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// texture for viewer color
	glGenTextures (1, &viewerColorTextureId);
	glBindTexture (GL_TEXTURE_2D, viewerColorTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);



	// texture & FBO for direct light
	glGenTextures (1, &directLightTextureId);
	glBindTexture (GL_TEXTURE_2D, directLightTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenFramebuffersEXT (1, &directLightFB);
	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, directLightFB);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, directLightTextureId, 0);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, shadow_texture_id, 0);

	// texture & FBO for SSDO
	glGenTextures (1, &ssdoTextureId);
	glBindTexture (GL_TEXTURE_2D, ssdoTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenFramebuffersEXT (1, &ssdoFB);
	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, ssdoFB);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, ssdoTextureId, 0);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, shadow_texture_id, 0);


	// texture & FBO for vertical blur
	glGenTextures (1, &geometryAwareBlurVerticalTextureId);
	glBindTexture (GL_TEXTURE_2D, geometryAwareBlurVerticalTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenFramebuffersEXT (1, &geometryAwareBlurVerticalFB);
	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, geometryAwareBlurVerticalFB);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, geometryAwareBlurVerticalTextureId, 0);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, shadow_texture_id, 0);


	// MRT FBO for position/normal/color
	glGenFramebuffersEXT (1, &mrtFB);
	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, mrtFB);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, viewerPositionTextureId, 0);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, viewerNormalTextureId, 0);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_2D, viewerColorTextureId, 0);
	glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, shadow_texture_id, 0);

	GLenum buffers[3] = {GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT};
	glDrawBuffers(3, buffers);
	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);



	generateSampleDirections();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture (GL_TEXTURE_2D, viewerPositionTextureId); // texture 0 is the position buffer
	glActiveTexture(GL_TEXTURE1);
	glBindTexture (GL_TEXTURE_2D, viewerNormalTextureId); // texture 1 is the normal buffer
	glActiveTexture(GL_TEXTURE2);
	glBindTexture (GL_TEXTURE_2D, viewerColorTextureId); // texture 2 is the color buffer
	glActiveTexture(GL_TEXTURE3);
	glBindTexture (GL_TEXTURE_2D, sampleDirectionsTextureId); // texture 3 contains the random numbers 
	glActiveTexture(GL_TEXTURE4); 
	glBindTexture (GL_TEXTURE_2D, blurredEnvMapTextureId); // texture 4 is the blurred envmap
	glActiveTexture(GL_TEXTURE5);
	glBindTexture (GL_TEXTURE_2D, ssdoTextureId); // texture 5 contains the SSDO result
	glActiveTexture(GL_TEXTURE6);
	glBindTexture (GL_TEXTURE_2D, directLightTextureId); // texture 6 is for the direct light
	glActiveTexture(GL_TEXTURE7);
	glBindTexture (GL_TEXTURE_2D, geometryAwareBlurVerticalTextureId); // texture 7 is for vertical blurring


	// check framebuffer status
	GLenum status = glCheckFramebufferStatusEXT (GL_FRAMEBUFFER_EXT);
	switch (status)
	{
	case GL_FRAMEBUFFER_COMPLETE_EXT:
		// cout << "FBO complete" << endl;
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
		cout << "FBO configuration unsupported" << endl;
		return 1;
	default:
		cout << "FBO programmer error" << endl;
		return 1;
	}
	glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

	// load neptune model
	loadOFF();

	// OpenGL initialization
	initGL();
	
	// GLSL initialization
	initGLSL();

	// Register GLUT callback functions   
	glutDisplayFunc(display);
	glutIdleFunc(display);     
	glutSpecialFunc(special);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);

	// we draw the geometry as a vertex buffer
	generateGeometryVertexBuffer();

	// Enter main loop
	glutMainLoop();

	return 0;
}

// generate a halton number
float halton(const int base, int index) {
	float x = 0.0f;
	float f = 1.0f / base;

	while(index) {
		x += f * (float) (index % base);
		index /= base;
		f *= 1.0f / base;
	}
	return x;
}

// generate random positions inside a unit hemisphere based on halton numbers
void generateSampleDirections()
{
	int patternSizeSquared = patternSize * patternSize;

	srand(0);

	int haltonIndex = 0;
	float* const seedPixels = new float[3 * maxSampleCount * patternSizeSquared];
	
	for(int i = 0; i < patternSizeSquared; i++) {
		for(int j = 0; j < maxSampleCount; j++) {

			Vector sample;
			do {
				sample = Vector(
							2.0f * halton(2, haltonIndex) - 1.0f, 
							2.0f * halton(3, haltonIndex) - 1.0f, 
							halton(5, haltonIndex));						
				haltonIndex++;

				// printf("sample dir %f %f %f\n",sample[0],sample[1],sample[2]);
			} while(sample.length() > 1.0);

			seedPixels[(i * maxSampleCount + j) * 3 + 0] = sample[0];
			seedPixels[(i * maxSampleCount + j) * 3 + 1] = sample[1];
			seedPixels[(i * maxSampleCount + j) * 3 + 2] = sample[2];
		}
	}
	
	glGenTextures (1, &sampleDirectionsTextureId);
	glBindTexture (GL_TEXTURE_2D, sampleDirectionsTextureId);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB32F_ARB, maxSampleCount, patternSizeSquared, 0, GL_RGB, GL_FLOAT, seedPixels); 
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
			
	delete seedPixels;
}

// load on OFF model (neptune)
void loadOFF()
{
	char line[1024], keyword[128]; 
	int dummy;

	float pos[3];
	int index[3];

	int numPoints;
	int numTris;

	FILE *fp = fopen("804_neptune_400ktriangles_uniform.off","r");

	if (fp == NULL) {
		printf("*** OFF file not found! ***\n");
		return;
	}

	fgets(line , 256, fp);
	sscanf(line, "%s \n", keyword);
	fgets(line , 256, fp);
	sscanf(line, "%d %d %d\n", &numPoints, &numTris, &dummy);
	printf("Loading %s model with %d points and %d triangles... \n",keyword,numPoints,numTris);


	center[0] = center[1] = center[2] = 0.0f;

	for (int i = 0 ; i < numPoints ; i++) {

		fgets(line , 256, fp);
		sscanf(line, "%f %f %f\n", &pos[0], &pos[1], &pos[2]);
		positions.push_back(Vector(pos[0], pos[1], pos[2]));

		center[0] += pos[0];
		center[1] += pos[1];
		center[2] += pos[2];
	}

	center[0] /= numPoints;
	center[1] /= numPoints;
	center[2] /= numPoints;

	float maxDistSqr = 0;
	for (int i = 0 ; i < numPoints ; i++) {
		float distX = positions[i][0] - center[0];
		float distY = positions[i][1] - center[1];
		float distZ = positions[i][2] - center[2];
		float distSqr = distX*distX + distY*distY + distZ*distZ;
		if (distSqr > maxDistSqr)
			maxDistSqr = distSqr;
	}
	r = 2.0*sqrt(maxDistSqr);
	
	normals.resize(positions.size());

	for (int i = 0 ; i < numTris ; i++) {

		fgets(line , 256, fp);
		sscanf(line, "%d %d %d %d\n", &dummy, &index[0], &index[1], &index[2]);
		// printf("Index %d %d %d\n",index[0], index[1], index[2]);
		
		Triangle tri(positions[index[0]], positions[index[1]], positions[index[2]]);

	    triangles.push_back(tri);

		indices.push_back(index[0]);
		indices.push_back(index[1]);
		indices.push_back(index[2]);

		Vector normal = tri.getNormal();

		normals[index[0]] = normal;    // our model does not have vertex normals so we use the face normal
		normals[index[1]] = normal;
		normals[index[2]] = normal;
	}

	fclose(fp);

	printf("finished\n");

}

GLuint positionBufferHandle;
GLuint normalBufferHandle;
GLuint indexBufferHandle;
void generateGeometryVertexBuffer()
{	
	glGenBuffersARB(1, &positionBufferHandle); 
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, positionBufferHandle);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, positions.size() * 3 * sizeof(float), &positions[0], GL_STATIC_DRAW);
	glVertexPointer(3, GL_FLOAT, 0, NULL);
	
	glGenBuffersARB(1, &normalBufferHandle); 
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalBufferHandle);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, normals.size() * 3 * sizeof(float), &normals[0], GL_STATIC_DRAW);
	glNormalPointer(GL_FLOAT, 0, NULL);
	
	glGenBuffersARB(1, &indexBufferHandle); 
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indexBufferHandle);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indices.size() * sizeof(int), &indices[0], GL_STATIC_DRAW);
}


void drawGeometryVertexBuffer()
{
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, positionBufferHandle);
	glVertexPointer(3, GL_FLOAT, 0, NULL);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalBufferHandle);
	glNormalPointer(GL_FLOAT, 0, NULL);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	glDrawElements(GL_TRIANGLES, 3 * triangles.size(), GL_UNSIGNED_INT, NULL);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
}