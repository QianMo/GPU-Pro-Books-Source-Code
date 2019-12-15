#include "tlingandblendingdemo.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GL/freeglut.h>

/* ======================= Forward Declarations ============================ */
void RenderFrame();
void RenderFrameClassic(bool splitScreenMode);
void RenderFrameByExample(bool splitScreenMode);
void KeyboardEventHandling(unsigned char c, int x, int y);
void KeyboardSpecialEventHandling(int c, int x, int y);
void MouseEventHandling(int button, int state, int x, int y);
void MouseDragHandling(int x, int y);
void Idle();
GLuint CreateGLTextureFromTextureDataStruct(const TextureDataFloat& im, GLenum wrapMode, bool generateMips, bool compress);
int ReadPPMtoTextureData(string path, TextureDataFloat& out);
GLuint CompileShader(const char* vertexFile, const char* fragmentFile);
unsigned long GetFileLength(ifstream& file);
int LoadShader(const char* filename, GLchar** shaderSource, GLint* len);

/* ========================== Global Variables ============================= */
const int WINDOW_WIDTH = 1920;
const int WINDOW_HEIGHT = 1080;
GLuint TilingRepeatShader;
GLuint TilingAndBlendingShader;
GLuint input_textureID;
GLuint Tinput_textureID;
GLuint invT_textureID;
vec2 textureScale = vec2(1.0f, 1.0f);
vec2 textureTranslate = vec2(0.0f, 0.0f);
int renderingMode = 2;
vec2 previousMousePosition = vec2(0.0f, 0.0f);

// Uniform variables to pass to the fragment shader
vec3 colorSpaceVector1, colorSpaceVector2, colorSpaceVector3;
vec3 colorSpaceOrigin;
vec3 DXTScalers; // COMPRESSION FIX

/* =============================== MAIN ==================================== */
int main(int argc, char** argv)
{
	// Ask user for input texture file name
	string inputSampleFile = "";
	cout << "Enter input sample file name (seamless, power of 2, ppm format):" << endl;
	cin >> inputSampleFile;
	if (inputSampleFile.find('.') == string::npos)
		inputSampleFile.append(".ppm");
	inputSampleFile.insert(0, "Textures/");

	// Read data from input texture sample
	TextureDataFloat input;
	if (ReadPPMtoTextureData(inputSampleFile, input) == -1)
	{
		cin.get();
		cin.get();
		exit(0);
	}
	else
	{
		cout << endl;
		cout << "Controls:" << endl;
		cout << "F1: display repeated input" << endl;
		cout << "F2: display infinite by-example output" << endl;
		cout << "F3: dual view" << endl << endl;
	}

	// Initialize GLUT, Glew and window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(200, 200);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("Procedural Stochastic Textures by Tiling and Blending");
	glutDisplayFunc(RenderFrame);
	glutKeyboardFunc(KeyboardEventHandling);
	glutSpecialFunc(KeyboardSpecialEventHandling);
	glutMouseFunc(MouseEventHandling);
	glutMotionFunc(MouseDragHandling);
	glutIdleFunc(Idle);
	GLenum err = glewInit();

	// Initialize textures
	input_textureID = CreateGLTextureFromTextureDataStruct(input, GL_REPEAT, true, USE_DXT_COMPRESSION);

	// preprocess
	TextureDataFloat Tinput;
	TextureDataFloat lut;
	Precomputations(input, Tinput, lut, colorSpaceVector1, colorSpaceVector2, colorSpaceVector3, colorSpaceOrigin);

	// If we use DXT compression
	// we need to rescale the Gaussian channels (see Section 1.6)
	if (USE_DXT_COMPRESSION)
	{
		DXTScalers.x = 1.0f / sqrtf(colorSpaceVector1.x*colorSpaceVector1.x + colorSpaceVector1.y*colorSpaceVector1.y + colorSpaceVector1.z*colorSpaceVector1.z);
		DXTScalers.y = 1.0f / sqrtf(colorSpaceVector2.x*colorSpaceVector2.x + colorSpaceVector2.y*colorSpaceVector2.y + colorSpaceVector2.z*colorSpaceVector2.z);
		DXTScalers.z = 1.0f / sqrtf(colorSpaceVector3.x*colorSpaceVector3.x + colorSpaceVector3.y*colorSpaceVector3.y + colorSpaceVector3.z*colorSpaceVector3.z);

		for (int y = 0; y < Tinput.height; y++)
		for (int x = 0; x < Tinput.width; x++)
		{
			float v1 = Tinput.GetPixel(x, y, 0);
			v1 = (v1 - 0.5f) / DXTScalers.x + 0.5f;
			Tinput.SetPixel(x, y, 0, v1);

			float v2 = Tinput.GetPixel(x, y, 1);
			v2 = (v2 - 0.5f) / DXTScalers.y + 0.5f;
			Tinput.SetPixel(x, y, 1, v2);

			float v3 = Tinput.GetPixel(x, y, 2);
			v3 = (v3 - 0.5f) / DXTScalers.z + 0.5f;
			Tinput.SetPixel(x, y, 2, v3);
		}
	}
	else
	{
		DXTScalers.x = -1.0f;
		DXTScalers.y = -1.0f;
		DXTScalers.z = -1.0f;
	}

	// Create the openGL textures
	Tinput_textureID = CreateGLTextureFromTextureDataStruct(Tinput, GL_REPEAT, true, USE_DXT_COMPRESSION);
	invT_textureID = CreateGLTextureFromTextureDataStruct(lut, GL_CLAMP_TO_EDGE, false, USE_DXT_COMPRESSION);
	
	// Initialize shaders
	TilingRepeatShader = CompileShader("Shaders/tiling_repeat.vert",
		"Shaders/tiling_repeat.frag");
	TilingAndBlendingShader = CompileShader("Shaders/tiling_and_blending.vert",
		"Shaders/tiling_and_blending.frag");

	// Go into GLUT main loop
	glutMainLoop();
	return 0;
}

/* ============================ GLUT Events ================================ */
void RenderFrame()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (renderingMode == 0)
		RenderFrameClassic(false);
	else if (renderingMode == 1)
		RenderFrameByExample(false);
	else
	{
		RenderFrameByExample(true);
		RenderFrameClassic(true);
	}

	glutSwapBuffers();
}

void RenderFrameClassic(bool splitScreenMode)
{
	// Bind shader
	glUseProgram(TilingRepeatShader);

	// Bind uniforms
	glUniform1f(glGetUniformLocation(TilingRepeatShader, "_SplitScreenMode"),
		splitScreenMode ? 1.0f : 0.0f);
	glUniform1f(glGetUniformLocation(TilingRepeatShader, "_AspectRatio"),
		glutGet(GLUT_WINDOW_WIDTH) / (float)glutGet(GLUT_WINDOW_HEIGHT));
	
	glUniform4f(glGetUniformLocation(TilingRepeatShader, "_ScaleTranslate"),
		textureScale.x, textureScale.y, textureTranslate.x, textureTranslate.y);
	int id = glGetUniformLocation(TilingRepeatShader, "input");
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, input_textureID);
	glUniform1i(id, 0);

	// Draw
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

void RenderFrameByExample(bool splitScreenMode)
{
	// Bind shader
	glUseProgram(TilingAndBlendingShader);

	// Bind uniforms
	glUniform1f(glGetUniformLocation(TilingAndBlendingShader, "_HalfScreenMode"),
		splitScreenMode ? 1.0f : 0.0f);
	glUniform1f(glGetUniformLocation(TilingAndBlendingShader, "_AspectRatio"),
		glutGet(GLUT_WINDOW_WIDTH) / (float)glutGet(GLUT_WINDOW_HEIGHT));

	glUniform4f(glGetUniformLocation(TilingAndBlendingShader, "_ScaleTranslate"),
		textureScale.x, textureScale.y, textureTranslate.x, textureTranslate.y);

	int id = glGetUniformLocation(TilingAndBlendingShader, "Tinput");
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Tinput_textureID);
	glUniform1i(id, 0);
	id = glGetUniformLocation(TilingAndBlendingShader, "invT");
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, invT_textureID);
	glUniform1i(id, 1);

	glUniform3f(glGetUniformLocation(TilingAndBlendingShader, "_colorSpaceVector1"),
		colorSpaceVector1.x, colorSpaceVector1.y, colorSpaceVector1.z);
	glUniform3f(glGetUniformLocation(TilingAndBlendingShader, "_colorSpaceVector2"),
		colorSpaceVector2.x, colorSpaceVector2.y, colorSpaceVector2.z);
	glUniform3f(glGetUniformLocation(TilingAndBlendingShader, "_colorSpaceVector3"),
		colorSpaceVector3.x, colorSpaceVector3.y, colorSpaceVector3.z);
	glUniform3f(glGetUniformLocation(TilingAndBlendingShader, "_colorSpaceOrigin"),
		colorSpaceOrigin.x, colorSpaceOrigin.y, colorSpaceOrigin.z);
	glUniform3f(glGetUniformLocation(TilingAndBlendingShader, "_DXTScalers"),
		DXTScalers.x, DXTScalers.y, DXTScalers.z);

	// Draw
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

void KeyboardEventHandling(unsigned char c, int x, int y)
{
	if (c == 27)
	{
		exit(0);
	}
}

void KeyboardSpecialEventHandling(int c, int x, int y)
{
	// Switch rendering mode
	if (c == GLUT_KEY_F1)
	{
		renderingMode = 0;
	}
	else if (c == GLUT_KEY_F2)
	{
		renderingMode = 1;
	}
	else if (c == GLUT_KEY_F3)
	{
		renderingMode = 2;
	}
	RenderFrame();
}

void MouseEventHandling(int button, int state, int x, int y)
{
	// Drag start
	if (button == GLUT_LEFT_BUTTON
		|| button == GLUT_MIDDLE_BUTTON
		|| button == GLUT_RIGHT_BUTTON)
	{
		previousMousePosition = vec2(x, y);
	}

	// Zoom
	if (button == 3)
	{
		textureScale *= 0.9f;
	}
	else if (button == 4)
	{
		textureScale *= 1.1f;
	}
	RenderFrame();
}

void MouseDragHandling(int x, int y)
{
	vec2 currentMousePos = vec2(x, y);
	textureTranslate -= (currentMousePos - previousMousePosition)
		* (textureScale.x * 0.002f);
	previousMousePosition = currentMousePos;
	RenderFrame();
}

void Idle()
{
  glutPostRedisplay();
}

/* ========================================================================= */


/* ========================== Texture Handling ============================= */
GLuint CreateGLTextureFromTextureDataStruct(const TextureDataFloat& im, GLenum wrapMode, bool generateMips, bool DXTcompress=false)
{
	if (im.data.empty())
		return 0;

	GLuint texture;
	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
		generateMips ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapMode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapMode);

	glTexImage2D(GL_TEXTURE_2D, 0, DXTcompress ? GL_COMPRESSED_RGB_S3TC_DXT1_EXT : GL_RGB8, im.width, im.height, 0,
		GL_RGB, GL_FLOAT, im.data.data());

	if (generateMips)
		glGenerateMipmap(GL_TEXTURE_2D);
	return texture;
}

int ReadPPMtoTextureData(string path, TextureDataFloat& out)
{
	filebuf fb;
	if (!fb.open(path, ios::in | ios::binary))
	{
		cerr << "Couldn't open ppm file !" << endl;
		return -1;
	}
	istream in(&fb);

	// Only RGB format supported
	out.channels = 3;

	// Header parsing while ignoring comment lines
	// Read magic number
	string s;
	in >> s;
	while (s.front() == '#')
	{
		in.ignore(numeric_limits<streamsize>::max(), '\n');
		in >> s;
	}
	string magicNumber = s;

	// Read width and height
	in >> s;
	while (s.front() == '#')
	{
		in.ignore(numeric_limits<streamsize>::max(), '\n');
		in >> s;
	}
	out.width = stoi(s);
	in >> s;
	out.height = stoi(s);

	// Read max value
	in >> s;
	while (s.front() == '#')
	{
		in.ignore(numeric_limits<streamsize>::max(), '\n');
		in >> s;
	}
	string maxValue = s;

	// handle error cases
	if (maxValue != string("255"))
	{
		cerr << "PPM file's max value needs to be 255" << endl;
		return -1;
	}
	if (magicNumber != string("P6"))
	{
		cerr << "PPM file's magic number needs to be P6" << endl;
		return -1;
	}

	// Pixel data reading
	in.ignore(numeric_limits<streamsize>::max(), '\n');
	vector<unsigned char> temp;
	temp.resize((out.width) * (out.height) * 3);
	in.read(reinterpret_cast<char*>(temp.data()), temp.size());

	// Store in floating point structure
	out.data.resize((out.width) * (out.height) * 3);
	for (unsigned int i = 0; i < temp.size(); i++)
		out.data[i] = temp[i] / 255.0f;

	if (!in)
	{
		cerr << "Error reading PPM file" << endl;
		return -1;
	}

	return 0;
}
/* ========================================================================= */


/* =========================== Shader Compiler ============================= */
GLuint CompileShader(const char* vertexFile, const char* fragmentFile)
{
	// Read vertex and fragment shaders source
	GLint vertexLen = 0;
	GLchar* vertexSource;
	LoadShader(vertexFile, &vertexSource, &vertexLen);
	GLint fragmentLen = 0;
	GLchar* fragmentSource;
	LoadShader(fragmentFile, &fragmentSource, &fragmentLen);

	// Create and compile vertex and fragment shaders
	GLuint vertexObject, fragmentObject;
	vertexObject = glCreateShader(GL_VERTEX_SHADER);
	fragmentObject = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(vertexObject, 1, &vertexSource, &vertexLen);
	glShaderSource(fragmentObject, 1, &fragmentSource, &fragmentLen);
	glCompileShader(vertexObject);
	glCompileShader(fragmentObject);

	// Check compilation status and log errors
	GLint compiled;
	glGetShaderiv(vertexObject, GL_COMPILE_STATUS, &compiled);
	if (!compiled)
	{
		GLint blen = 0;
		GLsizei slen = 0;
		glGetShaderiv(vertexObject, GL_INFO_LOG_LENGTH, &blen);
		if (blen > 1)
		{
			GLchar* compiler_log = (GLchar*)malloc(blen);
			glGetInfoLogARB(vertexObject, blen, &slen, compiler_log);
			cout << "Vertex compiler Log: " << compiler_log;
			free(compiler_log);
		}
	}
	glGetShaderiv(fragmentObject, GL_COMPILE_STATUS, &compiled);
	if (!compiled)
	{
		GLint blen = 0;
		GLsizei slen = 0;
		glGetShaderiv(fragmentObject, GL_INFO_LOG_LENGTH, &blen);
		if (blen > 1)
		{
			GLchar* compiler_log = (GLchar*)malloc(blen);
			glGetInfoLogARB(fragmentObject, blen, &slen, compiler_log);
			cout << "Fragment compiler Log: " << compiler_log;
			free(compiler_log);
		}
	}

	// Link vertex and fragment shaders
	GLuint res = glCreateProgram();
	glAttachShader(res, vertexObject);
	glAttachShader(res, fragmentObject);
	glLinkProgram(res);
	GLint linked;
	glGetProgramiv(res, GL_LINK_STATUS, &linked);
	if (!linked)
	{
		GLint blen = 0;
		GLsizei slen = 0;
		glGetShaderiv(fragmentObject, GL_INFO_LOG_LENGTH, &blen);
		if (blen > 1)
		{
			GLchar* compiler_log = (GLchar*)malloc(blen);
			glGetShaderInfoLog(fragmentObject, blen, &slen, compiler_log);
			cout << "Linking Log: " << compiler_log;
			free(compiler_log);
		}
	}

	// Free sources
	delete[] vertexSource;
	delete[] fragmentSource;

	return res;
}

unsigned long GetFileLength(ifstream& file)
{
	if (!file.good()) return 0;

	unsigned long pos = (unsigned long)file.tellg();
	file.seekg(0, ios::end);
	unsigned long len = (unsigned long)file.tellg();
	file.seekg(ios::beg);

	return len;
}

int LoadShader(const char* filename, GLchar** shaderSource, GLint* len)
{
	ifstream file;
	file.open(filename, ios::in);
	if (!file)
		return -1;

	*len = GetFileLength(file);
	if (len == 0)
		return -2;

	*shaderSource = (GLchar*)new char[*len + 1];
	if (*shaderSource == 0)
		return -3;

	(*shaderSource)[*len] = 0;
	unsigned int i = 0;
	while (file.good())
	{
		(*shaderSource)[i] = file.get();
		if (!file.eof())
			i++;
	}

	(*shaderSource)[i] = 0;
	file.close();
	return 0;
}
/* ========================================================================= */

