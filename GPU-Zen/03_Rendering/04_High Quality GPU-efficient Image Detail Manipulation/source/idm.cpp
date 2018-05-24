//
//	idm.cpp
//	Sub-window Variance Filter based Image Decomposition
//
//	Copyright (c) 2016, Kin-Ming Wong and Tien-Tsin Wong
//  All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification,
//	are permitted provided that the following conditions are met:
//
//	Redistributions of source code must retain the above copyright notice,
//	this list of conditions and the following disclaimer.
//
//	Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//	IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//	INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//	BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//	DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
//	OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
//	OF THE POSSIBILITY OF SUCH DAMAGE.
//
//	Please cite our original article
//	'High Quality GPU-efficient Image Detail Manipulation'
//	in GPU Zen if you use any part of the following code.
//

#pragma warning(disable : 4244)		//	Supresses implicit conversion

#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/label.h>
#include <nanogui/button.h>
#include <nanogui/textbox.h>
#include <nanogui/slider.h>
#include <nanogui/imagepanel.h>
#include <nanogui/imageview.h>
#if defined(_WIN32)
#include <windows.h>
#include <glad/glad.h>
#endif
#include <nanogui/opengl.h>
#include <nanogui/glutil.h>
#include <iostream>
#include <iomanip>
#include <string.h>

#include "bmp.h"

using std::cout;
using std::cerr;
using std::endl;


//
//	Global decomposition parameters
//
//	maintained by GUI
//
int		gRadius = 3;
float	gEpsilon = 0.011f;
float	gW1 = 1.0f;
float	gW2 = 1.0f;
char	fileName[512];


//
//  Tracking UI changes
//
bool	decompChanged = true;
bool	manipChanged = true;


//
//	Image input and default buffer size
//
#define TEX_SIZE	1024
#define SCREEN_W	1920
#define SCREEN_H	1080
int		imageWidth, imageHeight;	
int		lx, ux, ly, uy;				//	Image bounds
float	*inputImage;				//	input image, argv(1)


//
//	Dummy vertex array
//
GLuint	dummy_vao;


//
//	Texture buffers
//
//	 [0] : User input image
//	 [1] : Work buffer for SAT
//	 [2] : Work buffer for SAT
//	 [3] : SAT of I2|AK		 
//	 [4] : SAT of I|AK
//	 [5] : PerPatch AK
//	 [6] : PerPatch BK
//	 [7] : Intermediate Base
//	 [8] : Final Base
//	 [9] : Detail_01 (Fine)
//	[10] : Detail_02 (Medium)
//
GLuint texImage[11];


//
//	Shader objects
//
struct shaderObj {
	GLuint	gs;
	GLuint	vs;
	GLuint	fs;
};

GLuint		sqShaderObj;
GLuint		sat2ShaderObj;
GLuint		perPatchShaderObj;
GLuint		decompShaderObj;

shaderObj	orgShaderObj;
shaderObj	elementsShaderObj;
shaderObj	renderShaderObj;


//
//	Shader Program handles
//
GLuint	sqShader;				//	squared shader
GLuint	sat2Shader;				//	SAT (prefixsum) shader
GLuint	perPatchShader;			//	Per-patch Ak,Bk shader
GLuint	decompShader;			//	Decomposition shader

GLuint  orgShader;
GLuint	elementsShader;
GLuint	renderShader;			//	ReComp Render shader


//
//	Shader source file compilation module
//
GLuint compileShader(char *path, GLenum shaderType) {

#define BUFFER_SIZE		512
#define BUFFER_SIZE_1	513

	char buffer[BUFFER_SIZE_1];

	FILE *fp = fopen(path, "r+t");
	if (fp == NULL) {
		printf("unable to open %s", path);
		return -1;
	};

	printf("Compiling %s shader\n", path);

	//	Load texts from shader source file
	//
	std::string		text;
	text.clear();
	memset(buffer, 0, sizeof(char)* BUFFER_SIZE_1);

	size_t	byteRead = 0;
	while ((byteRead = fread(buffer, sizeof(char), BUFFER_SIZE, fp)) == BUFFER_SIZE) {
		buffer[byteRead] = 0;
		text += buffer;
		memset(buffer, 0, sizeof(char)* BUFFER_SIZE_1);
	}
	text += buffer;
	fclose(fp);

	GLchar *sourceText = &text[0];
	const GLchar *source = sourceText;
	GLuint shaderHandle = glCreateShader(shaderType);

	if (shaderHandle == 0) return false;
	glShaderSource(shaderHandle, 1, &source, NULL);
	glCompileShader(shaderHandle);

	GLint status;
	glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &status);
	if (status != GL_TRUE) {
		printf("%s shader compilation failed\n", path);
		return 0;
	}

	printf("%s shader compilation succeeded\n", path);
	return shaderHandle;

}


//
//	Initialization Point
//
bool prepShaders() {

	Bitmap	inputBMP;

	if (!inputBMP.create(fileName)) {
		printf("Unable to read boundary image!\n");
		return false;
	}

	imageWidth = inputBMP.getWidth();
	imageHeight = inputBMP.getHeight();

	inputImage = new float[TEX_SIZE * TEX_SIZE * 4];
	if (!inputImage) {
		printf("Unable to allocate memory!\n");
		return false;
	}

	int offsetX = (TEX_SIZE - imageWidth) / 2;
	int offsetY = (TEX_SIZE - imageHeight) / 2;

	int index = 0;
	for (int y = 0; y < imageHeight; y++) {
		for (int x = 0; x < imageWidth; x++) {
			index = ((TEX_SIZE - 1 - y - offsetY) * TEX_SIZE + x + offsetX) * 4;
			unsigned char r, g, b;
			inputBMP.getColor(x, y, r, g, b);
			inputImage[index] = r / 255.0;
			inputImage[index + 1] = g / 255.0;
			inputImage[index + 2] = b / 255.0;
			inputImage[index + 3] = 1.0;
		}
	}

	lx = offsetX;
	ux = TEX_SIZE - lx;
	ly = offsetY;
	uy = TEX_SIZE - ly;

	glGenTextures(11, texImage);

	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texImage[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, TEX_SIZE, TEX_SIZE, 0, GL_RGBA, GL_FLOAT, inputImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	for (int i = 1; i < 11; i++) {
		glBindTexture(GL_TEXTURE_2D, texImage[i]);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, TEX_SIZE, TEX_SIZE);
	}

	cout << "OpenGL " << GLVersion.major << "." << GLVersion.minor << "\n";
	cout << "\nCompiling Shaders .....\n";

	//	Compile shader sources
	//
	if ((sqShaderObj       = compileShader("./shaders/square.cs.glsl", GL_COMPUTE_SHADER)) == -1) return false;
	if ((sat2ShaderObj     = compileShader("./shaders/prefixSum.cs.glsl", GL_COMPUTE_SHADER)) == -1) return false;
	if ((perPatchShaderObj = compileShader("./shaders/perPatch.cs.glsl", GL_COMPUTE_SHADER)) == -1) return false;
	if ((decompShaderObj   = compileShader("./shaders/decomp.cs.glsl", GL_COMPUTE_SHADER)) == -1) return false;

	if ((orgShaderObj.gs = compileShader("./shaders/original.gs.glsl", GL_GEOMETRY_SHADER)) == -1) return false;
	if ((orgShaderObj.vs = compileShader("./shaders/original.vs.glsl", GL_VERTEX_SHADER)) == -1) return false;
	if ((orgShaderObj.fs = compileShader("./shaders/original.fs.glsl", GL_FRAGMENT_SHADER)) == -1) return false;

	if ((elementsShaderObj.gs = compileShader("./shaders/elements.gs.glsl", GL_GEOMETRY_SHADER)) == -1) return false;
	if ((elementsShaderObj.vs = compileShader("./shaders/elements.vs.glsl", GL_VERTEX_SHADER)) == -1) return false;
	if ((elementsShaderObj.fs = compileShader("./shaders/elements.fs.glsl", GL_FRAGMENT_SHADER)) == -1) return false;

	if ((renderShaderObj.gs = compileShader("./shaders/render.gs.glsl", GL_GEOMETRY_SHADER)) == -1) return false;
	if ((renderShaderObj.vs = compileShader("./shaders/render.vs.glsl", GL_VERTEX_SHADER)) == -1) return false;
	if ((renderShaderObj.fs = compileShader("./shaders/render.fs.glsl", GL_FRAGMENT_SHADER)) == -1) return false;

	cout << "Shader compilation done\n";

	if ((sqShader = glCreateProgram()) == 0) return false;
	glAttachShader(sqShader, sqShaderObj);
	glLinkProgram(sqShader);

	if ((sat2Shader = glCreateProgram()) == 0) return false;
	glAttachShader(sat2Shader, sat2ShaderObj);
	glLinkProgram(sat2Shader);

	if ((perPatchShader = glCreateProgram()) == 0) return false;
	glAttachShader(perPatchShader, perPatchShaderObj);
	glLinkProgram(perPatchShader);

	if ((decompShader = glCreateProgram()) == 0) return false;
	glAttachShader(decompShader, decompShaderObj);
	glLinkProgram(decompShader);

	if ((renderShader = glCreateProgram()) == 0) return false;
	glAttachShader(renderShader, renderShaderObj.gs);
	glAttachShader(renderShader, renderShaderObj.vs);
	glAttachShader(renderShader, renderShaderObj.fs);
	glLinkProgram(renderShader);

	if ((orgShader = glCreateProgram()) == 0) return false;
	glAttachShader(orgShader, orgShaderObj.gs);
	glAttachShader(orgShader, orgShaderObj.vs);
	glAttachShader(orgShader, orgShaderObj.fs);
	glLinkProgram(orgShader);

	if ((elementsShader = glCreateProgram()) == 0) return false;
	glAttachShader(elementsShader, elementsShaderObj.gs);
	glAttachShader(elementsShader, elementsShaderObj.vs);
	glAttachShader(elementsShader, elementsShaderObj.fs);
	glLinkProgram(elementsShader);

	glGenVertexArrays(1, &dummy_vao);

	return true;

}


//	Computer I^2 for each pixel I
//
inline void squared(GLuint input, GLuint output) {

	glUseProgram(sqShader);
	glBindImageTexture(0, input, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1, output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glDispatchCompute(TEX_SIZE, 1, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

}


//	Compute 2 SATs with intermediate work buffers
//
void computeSATs(GLuint input_0, GLuint input_1, GLuint work_0, GLuint work_1, GLuint output_0, GLuint output_1) {

	glUseProgram(sat2Shader);
	//	Pass #1
	glBindImageTexture(0, input_0, 0, GL_FALSE, 0,  GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1, input_1, 0, GL_FALSE, 0,  GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(2,  work_0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindImageTexture(3,  work_1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glDispatchCompute(TEX_SIZE, 1, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	//	Pass #2
	glBindImageTexture(0,   work_0, 0, GL_FALSE, 0,  GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1,   work_1, 0, GL_FALSE, 0,  GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(2, output_0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindImageTexture(3, output_1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glDispatchCompute(TEX_SIZE, 1, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

}


//
//	use perPatch shader, render results to textures Ak and Bk
//
void perPatchSVF(GLuint sqSat, GLuint sat, GLuint AK, GLuint BK)
{

	glUseProgram(perPatchShader);

	glBindImageTexture(0, sqSat, 0, GL_FALSE, 0,  GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1,   sat, 0, GL_FALSE, 0,  GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(2,    AK, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindImageTexture(3,    BK, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glDispatchCompute(TEX_SIZE, 1, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

}


void decomp( GLuint input, GLuint AK, GLuint BK, GLuint base, GLuint detail )
{

	glUseProgram(decompShader);

	glBindImageTexture(0,  input, 0, GL_FALSE, 0,  GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(1,     AK, 0, GL_FALSE, 0,  GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(2,     BK, 0, GL_FALSE, 0,  GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(3,   base, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindImageTexture(4, detail, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glDispatchCompute(TEX_SIZE, 1, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

}


inline void displayOriginal(GLuint org)
{
	
	glViewport(0, 0, SCREEN_W, SCREEN_H);
	glUseProgram(orgShader);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, org);
	glBindVertexArray(dummy_vao);

	glDrawArrays(GL_POINTS, 0, 1);

}


inline void displayResults(GLuint base, GLuint D1, GLuint D2, GLuint shader)
{

	glViewport(0, 0, SCREEN_W, SCREEN_H);
	glUseProgram(shader);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, base);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, D1);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, D2);
	glBindVertexArray(dummy_vao);

	glDrawArrays(GL_POINTS, 0, 1);

}


//
//	Single Decomposition
//
void svDecompProc( GLuint input, GLuint outBase, GLuint outDetail ) {

	//	Square the ORIGINAL
	squared( input, texImage[5] );
	computeSATs(texImage[5], input, texImage[1], texImage[2], texImage[3], texImage[4] );	
	perPatchSVF(texImage[3], texImage[4], texImage[5], texImage[6]);		
	computeSATs(texImage[5], texImage[6], texImage[1], texImage[2], texImage[3], texImage[4]);
	decomp(input, texImage[3], texImage[4], outBase, outDetail);

}


//
//	Display the orginal and results
//
void refreshDisplay() {

	displayOriginal(texImage[0]);
	displayResults(texImage[8], texImage[9], texImage[10], elementsShader);
	displayResults(texImage[8], texImage[9], texImage[10], renderShader);

}


//
//	svDecomp to be called by GUI loop
//
void svDecomp(int radius, float epsilon, float *weight, int layer) {

	GLuint	id;

	if (decompChanged) {

		//	Update shader uniform parameters

		//	perPatchShader (filter radius + epsilon)
		glUseProgram(perPatchShader);
		id = glGetUniformLocation(perPatchShader, "r");
		glUniform1i(id, radius);
		id = glGetUniformLocation(perPatchShader, "epsilon");
		glUniform1f(id, epsilon);

		//	decompShader (filter radius)
		glUseProgram(decompShader);
		id = glGetUniformLocation(decompShader, "r");
		glUniform1i(id, radius);
		id = glGetUniformLocation(decompShader, "lx");
		glUniform1i(id, lx);
		id = glGetUniformLocation(decompShader, "ux");
		glUniform1i(id, ux);
		id = glGetUniformLocation(decompShader, "ly");
		glUniform1i(id, ly);
		id = glGetUniformLocation(decompShader, "uy");
		glUniform1i(id, uy);

		//	First Decomposition
		svDecompProc(texImage[0], texImage[7], texImage[9]);

		//	perPatchShader (filter radius + epsilon)
		glUseProgram(perPatchShader);
		id = glGetUniformLocation(perPatchShader, "r");
		glUniform1i(id, radius * 2);
		id = glGetUniformLocation(perPatchShader, "epsilon");
		glUniform1f(id, epsilon * 1.0f);

		//	decompShader (filter radius)
		glUseProgram(decompShader);
		id = glGetUniformLocation(decompShader, "r");
		glUniform1i(id, radius * 2);

		//	Subsequent Decompositions
		svDecompProc(texImage[7], texImage[8], texImage[10]);

		decompChanged = false;

	}

	//	Render Only (blending weight)
	glUseProgram(renderShader);
	id = glGetUniformLocation(renderShader, "w1");
	glUniform1f(id, weight[0]);
	id = glGetUniformLocation(renderShader, "w2");
	glUniform1f(id, weight[1]);
	manipChanged = false;

	//	Refresh the displays
	refreshDisplay();

}


//
//	Simple idmApp Class for GUI
//
class idmApp : public nanogui::Screen {

public:
	idmApp() : nanogui::Screen(Eigen::Vector2i(SCREEN_W, SCREEN_H), "Real-time Edge-aware Multi-scale Image Manipulation") {

		using namespace nanogui;

		Window* window = new Window(this, "Original image");
		window->setPosition(Vector2i(10, 10));

		window = new Window(this, "Realtime Manipulation");
		window->setPosition(Vector2i(0.5 * SCREEN_W + 10, 10));

		window = new Window(this, "Base Layer");
		window->setPosition(Vector2i(5, 0.67 * SCREEN_H + 5));

		window = new Window(this, "Fine Layer");
		window->setPosition(Vector2i(0.25 * SCREEN_W + 5, 0.67 * SCREEN_H + 5));

		window = new Window(this, "Medium Layer");
		window->setPosition(Vector2i(0.5 * SCREEN_W + 5, 0.67 * SCREEN_H + 5));

		//
		//	Decomposition controls window
		//
		window = new Window(this, "Decomposition controls");
		window->setPosition(Vector2i(SCREEN_W - 350, 0.8 * SCREEN_H));

		GridLayout *layout =
			new GridLayout(Orientation::Horizontal, 3,
				Alignment::Middle, 15, 5);
		layout->setColAlignment(
		{ Alignment::Maximum, Alignment::Fill });
		layout->setSpacing(0, 10);
		window->setLayout(layout);

		//	1ST ROW
		//
		new Label(window, "radius", "sans-bold");

		Slider *slider = new Slider(window);
		slider->setValue(0.2f);
		slider->setFixedWidth(80);

		TextBox *textBox = new TextBox(window);
		textBox->setFixedSize(Vector2i(80, 21));
		textBox->setFontSize(20);
		textBox->setAlignment(TextBox::Alignment::Right);
		textBox->setValue("3");

		slider->setCallback([textBox](float value) {
			gRadius = (int)(value * 10 + 1);
			decompChanged = true;
			textBox->setValue(std::to_string((int)(value * 10 + 1)));
		});

		//	2nd ROW
		//
		new Label(window, "epsilon", "sans-bold");
		slider = new Slider(window);
		slider->setValue(0.1f);
		slider->setFixedWidth(80);

		textBox = new TextBox(window);
		textBox->setFontSize(20);
		textBox->setAlignment(TextBox::Alignment::Right);
		textBox->setFixedSize(Vector2i(80, 21));
		textBox->setValue("0.01100");

		slider->setCallback([textBox](float value) {
			gEpsilon = value * 0.1f + 0.001f;
			decompChanged = true;
			std::stringstream outStr;
			outStr << std::fixed;
			outStr << std::setprecision(5) << (float)(value * 0.1 + 0.001f);
			textBox->setValue(outStr.str());
		});


		//
		//	Manipulation controls window
		//
		window = new Window(this, "Blending control (Detail Layer Weights)");
		window->setPosition(Vector2i(SCREEN_W - 300, 5));

		layout = new GridLayout(Orientation::Horizontal, 3,
				Alignment::Middle, 15, 5);
		layout->setColAlignment(
		{ Alignment::Maximum, Alignment::Fill });
		layout->setSpacing(0, 10);
		window->setLayout(layout);

		//	1ST ROW
		//
		new Label(window, "Fine", "sans-bold");

		Slider *slider1 = new Slider(window);
		slider1->setValue(0.3333f);
		slider1->setFixedWidth(80);

		TextBox *textBox1 = new TextBox(window);
		textBox1->setFixedSize(Vector2i(80, 21));
		textBox1->setFontSize(20);
		textBox1->setAlignment(TextBox::Alignment::Right);
		textBox1->setValue("1.000");

		slider1->setCallback([textBox1](float value) {
			gW1 = value * 3.0f;
			manipChanged = true;
			std::stringstream outStr;
			outStr << std::fixed;
			outStr << std::setprecision(3) << (float)(value * 3.0f);
			textBox1->setValue(outStr.str());
		});

		//	2nd ROW
		//
		new Label(window, "Medium", "sans-bold");
		Slider *slider2 = new Slider(window);
		slider2->setValue(0.3333f);
		slider2->setFixedWidth(80);

		TextBox *textBox2 = new TextBox(window);
		textBox2->setFontSize(20);
		textBox2->setAlignment(TextBox::Alignment::Right);
		textBox2->setFixedSize(Vector2i(80, 21));
		textBox2->setValue("1.000");

		slider2->setCallback([textBox2](float value) {
			gW2 = value * 3.0f;
			manipChanged = true;
			std::stringstream outStr;
			outStr << std::fixed;
			outStr << std::setprecision(3) << (float)(value * 3.0f);
			textBox2->setValue(outStr.str());
		});


		//	3rd ROW
		//
		Button *b = new Button(window, "Zero All");
		b->setCallback([ slider1, slider2, textBox1, textBox2 ] {
			slider1->setValue(0.0f);
			slider2->setValue(0.0f);
			textBox1->setValue("0.000");
			textBox2->setValue("0.000");
			gW1 = gW2 = 0.0f;
			manipChanged = true;
		});

		b = new Button(window, "RESET");
		b->setBackgroundColor(Color(0, 0, 255, 25));
		b->setCallback([ slider1, slider2, textBox1, textBox2 ] {
			slider1->setValue(0.3333f);
			slider2->setValue(0.3333f);
			textBox1->setValue("1.000");
			textBox2->setValue("1.000");
			gW1 = gW2 = 1.0f;
			manipChanged = true;
		});

		performLayout(mNVGContext);


		//
		//	initialize our process and shaders
		//
		if (!prepShaders()) {
			printf("Shader compilation error.\n");
			exit(EXIT_FAILURE);
		};

	}

	~idmApp() {
	}

	virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) {
		if (Screen::keyboardEvent(key, scancode, action, modifiers))
			return true;
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
			setVisible(false);
			return true;
		}
		return false;
	}


	void draw(NVGcontext *ctx) {
		Screen::draw(ctx);
	}


	//
	//	The Main Draw loop here
	//
	void drawContents() {

		using namespace nanogui;

		float	w[2];
		w[0] = gW1;
		w[1] = gW2;

		if (decompChanged || manipChanged) {	//	Decomp or Manip parameters changed

			svDecomp(gRadius, gEpsilon, w, 2);

		} else refreshDisplay();				//	Just Plain Refresh

	}


};



int main(int argc, char ** argv) {

	if (argc != 2) {
		printf("usage: %s <bitmap>\n\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	try {

		strcpy(fileName, argv[1]);
		nanogui::init();

		{
			nanogui::ref<idmApp> app = new idmApp();
			app->drawAll();
			app->setVisible(true);
			nanogui::mainloop();
		}

		nanogui::shutdown();
	}

	catch (const std::runtime_error &e) {
		std::string error_msg = std::string("Caught a fatal error: ") + std::string(e.what());
#if defined(_WIN32)
		MessageBoxA(nullptr, error_msg.c_str(), NULL, MB_ICONERROR | MB_OK);
#else
		std::cerr << error_msg << endl;
#endif
		exit(EXIT_FAILURE);
	}

	return 0;
}
