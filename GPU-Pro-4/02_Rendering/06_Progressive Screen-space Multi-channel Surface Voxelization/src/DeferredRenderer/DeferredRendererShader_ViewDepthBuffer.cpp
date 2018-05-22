#include "shaders/DeferredRendererShader_ViewDepthBuffer.h"
#include "DeferredRenderer.h"
#include "SceneGraph.h"

DRShaderViewDepthBuffer::DRShaderViewDepthBuffer()
{
	initialized = false;
}

void DRShaderViewDepthBuffer::start()
{
	class Camera3D *camera = renderer->getSceneRoot()->getActiveCamera();

	DRShader::start();

	shader->setUniform1f(0,camera->getNear(),uniform_viewdepthbuffer_zNear);
	shader->setUniform1f(0,camera->getFar(),uniform_viewdepthbuffer_zFar);

	glEnable(GL_TEXTURE_2D); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, renderer->getBuffer(DR_TARGET_DEPTH));
	shader->setUniform1i(0,0,uniform_viewdepthbuffer_buffer);
}

void DRShaderViewDepthBuffer::stop()
{
	glBindTexture(GL_TEXTURE_2D,0);
	glDisable(GL_TEXTURE_2D);

	DRShader::stop();
}

bool DRShaderViewDepthBuffer::init(class DeferredRenderer* _renderer)
{
	if (initialized)
		return true;
	
	if (!DRShader::init(_renderer))
		return false;

	char * shader_text_vert = DRSH_ViewDepthBuffer_Vertex;
	char * shader_text_frag = DRSH_ViewDepthBuffer_Fragment;

	shader = shader_manager.loadfromMemory ("Common Vertex", shader_text_vert, "View Depth Buffer", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling view depth buffer shader.");
		return false;
	}
	else
	{
	    uniform_viewdepthbuffer_buffer	= shader->GetUniformLocation("buffer");
	    uniform_viewdepthbuffer_zNear	= shader->GetUniformLocation("zNear");
	    uniform_viewdepthbuffer_zFar	= shader->GetUniformLocation("zFar");
	}
	
	initialized = true;
	return true;
}

//----------------- Shader text ----------------------------

#define VIEWDEPTHBUFFER_3

#ifdef VIEWDEPTHBUFFER_1

char DRShaderViewDepthBuffer::DRSH_ViewDepthBuffer_Vertex[] = "\n\
\n\
#version 330 compatibility \n\
\n\
void main (void) \n\
{ \n\
	gl_Position = ftransform (); \n\
    gl_TexCoord[0] = gl_MultiTexCoord0; \n\
}";

char DRShaderViewDepthBuffer::DRSH_ViewDepthBuffer_Fragment[] = "\n\
#version 330 compatibility \n\
\n\
uniform sampler2D buffer; \n\
uniform float zNear, zFar; \n\
\n\
void main (void) \n\
{ \n\
	vec2 uv = gl_TexCoord[0].xy; \n\
	float depth = texture (buffer, uv).x; \n\
\n\
	// http://wiki.gamedev.net/index.php/D3DBook:Depth_of_Field \n\
	float Q = zFar / (zFar - zNear); \n\
	float d = (-zNear * Q) / (depth - Q); \n\
\n\
	gl_FragColor = vec4 (d, d, d, 1.0); \n\
\n\
#if 1 \n\
	// visualize linearity of the depth buffer \n\
		 if (d < 0.0) gl_FragColor = vec4 (0.0,0.0,0.0,1.0); \n\
	else if (d < 0.1) gl_FragColor = vec4 (1.0,0.0,0.0,1.0); \n\
	else if (d < 0.2) gl_FragColor = vec4 (0.0,1.0,0.0,1.0); \n\
	else if (d < 0.3) gl_FragColor = vec4 (0.0,0.0,1.0,1.0); \n\
	else if (d < 0.4) gl_FragColor = vec4 (1.0,0.5,0.0,1.0); \n\
	else if (d < 0.5) gl_FragColor = vec4 (0.5,1.0,0.0,1.0); \n\
	else if (d < 0.6) gl_FragColor = vec4 (0.5,0.0,1.0,1.0); \n\
	else if (d < 0.7) gl_FragColor = vec4 (1.0,0.0,0.5,1.0); \n\
	else if (d < 0.8) gl_FragColor = vec4 (0.0,1.0,0.5,1.0); \n\
	else if (d < 0.9) gl_FragColor = vec4 (0.0,0.5,1.0,1.0); \n\
	else			  gl_FragColor = vec4 (1.0,1.0,1.0,1.0); \n\
#endif \n\
}";

#endif

#ifdef VIEWDEPTHBUFFER_2

char DRShaderViewDepthBuffer::DRSH_ViewDepthBuffer_Vertex[] = "\n\
\n\
#version 330 compatibility \n\
\n\
uniform float zNear, zFar; \n\
\n\
void main (void) \n\
{ \n\
	// http://www.mvps.org/directx/articles/linear_z/linearz.htm \n\
    vec4 position = gl_ModelViewProjectionMatrix * gl_Vertex; \n\
	position.z *= position.w / zFar; \n\
	gl_Position = position; \n\
\n\
    gl_TexCoord[0] = gl_MultiTexCoord0; \n\
}";

char DRShaderViewDepthBuffer::DRSH_ViewDepthBuffer_Fragment[] = "\n\
#version 330 compatibility \n\
\n\
uniform sampler2D buffer; \n\
\n\
void main (void) \n\
{ \n\
	vec2 uv = gl_TexCoord[0].xy; \n\
	float d = texture (buffer, uv).x; \n\
\n\
	gl_FragColor = vec4 (d, d, d, 1.0); \n\
\n\
#if 1 \n\
	// visualize linearity of the depth buffer \n\
		 if (d < 0.0) gl_FragColor = vec4 (0.0,0.0,0.0,1.0); \n\
	else if (d < 0.1) gl_FragColor = vec4 (1.0,0.0,0.0,1.0); \n\
	else if (d < 0.2) gl_FragColor = vec4 (0.0,1.0,0.0,1.0); \n\
	else if (d < 0.3) gl_FragColor = vec4 (0.0,0.0,1.0,1.0); \n\
	else if (d < 0.4) gl_FragColor = vec4 (1.0,0.5,0.0,1.0); \n\
	else if (d < 0.5) gl_FragColor = vec4 (0.5,1.0,0.0,1.0); \n\
	else if (d < 0.6) gl_FragColor = vec4 (0.5,0.0,1.0,1.0); \n\
	else if (d < 0.7) gl_FragColor = vec4 (1.0,0.0,0.5,1.0); \n\
	else if (d < 0.8) gl_FragColor = vec4 (0.0,1.0,0.5,1.0); \n\
	else if (d < 0.9) gl_FragColor = vec4 (0.0,0.5,1.0,1.0); \n\
	else			  gl_FragColor = vec4 (1.0,1.0,1.0,1.0); \n\
#endif \n\
}";

#endif

#ifdef VIEWDEPTHBUFFER_3

char DRShaderViewDepthBuffer::DRSH_ViewDepthBuffer_Vertex[] = "\n\
\n\
#version 330 compatibility \n\
\n\
void main (void) \n\
{ \n\
	gl_Position = ftransform (); \n\
    gl_TexCoord[0] = gl_MultiTexCoord0; \n\
}";

char DRShaderViewDepthBuffer::DRSH_ViewDepthBuffer_Fragment[] = "\n\
\n\
#version 330 compatibility \n\
\n\
uniform sampler2D buffer; \n\
uniform float zNear, zFar; \n\
\n\
#define MAP_0TO1_MINUS1TO1(_value)	(2.0 * (_value) - 1.0)	// Map [0.0,1.0] to [-1.0,1.0] \n\
\n\
float linearizeDepth (sampler2D depthSampler, vec2 uv) \n\
{ \n\
	// http://www.geeks3d.com/20091216/geexlab-how-to-visualize-the-depth-buffer-in-glsl/ \n\
	// http://olivers.posterous.com/linear-depth-in-glsl-for-real \n\
	float z_depth = texture (depthSampler, uv).x; \n\
//	float z_ndc = MAP_0TO1_MINUS1TO1 (z_depth);		// screen space --> norm clip space \n\
//	float z_eye = (2.0 * zFar * zNear) / (zFar + zNear - z_ndc * (zFar - zNear));	// [0,zFar] \n\
\n\
	// http://www.humus.name/temp/Linearize%20depth.txt \n\
//	return (z_eye - zNear) / (zFar - zNear); // which is expanded to: \n\
//	return zNear * (z_ndc + 1.0) / (zFar + zNear - z_ndc * (zFar - zNear)); \n\
\n\
	return zNear * z_depth / (zFar - z_depth * (zFar - zNear)); \n\
} \n\
\n\
#define BETWEEN(_x, _lo, _hi) ((_x) >= (_lo) && ((_x) <= (_hi))) \n\
\n\
void main(void) \n\
{ \n\
	float d; \n\
	vec2 uv = gl_TexCoord[0].xy; \n\
\n\
#ifdef VIEWDEPTHBUFFER_HALF_HALF \n\
	// display half linearized and half as-is depth buffer \n\
	if (uv.x < 0.5) \n\
	{ \n\
		d = linearizeDepth (buffer, uv); \n\
	} \n\
	else \n\
	{ \n\
	//	uv.x -= 0.5;	// when enabled will display the same portion as the other half \n\
		d = texture (buffer, uv).x; \n\
	} \n\
#else \n\
	d = linearizeDepth (buffer, uv); \n\
//	d = texture (buffer, uv).x; \n\
#endif \n\
\n\
#ifdef VIEWDEPTHBUFFER_HALF_HALF \n\
	// make the left-right separation a red line \n\
	if (BETWEEN (uv.x, 0.5 - 0.001, 0.5 + 0.001)) gl_FragColor = vec4 (1.0, 0.0, 0.0, 1.0); \n\
#endif \n\
\n\
#if 0 \n\
	// visualize linearity of the depth buffer \n\
		 if (d < 0.0) gl_FragColor = vec4 (0.0,0.0,0.0,1.0); \n\
	else if (d < 0.1) gl_FragColor = vec4 (1.0,0.0,0.0,1.0); // r \n\
	else if (d < 0.2) gl_FragColor = vec4 (0.0,1.0,0.0,1.0); // g \n\
	else if (d < 0.3) gl_FragColor = vec4 (0.0,0.0,1.0,1.0); // b \n\
	else if (d < 0.4) gl_FragColor = vec4 (1.0,0.5,0.0,1.0); \n\
	else if (d < 0.5) gl_FragColor = vec4 (0.5,1.0,0.0,1.0); \n\
	else if (d < 0.6) gl_FragColor = vec4 (0.5,0.0,1.0,1.0); \n\
	else if (d < 0.7) gl_FragColor = vec4 (1.0,0.0,0.5,1.0); \n\
	else if (d < 0.8) gl_FragColor = vec4 (0.0,1.0,0.5,1.0); \n\
	else if (d < 0.9) gl_FragColor = vec4 (0.0,0.5,1.0,1.0); \n\
	else			  gl_FragColor = vec4 (1.0,1.0,1.0,1.0); \n\
#endif \n\
	gl_FragColor = vec4 (d,d,d,1.0); \n\
	//gl_FragColor = vec4 (vec3(1,1,1)*(d+max(sign(cos(6.128*1000*d)),0)), 1.0); \n\
}";

#endif
