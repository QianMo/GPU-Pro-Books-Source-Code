
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "OpenGLRenderer.h"

typedef GLvoid (APIENTRY *UNIFORM_FUNC)(GLint location, GLsizei count, const void *value);
typedef GLvoid (APIENTRY *UNIFORM_MAT_FUNC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);

struct Texture {
	union {
		GLuint glTexID;
		GLuint glDepthID;
	};
	GLuint glTarget;
	FORMAT format;
	uint flags;
	int width, height;
	float lod;

	SamplerStateID samplerState;
};

ConstantType getConstantType(GLenum type){
	switch (type){
		case GL_FLOAT:          return CONSTANT_FLOAT;
		case GL_FLOAT_VEC2_ARB: return CONSTANT_VEC2;
		case GL_FLOAT_VEC3_ARB: return CONSTANT_VEC3;
		case GL_FLOAT_VEC4_ARB: return CONSTANT_VEC4;
		case GL_INT:            return CONSTANT_INT;
		case GL_INT_VEC2_ARB:   return CONSTANT_IVEC2;
		case GL_INT_VEC3_ARB:   return CONSTANT_IVEC3;
		case GL_INT_VEC4_ARB:   return CONSTANT_IVEC4;
		case GL_BOOL_ARB:       return CONSTANT_BOOL;
		case GL_BOOL_VEC2_ARB:  return CONSTANT_BVEC2;
		case GL_BOOL_VEC3_ARB:  return CONSTANT_BVEC3;
		case GL_BOOL_VEC4_ARB:  return CONSTANT_BVEC4;
		case GL_FLOAT_MAT2_ARB: return CONSTANT_MAT2;
		case GL_FLOAT_MAT3_ARB: return CONSTANT_MAT3;
		case GL_FLOAT_MAT4_ARB: return CONSTANT_MAT4;
	}

	return (ConstantType) -1;
}

struct Constant {
	char *name;
	ubyte *data;
	uint index;
	ConstantType type;
	int nElements;
	bool dirty;
};

int constantComp(const void *s0, const void *s1){
	return strcmp(((Constant *) s0)->name, ((Constant *) s1)->name);
}

struct Sampler {
	char *name;
	uint index;
};

int samplerComp(const void *s0, const void *s1){
	return strcmp(((Sampler *) s0)->name, ((Sampler *) s1)->name);
}


struct Shader {
	GLhandleARB program;
	GLhandleARB vertexShader;
	GLhandleARB fragmentShader;

	Constant *uniforms;
	Sampler  *samplers;

	uint nUniforms;
	uint nSamplers;
};

struct Attrib {
	int stream;
	int size;
	AttributeFormat format;
	int offset;
};

#define MAX_GENERIC 8
#define MAX_TEXCOORD 8

struct VertexFormat {
	Attrib generic[MAX_GENERIC];
	Attrib texCoord[MAX_TEXCOORD];
	Attrib vertex;
	Attrib normal;

	int vertexSize[MAX_VERTEXSTREAM];
	int maxGeneric;
	int maxTexCoord;
};

struct VertexBuffer {
	GLuint vboVB;
	GLsizei size;
};

struct IndexBuffer {
	GLuint vboIB;

	uint nIndices;
	uint indexSize;
};

struct SamplerState {
	GLint minFilter;
	GLint magFilter;
	GLint wrapS;
	GLint wrapT;
	GLint wrapR;
	GLint aniso;
	float lod;
};

struct BlendState {
	GLenum srcFactorRGB;
	GLenum dstFactorRGB;
	GLenum srcFactorAlpha;
	GLenum dstFactorAlpha;
	GLenum blendModeRGB;
	GLenum blendModeAlpha;
	int mask;
	bool blendEnable;
	bool alphaToCoverageEnable;
};

struct DepthState {
	int depthFunc;
	bool depthTest;
	bool depthWrite;
//	bool stencilTest;
//	uint8 stencilMask;
//	int stencilFunc;
//	int stencilFail;
//	int depthFail;
//	int stencilPass;
};

struct RasterizerState {
	int cullMode;
	int fillMode;
	bool multiSample;
	bool scissor;
};

// Blending constants
const int ZERO                 = GL_ZERO;
const int ONE                  = GL_ONE;
const int SRC_COLOR            = GL_SRC_COLOR;
const int ONE_MINUS_SRC_COLOR  = GL_ONE_MINUS_SRC_COLOR;
const int DST_COLOR            = GL_DST_COLOR;
const int ONE_MINUS_DST_COLOR  = GL_ONE_MINUS_DST_COLOR;
const int SRC_ALPHA            = GL_SRC_ALPHA;
const int ONE_MINUS_SRC_ALPHA  = GL_ONE_MINUS_SRC_ALPHA;
const int DST_ALPHA            = GL_DST_ALPHA;
const int ONE_MINUS_DST_ALPHA  = GL_ONE_MINUS_DST_ALPHA;
const int SRC_ALPHA_SATURATE   = GL_SRC_ALPHA_SATURATE;

const int BM_ADD              = GL_FUNC_ADD_EXT;
const int BM_SUBTRACT         = GL_FUNC_SUBTRACT_EXT;
const int BM_REVERSE_SUBTRACT = GL_FUNC_REVERSE_SUBTRACT_EXT;
const int BM_MIN              = GL_MIN_EXT;
const int BM_MAX              = GL_MAX_EXT;

// Depth-test constants
const int NEVER    = GL_NEVER;
const int LESS     = GL_LESS;
const int EQUAL    = GL_EQUAL;
const int LEQUAL   = GL_LEQUAL;
const int GREATER  = GL_GREATER;
const int NOTEQUAL = GL_NOTEQUAL;
const int GEQUAL   = GL_GEQUAL;
const int ALWAYS   = GL_ALWAYS;

// Stencil-test constants
const int KEEP     = GL_KEEP;
const int SET_ZERO = GL_ZERO;
const int REPLACE  = GL_REPLACE;
const int INVERT   = GL_INVERT;
const int INCR     = GL_INCR_WRAP;
const int DECR     = GL_DECR_WRAP;
const int INCR_SAT = GL_INCR;
const int DECR_SAT = GL_DECR;

// Culling constants
const int CULL_NONE  = 0;
const int CULL_BACK  = GL_BACK;
const int CULL_FRONT = GL_FRONT;

// Fillmode constants
const int SOLID = GL_FILL;
const int WIREFRAME = GL_LINE;

void setGLdefaults(){
	// Set some of my preferred defaults
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glFrontFace(GL_CW);
	glPixelStorei(GL_PACK_ALIGNMENT,   1);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

#if defined(_WIN32)

OpenGLRenderer::OpenGLRenderer(HDC hDc, HGLRC hGlrc) : Renderer(){
	hdc = hDc;
	hglrc = hGlrc;

#elif defined(LINUX)

OpenGLRenderer::OpenGLRenderer(Window win, GLXContext glXc, Display *disp, int scr) : Renderer(){
	window = win;
	glxc = glXc;
	display = disp;
	screen = scr;

#elif defined(__APPLE__)

OpenGLRenderer::OpenGLRenderer(AGLContext aGlc) : Renderer(){
	aglc = aGlc;

#endif

	GLint units = 1;
	if (GL_ARB_fragment_shader_supported || GL_ARB_fragment_program_supported){
		glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS_ARB, &units);
	} else {
		glGetIntegerv(GL_MAX_TEXTURE_UNITS, &units);
	}
	nImageUnits = units;

	memset(textureLod, 0, sizeof(textureLod));

	if (GL_ARB_draw_buffers_supported){
		GLint mrt = 1;
		glGetIntegerv(GL_MAX_DRAW_BUFFERS_ARB, &mrt);
		nMRTs = mrt;
	}
	if (nMRTs > MAX_MRTS) nMRTs = MAX_MRTS;

	for (uint i = 0; i < nMRTs; i++){
		drawBuffers[i] = GL_COLOR_ATTACHMENT0_EXT + i;
	}

	if (GL_EXT_texture_filter_anisotropic_supported){
		GLint aniso = 1;
		glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);
		maxAnisotropic = aniso;
	}

	fbo = 0;
	currentFrontFace = GL_CW;
	currentVBO = 0;

	texVF   = VF_NONE;
	plainVF = VF_NONE;

	setGLdefaults();
	resetToDefaults();

	memset(activeVertexFormat, VF_NONE, sizeof(activeVertexFormat));

	uniformFuncs[CONSTANT_FLOAT] = (void *) glUniform1fvARB;
	uniformFuncs[CONSTANT_VEC2]  = (void *) glUniform2fvARB;
	uniformFuncs[CONSTANT_VEC3]  = (void *) glUniform3fvARB;
	uniformFuncs[CONSTANT_VEC4]  = (void *) glUniform4fvARB;
	uniformFuncs[CONSTANT_INT]   = (void *) glUniform1ivARB;
	uniformFuncs[CONSTANT_IVEC2] = (void *) glUniform2ivARB;
	uniformFuncs[CONSTANT_IVEC3] = (void *) glUniform3ivARB;
	uniformFuncs[CONSTANT_IVEC4] = (void *) glUniform4ivARB;
	uniformFuncs[CONSTANT_BOOL]  = (void *) glUniform1ivARB;
	uniformFuncs[CONSTANT_BVEC2] = (void *) glUniform2ivARB;
	uniformFuncs[CONSTANT_BVEC3] = (void *) glUniform3ivARB;
	uniformFuncs[CONSTANT_BVEC4] = (void *) glUniform4ivARB;
	uniformFuncs[CONSTANT_MAT2]  = (void *) glUniformMatrix2fvARB;
	uniformFuncs[CONSTANT_MAT3]  = (void *) glUniformMatrix3fvARB;
	uniformFuncs[CONSTANT_MAT4]  = (void *) glUniformMatrix4fvARB;
}

OpenGLRenderer::~OpenGLRenderer(){
	// Restore states to default
	reset();
	apply();

	// Delete shaders
	for (uint i = 0; i < shaders.getCount(); i++){
		for (uint j = 0; j < shaders[i].nSamplers; j++){
			delete shaders[i].samplers[j].name;
		}
		for (uint j = 0; j < shaders[i].nUniforms; j++){
			delete shaders[i].uniforms[j].name;
			delete shaders[i].uniforms[j].data;
		}
		delete shaders[i].samplers;
		delete shaders[i].uniforms;
		glDeleteObjectARB(shaders[i].vertexShader);
		glDeleteObjectARB(shaders[i].fragmentShader);
		glDeleteObjectARB(shaders[i].program);
	}

    // Delete vertex buffers
	for (uint i = 0; i < vertexBuffers.getCount(); i++){
		glDeleteBuffersARB(1, &vertexBuffers[i].vboVB);
	}

	// Delete index buffers
	for (uint i = 0; i < indexBuffers.getCount(); i++){
		glDeleteBuffersARB(1, &indexBuffers[i].vboIB);
	}

	// Delete textures
	for (uint i = 0; i < textures.getCount(); i++){
		removeTexture(i);
	}

	if (fbo) glDeleteFramebuffersEXT(1, &fbo);
}

void OpenGLRenderer::resetToDefaults(){
	Renderer::resetToDefaults();

	for (uint i = 0; i < MAX_TEXTUREUNIT; i++){
		currentTextures[i] = TEXTURE_NONE;
	}

	for (uint i = 0; i < MAX_SAMPLERSTATE; i++){
		currentSamplerStates[i] = SS_NONE;
	}

	currentSrcFactorRGB = currentSrcFactorAlpha = ONE;
	currentDstFactorRGB = currentDstFactorAlpha = ZERO;
	currentBlendModeRGB = currentBlendModeAlpha = BM_ADD;
	currentMask = ALL;
	currentBlendEnable = false;
	currentAlphaToCoverageEnable = false;

	currentDepthFunc = LEQUAL;
	currentDepthTestEnable = true;
	currentDepthWriteEnable = true;

	currentCullMode = CULL_NONE;
	currentFillMode = SOLID;
	currentMultiSampleEnable = true;
	currentScissorEnable = false;
}

void OpenGLRenderer::reset(const uint flags){
	Renderer::reset(flags);


	if (flags & RESET_TEX){
		for (uint i = 0; i < MAX_TEXTUREUNIT; i++){
			selectedTextures[i] = TEXTURE_NONE;
		}
	}

/*	if (flags & RESET_SS){
		for (uint i = 0; i < MAX_SAMPLERSTATE; i++){
			selectedSamplerStates[i] = SS_NONE;
		}
	}*/
}

static const GLint internalFormats[] = {
	0,
	// Unsigned formats
	GL_INTENSITY8,
	GL_LUMINANCE8_ALPHA8,
	GL_RGB8,
	GL_RGBA8,

	GL_INTENSITY16,
	GL_LUMINANCE16_ALPHA16,
	GL_RGB16,
	GL_RGBA16,

	// Signed formats
	0,
	0,
	0,
	0,

	0,
	0,
	0,
	0,

	// Float formats
	GL_INTENSITY_FLOAT16_ATI,
	GL_LUMINANCE_ALPHA_FLOAT16_ATI,
	GL_RGB_FLOAT16_ATI,
	GL_RGBA_FLOAT16_ATI,

	GL_INTENSITY_FLOAT32_ATI,
	GL_LUMINANCE_ALPHA_FLOAT32_ATI,
	GL_RGB_FLOAT32_ATI,
	GL_RGBA_FLOAT32_ATI,

	// Signed integer formats
	0, // GL_INTENSITY16I_EXT,
	0, // GL_LUMINANCE_ALPHA16I_EXT,
	0, // GL_RGB16I_EXT,
	0, // GL_RGBA16I_EXT,

	0, // GL_INTENSITY32I_EXT,
	0, // GL_LUMINANCE_ALPHA32I_EXT,
	0, // GL_RGB32I_EXT,
	0, // GL_RGBA32I_EXT,

	// Unsigned integer formats
	0, // GL_INTENSITY16UI_EXT,
	0, // GL_LUMINANCE_ALPHA16UI_EXT,
	0, // GL_RGB16UI_EXT,
	0, // GL_RGBA16UI_EXT,

	0, // GL_INTENSITY32UI_EXT,
	0, // GL_LUMINANCE_ALPHA32UI_EXT,
	0, // GL_RGB32UI_EXT,
	0, // GL_RGBA32UI_EXT,

	// Packed formats
	0, // RGBE8 not directly supported
	0, // GL_RGB9_E5,
	0, // GL_R11F_G11F_B10F,
	GL_RGB5,
	GL_RGBA4,
	GL_RGB10_A2,

	// Depth formats
	GL_DEPTH_COMPONENT16,
	GL_DEPTH_COMPONENT24,
	GL_DEPTH24_STENCIL8_EXT,
	0, // GL_DEPTH_COMPONENT32F,

	// Compressed formats
	GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
	GL_COMPRESSED_RGBA_S3TC_DXT3_EXT,
	GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
	0, // ATI1N not yet supported
	GL_COMPRESSED_LUMINANCE_ALPHA_3DC_ATI,
};

static const GLenum srcFormats[] = { 0, GL_LUMINANCE, GL_LUMINANCE_ALPHA, GL_RGB, GL_RGBA };

static const GLenum srcTypes[] = {
	0,
	// Unsigned formats
	GL_UNSIGNED_BYTE,
	GL_UNSIGNED_BYTE,
	GL_UNSIGNED_BYTE,
	GL_UNSIGNED_BYTE,

	GL_UNSIGNED_SHORT,
	GL_UNSIGNED_SHORT,
	GL_UNSIGNED_SHORT,
	GL_UNSIGNED_SHORT,

	// Signed formats
	0,
	0,
	0,
	0,

	0,
	0,
	0,
	0,

	// Float formats
	0,//GL_HALF_FLOAT_ARB,
	0,//GL_HALF_FLOAT_ARB,
	0,//GL_HALF_FLOAT_ARB,
	0,//GL_HALF_FLOAT_ARB,

	GL_FLOAT,
	GL_FLOAT,
	GL_FLOAT,
	GL_FLOAT,

	// Signed integer formats
	0,
	0,
	0,
	0,

	0,
	0,
	0,
	0,

	// Unsigned integer formats
	0,
	0,
	0,
	0,

	0,
	0,
	0,
	0,

	// Packed formats
	0, // RGBE8 not directly supported
	0, // RGBE9E5 not supported
	0, // RG11B10F not supported
	GL_UNSIGNED_SHORT_5_6_5,
	GL_UNSIGNED_SHORT_4_4_4_4_REV,
	GL_UNSIGNED_INT_2_10_10_10_REV,

	// Depth formats
	GL_UNSIGNED_SHORT,
	GL_UNSIGNED_INT,
	GL_UNSIGNED_INT_24_8_EXT,
	0, // D32F not supported

	// Compressed formats
	0,
	0,
	0,
	0,
	0,
};


TextureID OpenGLRenderer::addTexture(Image &img, const SamplerStateID samplerState, uint flags){
	ASSERT(samplerState != SS_NONE);

	Texture tex;
	memset(&tex, 0, sizeof(tex));

//	if (img.isCube()) flags |= (CLAMP_S | CLAMP_T);


	// If the target hardware doesn't support the compressed texture format, just decompress it to a compatible format
	FORMAT format = img.getFormat();
/*	if (isCompressedFormat(format)){
		if (!GL_ARB_texture_compression_supported || (format <= FORMAT_DXT5 && !GL_EXT_texture_compression_s3tc_supported) || (format == FORMAT_3DC && !GL_ATI_texture_compression_3dc_supported)){
			img.uncompressImage();
			format = img.getFormat();
		}
	} else if (format == FORMAT_RGBE8){
		img.convert(FORMAT_RGB32F);
		format = img.getFormat();
	}*/

	GLenum srcFormat = srcFormats[getChannelCount(format)];
    GLenum srcType = srcTypes[format];
	GLint internalFormat = internalFormats[format];
	if ((flags & HALF_FLOAT) != 0 && format >= FORMAT_I32F && format <= FORMAT_RGBA32F){
        internalFormat = internalFormats[format - (FORMAT_I32F - FORMAT_I16F)];
	}

	tex.lod = samplerStates[samplerState].lod;
	tex.format = format;
	tex.glTarget = img.isCube()? GL_TEXTURE_CUBE_MAP : img.is3D()? GL_TEXTURE_3D : img.is2D()? GL_TEXTURE_2D : GL_TEXTURE_1D;
	// Generate a texture
	glGenTextures(1, &tex.glTexID);
	glBindTexture(tex.glTarget, tex.glTexID);

	setupSampler(tex.glTarget, samplerStates[samplerState]);
	tex.samplerState = samplerState;

	// Upload it all
	ubyte *src;
	int mipMapLevel = 0;
	while ((src = img.getPixels(mipMapLevel)) != NULL){
		if (img.isCube()){
			int size = img.getMipMappedSize(mipMapLevel, 1) / 6;
			for (uint i = 0; i < 6; i++){
				if (isCompressedFormat(format)){
					glCompressedTexImage2DARB(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, mipMapLevel, internalFormat, img.getWidth(mipMapLevel), img.getHeight(mipMapLevel), 0, size, src + i * size);
				} else {
					glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, mipMapLevel, internalFormat, img.getWidth(mipMapLevel), img.getHeight(mipMapLevel), 0, srcFormat, srcType, src + i * size);
				}
			}
		} else if (img.is3D()){
			if (isCompressedFormat(format)){
				glCompressedTexImage3DARB(tex.glTarget, mipMapLevel, internalFormat, img.getWidth(mipMapLevel), img.getHeight(mipMapLevel), img.getDepth(mipMapLevel), 0, img.getMipMappedSize(mipMapLevel, 1), src);
			} else {
				glTexImage3D(tex.glTarget, mipMapLevel, internalFormat, img.getWidth(mipMapLevel), img.getHeight(mipMapLevel), img.getDepth(mipMapLevel), 0, srcFormat, srcType, src);
			}
		} else if (img.is2D()){
			if (isCompressedFormat(format)){
				glCompressedTexImage2DARB(tex.glTarget, mipMapLevel, internalFormat, img.getWidth(mipMapLevel), img.getHeight(mipMapLevel), 0, img.getMipMappedSize(mipMapLevel, 1), src);
			} else {
				glTexImage2D(tex.glTarget, mipMapLevel, internalFormat, img.getWidth(mipMapLevel), img.getHeight(mipMapLevel), 0, srcFormat, srcType, src);
			}
		} else {
			glTexImage1D(tex.glTarget, mipMapLevel, internalFormat, img.getWidth(mipMapLevel), 0, srcFormat, srcType, src);
		}
		mipMapLevel++;
	}

	glBindTexture(tex.glTarget, 0);

	return textures.add(tex);
}

TextureID OpenGLRenderer::addRenderTarget(const int width, const int height, const int depth, const int mipMapCount, const int arraySize, const FORMAT format, const int msaaSamples, const SamplerStateID samplerState, uint flags){
	if (isCompressedFormat(format) || arraySize > 1) return false;

	// TODO: Implement
	if (mipMapCount > 1) return false;

//	if (flags & CUBEMAP) flags |= CLAMP_ST;

	Texture tex;
	memset(&tex, 0, sizeof(tex));
	tex.glTarget = (flags & CUBEMAP)? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;
	tex.flags  = flags;
	tex.format = format;

	glGenTextures(1, &tex.glTexID);
	glBindTexture(tex.glTarget, tex.glTexID);
	setupSampler(tex.glTarget, samplerStates[samplerState]);
	tex.samplerState = samplerState;

	TextureID rt = textures.add(tex);
	setRenderTargetSize(rt, width, height);

	glBindTexture(tex.glTarget, 0);

	return rt;
}

TextureID OpenGLRenderer::addRenderDepth(const int width, const int height, const int arraySize, const FORMAT format, const int msaaSamples, const SamplerStateID samplerState, uint flags){
	if (arraySize > 1) return false;

	Texture tex;
	memset(&tex, 0, sizeof(tex));
	tex.glTarget = GL_RENDERBUFFER_EXT;
	tex.format = format;

	// Create depth renderbuffer
	glGenRenderbuffersEXT(1, &tex.glDepthID);

	TextureID rt = textures.add(tex);
	setRenderTargetSize(rt, width, height);

	return rt;
}

void OpenGLRenderer::setRenderTargetSize(const TextureID renderTarget, const int width, const int height){
	textures[renderTarget].width  = width;
	textures[renderTarget].height = height;

	if (textures[renderTarget].glTarget == GL_RENDERBUFFER_EXT){
		FORMAT format = textures[renderTarget].format;

		// Bind render buffer
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, textures[renderTarget].glDepthID);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, internalFormats[format], width, height);

		// Restore renderbuffer
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

	} else {
		FORMAT format = textures[renderTarget].format;

		GLint internalFormat = internalFormats[format];
		GLenum srcFormat = srcFormats[getChannelCount(format)];
		GLenum srcType = srcTypes[format];
		if (isDepthFormat(format)){
			if (isStencilFormat(format)){
				srcFormat = GL_DEPTH_STENCIL_EXT;
			} else {
				srcFormat = GL_DEPTH_COMPONENT;
			}
		}

		if (isFloatFormat(format)) srcType = GL_FLOAT;

		// Allocate all required surfaces.
		glBindTexture(textures[renderTarget].glTarget, textures[renderTarget].glTexID);
		if (textures[renderTarget].flags & CUBEMAP){
			for (int i = GL_TEXTURE_CUBE_MAP_POSITIVE_X; i <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z; i++)
				glTexImage2D(i, 0, internalFormat, width, height, 0, srcFormat, srcType, NULL);
		} else {
			glTexImage2D(textures[renderTarget].glTarget, 0, internalFormat, width, height, 0, srcFormat, srcType, NULL);
		}
		glBindTexture(textures[renderTarget].glTarget, 0);
	}
}

bool OpenGLRenderer::resizeRenderTarget(const TextureID renderTarget, const int width, const int height, const int depth, const int mipMapCount, const int arraySize){
	if (depth > 1 || arraySize > 1 || mipMapCount > 1) return false;

	if (width != textures[renderTarget].width || height != textures[renderTarget].height){
		setRenderTargetSize(renderTarget, width, height);
	}

	return true;
}

bool OpenGLRenderer::generateMipMaps(const TextureID renderTarget){
	// TODO: Implement
	return false;
}

void OpenGLRenderer::removeTexture(const TextureID texture){
	if (textures[texture].glTarget){
		if (textures[texture].glTarget == GL_RENDERBUFFER_EXT){
			glDeleteRenderbuffersEXT(1, &textures[texture].glDepthID);
		} else {
			glDeleteTextures(1, &textures[texture].glTexID);
		}
		textures[texture].glTarget = 0;
	}
}

int samplerCompare(const void *sampler0, const void *sampler1){
	return strcmp(((Sampler *) sampler0)->name, ((Sampler *) sampler1)->name);
}

ShaderID OpenGLRenderer::addShader(const char *vsText, const char *gsText, const char *fsText, const int vsLine, const int gsLine, const int fsLine,
                                   const char *header, const char *extra, const char *fileName, const char **attributeNames, const int nAttributes, const uint flags){

	if (!GLSL_supported || (vsText == NULL && fsText == NULL) || gsText != NULL) return SHADER_NONE;

	Shader shader;
	const GLcharARB *shaderStrings[6];
	int strIndex = 0;
	char line[16];
	GLint vsResult, fsResult, linkResult;
	char infoLog[2048];
	GLint len, infoLogPos = 0;

	// Compile to the highest supported language version
	if (GL_ARB_shading_language_100_supported){
		static char versionString[16];
		static bool first = true;
		if (first){
			const char *version = (const char *) glGetString(GL_SHADING_LANGUAGE_VERSION);
			if (version){
				int major = atoi(version);
				int minor = atoi(strchr(version, '.') + 1);
				sprintf(versionString, "#version %u%02u\n", major, minor);
			}
			first = false;
		}
		shaderStrings[strIndex++] = versionString;
	}

	// A couple of defines to make life easier for HLSL users
	shaderStrings[strIndex++] =
		"#define saturate(x) clamp(x,0.0,1.0)\n"
		"#define lerp mix\n";
	if (extra) shaderStrings[strIndex++] = extra;
	if (header) shaderStrings[strIndex++] = header;
	shaderStrings[strIndex++] = line;

	shader.vertexShader = 0;
	shader.fragmentShader = 0;
	shader.program = glCreateProgramObjectARB();

	// Compile the vertex shader
	if (vsText != NULL){
		shader.vertexShader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
		sprintf(line, "#line %d\n", vsLine);
		shaderStrings[strIndex] = vsText;
		glShaderSourceARB(shader.vertexShader, strIndex + 1, shaderStrings, NULL);
		glCompileShaderARB(shader.vertexShader);
		glGetObjectParameterivARB(shader.vertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &vsResult);
		if (vsResult){
			glAttachObjectARB(shader.program, shader.vertexShader);
		} else {
			infoLogPos += sprintf(infoLog + infoLogPos, "Vertex shader error:\n");
		}
		glGetInfoLogARB(shader.vertexShader, sizeof(infoLog) - infoLogPos, &len, infoLog + infoLogPos);
		infoLogPos += len;
	} else vsResult = GL_TRUE;

	// Compile the fragment shader
	if (fsText != NULL){
		shader.fragmentShader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
		sprintf(line, "#line %d\n", fsLine);
		shaderStrings[strIndex] = fsText;
		glShaderSourceARB(shader.fragmentShader, strIndex + 1, shaderStrings, NULL);
		glCompileShaderARB(shader.fragmentShader);
		glGetObjectParameterivARB(shader.fragmentShader, GL_OBJECT_COMPILE_STATUS_ARB, &fsResult);
		if (fsResult){
			glAttachObjectARB(shader.program, shader.fragmentShader);
		} else {
			infoLogPos += sprintf(infoLog + infoLogPos, "Fragment shader error:\n");
		}
		glGetInfoLogARB(shader.fragmentShader, sizeof(infoLog) - infoLogPos, &len, infoLog + infoLogPos);
		infoLogPos += len;
	} else fsResult = GL_TRUE;

	// Link the shaders
	if (vsResult && fsResult){
		for (int i = 0; i < nAttributes; i++){
			if (attributeNames[i]) glBindAttribLocationARB(shader.program, i, attributeNames[i]);
		}

		glLinkProgramARB(shader.program);
		glGetObjectParameterivARB(shader.program, GL_OBJECT_LINK_STATUS_ARB, &linkResult);
		glGetInfoLogARB(shader.program, sizeof(infoLog) - infoLogPos, &len, infoLog + infoLogPos);
		infoLogPos += len;

		outputDebugString(infoLog);

		if (linkResult){
			GLuint currProgram = (currentShader == SHADER_NONE)? 0 : shaders[currentShader].program;

			glUseProgram(shader.program);

			GLint uniformCount, maxLength;
			glGetObjectParameterivARB(shader.program, GL_OBJECT_ACTIVE_UNIFORMS_ARB, &uniformCount);
			glGetObjectParameterivARB(shader.program, GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH_ARB, &maxLength);

			Sampler  *samplers = (Sampler  *) malloc(uniformCount * sizeof(Sampler));
			Constant *uniforms = (Constant *) malloc(uniformCount * sizeof(Constant));

			int nSamplers = 0;
			int nUniforms = 0;
			char *name = new char[maxLength];
			for (int i = 0; i < uniformCount; i++){
				GLenum type;
				GLint length, size;
				glGetActiveUniform(shader.program, i, maxLength, &length, &size, &type, name);

				if (type >= GL_SAMPLER_1D && type <= GL_SAMPLER_2D_RECT_SHADOW_ARB){
					// Assign samplers to image units
					GLint location = glGetUniformLocation(shader.program, name);
					glUniform1i(location, nSamplers);

					samplers[nSamplers].name = new char[length + 1];
					samplers[nSamplers].index = nSamplers;
					strcpy(samplers[nSamplers].name, name);
					nSamplers++;
				} else {
					// Store all non-gl uniforms
					if (strncmp(name, "gl_", 3) != 0){
						char *bracket = strchr(name, '[');
						if (bracket == NULL || (bracket[1] == '0' && bracket[2] == ']')){
							if (bracket){
								*bracket = '\0';
								length = (GLint) (bracket - name);
							}

							uniforms[nUniforms].name = new char[length + 1];
							uniforms[nUniforms].index = glGetUniformLocation(shader.program, name);
							uniforms[nUniforms].type = getConstantType(type);
							uniforms[nUniforms].nElements = size;
							strcpy(uniforms[nUniforms].name, name);
							nUniforms++;
						} else if (bracket != NULL && bracket[1] > '0'){
							*bracket = '\0';
							for (int i = nUniforms - 1; i >= 0; i--){
								if (strcmp(uniforms[i].name, name) == 0){
									int index = atoi(bracket + 1) + 1;
									if (index > uniforms[i].nElements){
										uniforms[i].nElements = index;
									}
								}
							}
						}
					}
				}
			}

			// Restore the current program
			glUseProgram(currProgram);

			delete [] name;

			// Shorten arrays to actual count
			samplers = (Sampler  *) realloc(samplers, nSamplers * sizeof(Sampler));
			uniforms = (Constant *) realloc(uniforms, nUniforms * sizeof(Constant));
			qsort(samplers, nSamplers, sizeof(Sampler),  samplerComp);
			qsort(uniforms, nUniforms, sizeof(Constant), constantComp);

			for (int i = 0; i < nUniforms; i++){
				int constantSize = constantTypeSizes[uniforms[i].type] * uniforms[i].nElements;
				uniforms[i].data = new ubyte[constantSize];
				memset(uniforms[i].data, 0, constantSize);
				uniforms[i].dirty = false;
			}
			shader.uniforms  = uniforms;
			shader.samplers  = samplers;
			shader.nUniforms = nUniforms;
			shader.nSamplers = nSamplers;

			return shaders.add(shader);
		}
	}

	ErrorMsg(infoLog);
	return SHADER_NONE;
}

VertexFormatID OpenGLRenderer::addVertexFormat(const FormatDesc *formatDesc, const uint nAttribs, const ShaderID shader){
	VertexFormat vertexFormat;

	memset(&vertexFormat, 0, sizeof(vertexFormat));

	uint nGeneric  = 0;
	uint nTexCoord = 0;
	for (uint i = 0; i < nAttribs; i++){
		// Generic attribute 0 aliases with gl_Vertex
		if (formatDesc[i].type == TYPE_VERTEX){
			nGeneric = 1;
			break;
		}
	}

	for (uint i = 0; i < nAttribs; i++){
		int stream = formatDesc[i].stream;

		switch (formatDesc[i].type){
		case TYPE_GENERIC:
		case TYPE_TANGENT:
		case TYPE_BINORMAL:
			vertexFormat.generic[nGeneric].stream = stream;
			vertexFormat.generic[nGeneric].size   = formatDesc[i].size;
			vertexFormat.generic[nGeneric].offset = vertexFormat.vertexSize[stream];
			vertexFormat.generic[nGeneric].format = formatDesc[i].format;
			nGeneric++;
			break;
		case TYPE_VERTEX:
			vertexFormat.vertex.stream = stream;
			vertexFormat.vertex.size   = formatDesc[i].size;
			vertexFormat.vertex.offset = vertexFormat.vertexSize[stream];
			vertexFormat.vertex.format = formatDesc[i].format;
			break;
		case TYPE_NORMAL:
			vertexFormat.normal.stream = stream;
			vertexFormat.normal.size   = formatDesc[i].size;
			vertexFormat.normal.offset = vertexFormat.vertexSize[stream];
			vertexFormat.normal.format = formatDesc[i].format;
			break;
		case TYPE_TEXCOORD:
			vertexFormat.texCoord[nTexCoord].stream = stream;
			vertexFormat.texCoord[nTexCoord].size   = formatDesc[i].size;
			vertexFormat.texCoord[nTexCoord].offset = vertexFormat.vertexSize[stream];
			vertexFormat.texCoord[nTexCoord].format = formatDesc[i].format;
			nTexCoord++;
			break;
		}
		
		vertexFormat.vertexSize[stream] += formatDesc[i].size * getFormatSize(formatDesc[i].format);
	}

	vertexFormat.maxGeneric = nGeneric;
	vertexFormat.maxTexCoord = nTexCoord;

	return vertexFormats.add(vertexFormat);
}

GLenum usages[] = {
	GL_STATIC_DRAW,
	GL_DYNAMIC_DRAW,
	GL_STREAM_DRAW,
};

VertexBufferID OpenGLRenderer::addVertexBuffer(const long size, const BufferAccess bufferAccess, const void *data){
	VertexBuffer vb;
	vb.size = size;

	glGenBuffersARB(1, &vb.vboVB);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vb.vboVB);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, size, data, usages[bufferAccess]);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	return vertexBuffers.add(vb);
}

IndexBufferID OpenGLRenderer::addIndexBuffer(const uint nIndices, const uint indexSize, const BufferAccess bufferAccess, const void *data){
	IndexBuffer ib;

	ib.nIndices = nIndices;
	ib.indexSize = indexSize;

	uint size = nIndices * indexSize;
	glGenBuffersARB(1, &ib.vboIB);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, ib.vboIB);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, size, data, usages[bufferAccess]);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

	return indexBuffers.add(ib);
}

GLint minFilters[] = {
	GL_NEAREST,
	GL_LINEAR,
	GL_LINEAR_MIPMAP_NEAREST,
	GL_LINEAR_MIPMAP_LINEAR,
	GL_LINEAR_MIPMAP_NEAREST,
	GL_LINEAR_MIPMAP_LINEAR,
};

SamplerStateID OpenGLRenderer::addSamplerState(const Filter filter, const AddressMode s, const AddressMode t, const AddressMode r, const float lod, const uint maxAniso, const int compareFunc, const float *border_color){
	// TODO: Implement this ...
	if (compareFunc != 0 || border_color != NULL) return SS_NONE;

	SamplerState samplerState;

	samplerState.minFilter = minFilters[filter];
	samplerState.magFilter = (filter == NEAREST)? GL_NEAREST : GL_LINEAR;
	samplerState.wrapS = (s == WRAP)? GL_REPEAT : GL_CLAMP_TO_EDGE;
	samplerState.wrapT = (t == WRAP)? GL_REPEAT : GL_CLAMP_TO_EDGE;
	samplerState.wrapR = (r == WRAP)? GL_REPEAT : GL_CLAMP_TO_EDGE;
	samplerState.aniso = hasAniso(filter)? min((uint) maxAnisotropic, maxAniso) : 1;
	samplerState.lod = lod;

	return samplerStates.add(samplerState);
}

BlendStateID OpenGLRenderer::addBlendState(const int srcFactorRGB, const int destFactorRGB, const int srcFactorAlpha, const int destFactorAlpha, const int blendModeRGB, const int blendModeAlpha, const int mask, const bool alphaToCoverage){
	BlendState blendState;

	blendState.srcFactorRGB   = srcFactorRGB;
	blendState.dstFactorRGB   = destFactorRGB;
	blendState.srcFactorAlpha = srcFactorAlpha;
	blendState.dstFactorAlpha = destFactorAlpha;
	blendState.blendModeRGB   = blendModeRGB;
	blendState.blendModeAlpha = blendModeAlpha;
	blendState.mask = mask;
	blendState.blendEnable = (srcFactorRGB != ONE || destFactorRGB != ZERO || srcFactorAlpha != ONE || destFactorAlpha != ZERO);
	blendState.alphaToCoverageEnable = alphaToCoverage;

	return blendStates.add(blendState);
}

DepthStateID OpenGLRenderer::addDepthState(const bool depthTest, const bool depthWrite, const int depthFunc, const bool stencilTest, const uint8 stencilReadMask, const uint8 stencilWriteMask,
						   const int stencilFuncFront, const int stencilFuncBack, const int stencilFailFront, const int stencilFailBack,
						   const int depthFailFront, const int depthFailBack, const int stencilPassFront, const int stencilPassBack){
	DepthState depthState;

	depthState.depthTest  = depthTest;
	depthState.depthWrite = depthWrite;
	depthState.depthFunc  = depthFunc;

	return depthStates.add(depthState);
}

RasterizerStateID OpenGLRenderer::addRasterizerState(const int cullMode, const int fillMode, const bool multiSample, const bool scissor, const float depthBias, const float slopeDepthBias){
	RasterizerState rasterizerState;

	rasterizerState.cullMode = cullMode;
	rasterizerState.fillMode = fillMode;
	rasterizerState.multiSample = multiSample;
	rasterizerState.scissor = scissor;

	return rasterizerStates.add(rasterizerState);
}

int OpenGLRenderer::getSamplerUnit(const ShaderID shader, const char *samplerName) const {
	ASSERT(shader != SHADER_NONE);

	Sampler *samplers = shaders[shader].samplers;
	int minSampler = 0;
	int maxSampler = shaders[shader].nSamplers - 1;

	// Do a quick lookup in the sorted table with a binary search
	while (minSampler <= maxSampler){
		int currSampler = (minSampler + maxSampler) >> 1;
        int res = strcmp(samplerName, samplers[currSampler].name);
		if (res == 0){
			return samplers[currSampler].index;
		} else if (res > 0){
            minSampler = currSampler + 1;
		} else {
            maxSampler = currSampler - 1;
		}
	}

	return -1;
}

/*
void OpenGLRenderer::changeTexture(const uint imageUnit, const TextureID texture){
	TextureID currTex = currentTextures[imageUnit];

	if (texture != currTex){
		glActiveTextureARB(GL_TEXTURE0 + imageUnit);
		if (texture == TEXTURE_NONE){
			glDisable(textures[currTex].glTarget);
		} else {
			if (currTex == TEXTURE_NONE){
				glEnable(textures[texture].glTarget);
			} else if (textures[texture].glTarget != textures[currTex].glTarget){
				glDisable(textures[currTex].glTarget);
				glEnable(textures[texture].glTarget);
			}
			glBindTexture(textures[texture].glTarget, textures[texture].glTexID);
			//if (textures[textureID].lod != textureLod[imageUnit]){
			//	glTexEnvf(GL_TEXTURE_FILTER_CONTROL_EXT, GL_TEXTURE_LOD_BIAS_EXT, textureLod[imageUnit] = textures[textureID].lod);
			//}
		}

		currentTextures[imageUnit] = texture;
	}
}

void OpenGLRenderer::changeSamplerState(const uint samplerUnit, const SamplerStateID samplerState){
	TextureID currTex = currentTextures[samplerUnit];
	if (currTex != TEXTURE_NONE){
		if (samplerState != textures[currTex].samplerState){
			ASSERT(samplerState != SS_NONE);
			
			glActiveTextureARB(GL_TEXTURE0 + samplerUnit);
			setupSampler(textures[currTex].glTarget, samplerStates[samplerState]);

			textures[currTex].samplerState = samplerState;
		}
	}
}

*/
void OpenGLRenderer::setTexture(const char *textureName, const TextureID texture){
	ASSERT(selectedShader != SHADER_NONE);

	int unit = getSamplerUnit(selectedShader, textureName);
	if (unit >= 0){
		selectedTextures[unit] = texture;
	}
}

void OpenGLRenderer::setTexture(const char *textureName, const TextureID texture, const SamplerStateID samplerState){
	ASSERT(selectedShader != SHADER_NONE);

	int unit = getSamplerUnit(selectedShader, textureName);
	if (unit >= 0){
		selectedTextures[unit] = texture;
		selectedSamplerStates[unit] = samplerState;
	}
}

void OpenGLRenderer::setTextureSlice(const char *textureName, const TextureID texture, const int slice){
	ASSERT(0);
}

void OpenGLRenderer::applyTextures(){
	for (uint i = 0; i < MAX_TEXTUREUNIT; i++){
		TextureID texture = selectedTextures[i];
		TextureID currTex = currentTextures[i];

		if (texture != currTex){
			glActiveTextureARB(GL_TEXTURE0 + i);
			if (texture == TEXTURE_NONE){
				glDisable(textures[currTex].glTarget);
				glBindTexture(textures[currTex].glTarget, 0);
			} else {
				if (currTex == TEXTURE_NONE){
					glEnable(textures[texture].glTarget);
				} else if (textures[texture].glTarget != textures[currTex].glTarget){
					glDisable(textures[currTex].glTarget);
					glEnable(textures[texture].glTarget);
				}
				glBindTexture(textures[texture].glTarget, textures[texture].glTexID);
				if (textures[texture].lod != textureLod[i]){
					glTexEnvf(GL_TEXTURE_FILTER_CONTROL_EXT, GL_TEXTURE_LOD_BIAS_EXT, textureLod[i] = textures[texture].lod);
				}
			}

			currentTextures[i] = texture;
		}
	}
}

void OpenGLRenderer::setSamplerState(const char *samplerName, const SamplerStateID samplerState){

}

void OpenGLRenderer::applySamplerStates(){

}

void OpenGLRenderer::setShaderConstantRaw(const char *name, const void *data, const int size){
	int minUniform = 0;
	int maxUniform = shaders[selectedShader].nUniforms - 1;
	Constant *uniforms = shaders[selectedShader].uniforms;

	// Do a quick lookup in the sorted table with a binary search
	while (minUniform <= maxUniform){
		int currUniform = (minUniform + maxUniform) >> 1;
		int res = strcmp(name, uniforms[currUniform].name);
		if (res == 0){
			Constant *uni = uniforms + currUniform;

			if (memcmp(uni->data, data, size)){
				memcpy(uni->data, data, size);
				uni->dirty = true;
			}
			return;

		} else if (res > 0){
			minUniform = currUniform + 1;
		} else {
			maxUniform = currUniform - 1;
		}
	}
}

void OpenGLRenderer::applyConstants(){
	if (currentShader != SHADER_NONE){
		for (uint i = 0; i < shaders[currentShader].nUniforms; i++){
			Constant *uni = shaders[currentShader].uniforms + i;
			if (uni->dirty){
				if (uni->type >= CONSTANT_MAT2){
					((UNIFORM_MAT_FUNC) uniformFuncs[uni->type])(uni->index, uni->nElements, GL_TRUE, (float *) uni->data);
				} else {
					((UNIFORM_FUNC) uniformFuncs[uni->type])(uni->index, uni->nElements, (float *) uni->data);
				}
				uni->dirty = false;
			}
		}
	}
}

void OpenGLRenderer::changeRenderTargets(const TextureID *colorRTs, const uint nRenderTargets, const TextureID depthRT, const int depthSlice, const int *slices){
	// Reset states to default before switching render target
	reset();
	apply();

	if (fbo == 0) glGenFramebuffersEXT(1, &fbo);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);

	for (uint i = 0; i < nRenderTargets; i++){
		TextureID colorRT = colorRTs[i];

		if (textures[colorRT].flags & CUBEMAP){
			if (colorRT != currentColorRT[i] || currentColorRTSlice[i] != slices[i]){
				glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT + i, GL_TEXTURE_CUBE_MAP_POSITIVE_X + slices[i], textures[colorRT].glTexID, 0);
				currentColorRTSlice[i] = slices[i];
			}
		} else {
			if (colorRT != currentColorRT[i]){
				glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT + i, GL_TEXTURE_2D, textures[colorRT].glTexID, 0);
				currentColorRT[i] = colorRT;
			}
		}
	}
	
	if (nRenderTargets != nCurrentRenderTargets){
		for (uint i = nRenderTargets; i < nCurrentRenderTargets; i++){
			glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT + i, GL_TEXTURE_2D, 0, 0);
			currentColorRT[i] = TEXTURE_NONE;
			currentColorRTSlice[i] = NO_SLICE;
		}
		
		if (nRenderTargets == 0){
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
		} else {
			glDrawBuffersARB(nRenderTargets, drawBuffers);
			glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
		}
		nCurrentRenderTargets = nRenderTargets;
	}

	if (depthRT != currentDepthRT){
		if (depthRT >= 0 && textures[depthRT].glTarget != GL_RENDERBUFFER_EXT){
			glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, textures[depthRT].glTexID, 0);
			if (isStencilFormat(textures[depthRT].format)){
				glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, textures[depthRT].glTexID, 0);
			} else {
				glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, 0, 0);
			}
		} else {
			glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, (depthRT < 0)? 0 : textures[depthRT].glTexID);
			if (depthRT >= 0 && isStencilFormat(textures[depthRT].format)){
				glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, textures[depthRT].glTexID);
			} else {
				glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);
			}
		}
		currentDepthRT = depthRT;
	}

	if (nRenderTargets > 0){
		if (textures[colorRTs[0]].flags & CUBEMAP){
			changeFrontFace(GL_CCW);
		} else {
			changeFrontFace(GL_CW);
		}
		glViewport(0, 0, textures[colorRTs[0]].width, textures[colorRTs[0]].height);
	} else if(depthRT >= 0){
		changeFrontFace(GL_CW);
		glViewport(0, 0, textures[depthRT].width, textures[depthRT].height);
	}
}

void OpenGLRenderer::changeToMainFramebuffer(){
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glViewport(0, 0, viewportWidth, viewportHeight);
	changeFrontFace(GL_CW);
}

void OpenGLRenderer::changeShader(const ShaderID shader){
	if (shader != currentShader){
		if (shader == SHADER_NONE){
			glUseProgramObjectARB(0);
		} else {
			glUseProgramObjectARB(shaders[shader].program);
		}
		currentShader = shader;
	}
}

void OpenGLRenderer::changeVertexFormat(const VertexFormatID vertexFormat){
	if (vertexFormat != currentVertexFormat){
		static VertexFormat zero;
		VertexFormat *curr = &zero, *sel = &zero;

		if (currentVertexFormat != VF_NONE) curr = &vertexFormats[currentVertexFormat];
		if (vertexFormat != VF_NONE) sel = &vertexFormats[vertexFormat];

		// Change array enables as needed
		if ( sel->vertex.size && !curr->vertex.size) glEnableClientState (GL_VERTEX_ARRAY);
		if (!sel->vertex.size &&  curr->vertex.size) glDisableClientState(GL_VERTEX_ARRAY);
		if ( sel->normal.size && !curr->normal.size) glEnableClientState (GL_NORMAL_ARRAY);
		if (!sel->normal.size &&  curr->normal.size) glDisableClientState(GL_NORMAL_ARRAY);

		for (int i = 0; i < MAX_GENERIC; i++){
			if ( sel->generic[i].size && !curr->generic[i].size) glEnableVertexAttribArrayARB(i);
			if (!sel->generic[i].size &&  curr->generic[i].size) glDisableVertexAttribArrayARB(i);
		}

		for (int i = 0; i < MAX_TEXCOORD; i++){
			if ((sel->texCoord[i].size > 0) ^ (curr->texCoord[i].size > 0)){
				glClientActiveTexture(GL_TEXTURE0 + i);
				if (sel->texCoord[i].size > 0){
					glEnableClientState(GL_TEXTURE_COORD_ARRAY);
				} else {
		            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
				}
			}
		}

		currentVertexFormat = vertexFormat;
	}
}

void OpenGLRenderer::changeVertexBuffer(const int stream, const VertexBufferID vertexBuffer, const intptr offset){
	const GLsizei glTypes[] = {
		GL_FLOAT,
		0, // No half float support
		GL_UNSIGNED_BYTE,
	};

	GLuint vbo = 0;
	if (vertexBuffer != VB_NONE) vbo = vertexBuffers[vertexBuffer].vboVB;

	if (vbo != currentVBO){
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, currentVBO = vbo);
	}

	if (vertexBuffer != currentVertexBuffers[stream] || offset != currentOffsets[stream] || currentVertexFormat != activeVertexFormat[stream]){
		if (currentVertexFormat != VF_NONE){
			char *base = (char *) offset;

			VertexFormat *cvf = &vertexFormats[currentVertexFormat];
			int vertexSize = cvf->vertexSize[stream];

			if (cvf->vertex.stream == stream && cvf->vertex.size){
				glVertexPointer(cvf->vertex.size, glTypes[cvf->vertex.format], vertexSize, base + cvf->vertex.offset);
			}

			if (cvf->normal.stream == stream && cvf->normal.size){
				glNormalPointer(glTypes[cvf->normal.format], vertexSize, base + cvf->normal.offset);
			}

			for (int i = 0; i < MAX_GENERIC; i++){
				if (cvf->generic[i].stream == stream && cvf->generic[i].size){
					glVertexAttribPointerARB(i, cvf->generic[i].size, glTypes[cvf->generic[i].format], GL_TRUE, vertexSize, base + cvf->generic[i].offset);
				}
			}

			for (int i = 0; i < MAX_TEXCOORD; i++){
				if (cvf->texCoord[i].stream == stream && cvf->texCoord[i].size){
					glClientActiveTexture(GL_TEXTURE0 + i);
					glTexCoordPointer(cvf->texCoord[i].size, glTypes[cvf->texCoord[i].format], vertexSize, base + cvf->texCoord[i].offset);
				}
			}
		}

		currentVertexBuffers[stream] = vertexBuffer;
		currentOffsets[stream] = offset;
		activeVertexFormat[stream] = currentVertexFormat;
	}
}

void OpenGLRenderer::changeIndexBuffer(const IndexBufferID indexBuffer){
	if (indexBuffer != currentIndexBuffer){
		if (indexBuffer == IB_NONE){
			glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
		} else {
			glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indexBuffers[indexBuffer].vboIB);
		}

		currentIndexBuffer = indexBuffer;
	}
}

void OpenGLRenderer::changeBlendState(const BlendStateID blendState, const uint sampleMask){
	if (blendState != currentBlendState){
		if (blendState == BS_NONE || !blendStates[blendState].blendEnable){
			if (currentBlendEnable){
				glDisable(GL_BLEND);
				currentBlendEnable = false;
			}
		} else {
			if (blendStates[blendState].blendEnable){
				if (!currentBlendEnable){
					glEnable(GL_BLEND);
					currentBlendEnable = true;
				}
				if (blendStates[blendState].srcFactorRGB   != currentSrcFactorRGB ||
					blendStates[blendState].dstFactorRGB   != currentDstFactorRGB ||
					blendStates[blendState].srcFactorAlpha != currentSrcFactorAlpha ||
					blendStates[blendState].dstFactorAlpha != currentDstFactorAlpha){

					glBlendFuncSeparate(
						currentSrcFactorRGB   = blendStates[blendState].srcFactorRGB,
						currentDstFactorRGB   = blendStates[blendState].dstFactorRGB,
						currentSrcFactorAlpha = blendStates[blendState].srcFactorAlpha,
						currentDstFactorAlpha = blendStates[blendState].dstFactorAlpha);
				}
				if (blendStates[blendState].blendModeRGB   != currentBlendModeRGB ||
					blendStates[blendState].blendModeAlpha != currentBlendModeAlpha){
					glBlendEquationSeparate(
						currentBlendModeRGB   = blendStates[blendState].blendModeRGB,
						currentBlendModeAlpha = blendStates[blendState].blendModeAlpha);
				}
			}
		}

		int mask = ALL;
		bool alphaToCoverage = false;
		if (blendState != BS_NONE){
			mask = blendStates[blendState].mask;
			alphaToCoverage = blendStates[blendState].alphaToCoverageEnable;
		}

		if (mask != currentMask){
			glColorMask((mask & RED) & 1, ((mask & GREEN) >> 1) & 1, ((mask & BLUE) >> 2) & 1, ((mask & ALPHA) >> 3) & 1);

			currentMask = mask;
		}

		if (alphaToCoverage != currentAlphaToCoverageEnable)
		{
			if (alphaToCoverage)
				glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
			else
				glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);

			currentAlphaToCoverageEnable = alphaToCoverage;
		}

		currentBlendState = blendState;
	}
}

void OpenGLRenderer::changeDepthState(const DepthStateID depthState, const uint stencilRef){
	// stencilRef currently not used

	if (depthState != currentDepthState){
		if (depthState == DS_NONE){
			if (!currentDepthTestEnable){
				glEnable(GL_DEPTH_TEST);
				currentDepthTestEnable = true;
			}

			if (!currentDepthWriteEnable){
				glDepthMask(GL_TRUE);
				currentDepthWriteEnable = true;
			}

			if (currentDepthFunc != LEQUAL){
				glDepthFunc(currentDepthFunc = LEQUAL);
			}
		} else {
			if (depthStates[depthState].depthTest){
				if (!currentDepthTestEnable){
					glEnable(GL_DEPTH_TEST);
					currentDepthTestEnable = true;
				}
				if (depthStates[depthState].depthWrite != currentDepthWriteEnable){
					glDepthMask((currentDepthWriteEnable = depthStates[depthState].depthWrite)? GL_TRUE : GL_FALSE);
				}
				if (depthStates[depthState].depthFunc != currentDepthFunc){
					glDepthFunc(currentDepthFunc = depthStates[depthState].depthFunc);
				}
			} else {
				if (currentDepthTestEnable){
					glDisable(GL_DEPTH_TEST);
					currentDepthTestEnable = false;
				}
			}
		}

		currentDepthState = depthState;
	}
}

void OpenGLRenderer::changeRasterizerState(const RasterizerStateID rasterizerState){
	if (rasterizerState != currentRasterizerState){
		RasterizerState state;
		if (rasterizerState == RS_NONE){
			state.cullMode = CULL_NONE;
			state.fillMode = SOLID;
			state.multiSample = true;
			state.scissor = false;
		} else {
			state = rasterizerStates[rasterizerState];
		}


		if (state.cullMode != currentCullMode){
			if (state.cullMode == CULL_NONE){
				glDisable(GL_CULL_FACE);
			} else {
				if (currentCullMode == CULL_NONE) glEnable(GL_CULL_FACE);
				glCullFace(state.cullMode);
			}

			currentCullMode = state.cullMode;
		}

		if (state.fillMode != currentFillMode){
			glPolygonMode(GL_FRONT_AND_BACK, currentFillMode = state.fillMode);
		}

		if (state.multiSample != currentMultiSampleEnable){
			if (state.multiSample){
				glEnable(GL_MULTISAMPLE);
			} else {
				glDisable(GL_MULTISAMPLE);
			}
			currentMultiSampleEnable = state.multiSample;
		}

		if (state.scissor != currentScissorEnable){
			if (state.scissor){
				glEnable(GL_SCISSOR_TEST);
			} else {
				glDisable(GL_SCISSOR_TEST);
			}
			currentScissorEnable = state.scissor;
		}

		currentRasterizerState = rasterizerState;
	}
}
/*
void OpenGLRenderer::changeShaderConstant1i(const char *name, const int constant){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniform1iARB(location, constant);
}

void OpenGLRenderer::changeShaderConstant1f(const char *name, const float constant){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniform1fARB(location, constant);
}

void OpenGLRenderer::changeShaderConstant2f(const char *name, const vec2 &constant){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniform2fvARB(location, 1, (float *) &constant);
}

void OpenGLRenderer::changeShaderConstant3f(const char *name, const vec3 &constant){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniform3fvARB(location, 1, (float *) &constant);
}

void OpenGLRenderer::changeShaderConstant4f(const char *name, const vec4 &constant){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniform4fvARB(location, 1, (float *) &constant);
}

void OpenGLRenderer::changeShaderConstant3x3f(const char *name, const mat3 &constant){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniformMatrix3fvARB(location, 1, GL_TRUE, (float *) &constant);
}

void OpenGLRenderer::changeShaderConstant4x4f(const char *name, const mat4 &constant){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniformMatrix4fvARB(location, 1, GL_TRUE, (float *) &constant);
}

void OpenGLRenderer::changeShaderConstantArray1f(const char *name, const float *constant, const uint count){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniform1fvARB(location, count, (float *) constant);
}

void OpenGLRenderer::changeShaderConstantArray2f(const char *name, const vec2 *constant, const uint count){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniform2fvARB(location, count, (float *) constant);
}

void OpenGLRenderer::changeShaderConstantArray3f(const char *name, const vec3 *constant, const uint count){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniform3fvARB(location, count, (float *) constant);
}

void OpenGLRenderer::changeShaderConstantArray4f(const char *name, const vec4 *constant, const uint count){
	ASSERT(currentShader != SHADER_NONE);
	int location = glGetUniformLocationARB(shaders[currentShader].program, name);
	ASSERT(location >= 0);
	glUniform4fvARB(location, count, (float *) constant);
}
*/



const GLenum glPrim[] = {
	GL_TRIANGLES,
	GL_TRIANGLE_FAN,
	GL_TRIANGLE_STRIP,
	GL_QUADS,
	GL_LINES,
	GL_LINE_STRIP,
	GL_LINE_LOOP,
	GL_POINTS,
};

void OpenGLRenderer::clear(const bool clearColor, const bool clearDepth, const bool clearStencil, const float *color, const float depth, const uint stencil){
	GLbitfield clearBits = 0;

	if (clearColor){
		clearBits |= GL_COLOR_BUFFER_BIT;
		glClearColor(color[0], color[1], color[2], color[3]);

		if (currentMask != ALL){
			glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
			currentMask = ALL;
		}
	}
	if (clearDepth){
		clearBits |= GL_DEPTH_BUFFER_BIT;
		glClearDepth(depth);
		if (!currentDepthWriteEnable){
			glDepthMask(GL_TRUE);
			currentDepthWriteEnable = true;
		}
	}
	if (clearStencil){
		clearBits |= GL_STENCIL_BUFFER_BIT;
		glClearStencil(stencil);
	}

	if (clearBits){
		glClear(clearBits);
	}
}

void OpenGLRenderer::drawArrays(const Primitives primitives, const int firstVertex, const int nVertices){
	glDrawArrays(glPrim[primitives], firstVertex, nVertices);

	nDrawCalls++;
}

void OpenGLRenderer::drawElements(const Primitives primitives, const int firstIndex, const int nIndices, const int firstVertex, const int nVertices){
	uint indexSize = indexBuffers[currentIndexBuffer].indexSize;

	glDrawElements(glPrim[primitives], nIndices, indexSize == 2? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT, BUFFER_OFFSET(indexSize * firstIndex));

	nDrawCalls++;
}

void OpenGLRenderer::setup2DMode(const float left, const float right, const float top, const float bottom){
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(left, right, bottom, top, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void OpenGLRenderer::drawPlain(const Primitives primitives, vec2 *vertices, const uint nVertices, const BlendStateID blendState, const DepthStateID depthState, const vec4 *color){
	if (plainVF == VF_NONE){
		FormatDesc format[] = {
			{ 0, TYPE_VERTEX, FORMAT_FLOAT, 2 }
		};
		plainVF = addVertexFormat(format, 1);
	}

	reset();
	setBlendState(blendState);
	setDepthState(depthState);
	setVertexFormat(plainVF);
	setVertexBuffer(0, vertices);
	apply();
	
	if (color) glColor4fv(*color); else glColor3f(1, 1, 1);

	glDrawArrays(glPrim[primitives], 0, nVertices);

	nDrawCalls++;
}

void OpenGLRenderer::drawTextured(const Primitives primitives, TexVertex *vertices, const uint nVertices, const TextureID texture, const SamplerStateID samplerState, const BlendStateID blendState, const DepthStateID depthState, const vec4 *color){
	if (texVF == VF_NONE){
		FormatDesc format[] = {
			{ 0, TYPE_VERTEX,   FORMAT_FLOAT, 2 },
			{ 0, TYPE_TEXCOORD, FORMAT_FLOAT, 2 },
		};
		texVF = addVertexFormat(format, elementsOf(format));
	}

	reset();
	setTexture(texture);
	setBlendState(blendState);
	setDepthState(depthState);
	setVertexFormat(texVF);
	setVertexBuffer(0, vertices);
	apply();

	if (color) glColor4fv(*color); else glColor3f(1, 1, 1);

	glDrawArrays(glPrim[primitives], 0, nVertices);

	nDrawCalls++;
}

void OpenGLRenderer::flush(){
	glFlush();
}

void OpenGLRenderer::finish(){
	glFinish();
}

void OpenGLRenderer::changeFrontFace(const GLenum frontFace){
	if (frontFace != currentFrontFace){
		glFrontFace(currentFrontFace = frontFace);
	}
}

void OpenGLRenderer::setupSampler(GLenum glTarget, const SamplerState &ss){
	// Set requested wrapping modes
	glTexParameteri(glTarget, GL_TEXTURE_WRAP_S, ss.wrapS);
	if (glTarget != GL_TEXTURE_1D) glTexParameteri(glTarget, GL_TEXTURE_WRAP_T, ss.wrapT);
	if (glTarget == GL_TEXTURE_3D) glTexParameteri(glTarget, GL_TEXTURE_WRAP_R, ss.wrapR);

	// Set requested filter modes
	glTexParameteri(glTarget, GL_TEXTURE_MAG_FILTER, ss.magFilter);
	glTexParameteri(glTarget, GL_TEXTURE_MIN_FILTER, ss.minFilter);

	// Setup anisotropic filtering
	if (ss.aniso > 1 && /*glTarget == GL_TEXTURE_2D && */GL_EXT_texture_filter_anisotropic_supported){
		glTexParameteri(glTarget, GL_TEXTURE_MAX_ANISOTROPY_EXT, ss.aniso);
	}
}
