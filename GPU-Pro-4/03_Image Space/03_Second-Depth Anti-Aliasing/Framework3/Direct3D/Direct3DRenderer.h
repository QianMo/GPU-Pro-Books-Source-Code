
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

#ifndef _DIRECT3DRENDERER_H_
#define _DIRECT3DRENDERER_H_

#include "../Renderer.h"
#include <d3d9.h>
#include <d3dx9.h>

#define R2VB 0x1000

#define VB_INVALID (-2)

#define FB_DEPTH (-2)

class Direct3DRenderer : public Renderer {
public:
	Direct3DRenderer(const LPDIRECT3DDEVICE9 d3ddev, const D3DCAPS9 &d3dcaps);
	~Direct3DRenderer();

	void resetToDefaults();
	void reset(const uint flags = RESET_ALL);
	void setD3Ddefaults();

	bool resetDevice(D3DPRESENT_PARAMETERS &d3dpp);

	TextureID addTexture(Image &img, const SamplerStateID samplerState = SS_NONE, uint flags = 0);

	TextureID addRenderTarget(const int width, const int height, const int depth, const int mipMapCount, const int arraySize, const FORMAT format, const int msaaSamples = 1, const SamplerStateID samplerState = SS_NONE, uint flags = 0);
	TextureID addRenderDepth(const int width, const int height, const int arraySize, const FORMAT format, const int msaaSamples = 1, const SamplerStateID samplerState = SS_NONE, uint flags = 0);

	bool resizeRenderTarget(const TextureID renderTarget, const int width, const int height, const int depth, const int mipMapCount, const int arraySize);
	bool generateMipMaps(const TextureID renderTarget);

	void removeTexture(const TextureID texture);

	ShaderID addShader(const char *vsText, const char *gsText, const char *fsText, const int vsLine, const int gsLine, const int fsLine,
		const char *header = NULL, const char *extra = NULL, const char *fileName = NULL, const char **attributeNames = NULL, const int nAttributes = 0, const uint flags = 0);
	VertexFormatID addVertexFormat(const FormatDesc *formatDesc, const uint nAttribs, const ShaderID shader = SHADER_NONE);
	VertexBufferID addVertexBuffer(const long size, const BufferAccess bufferAccess, const void *data = NULL);
	IndexBufferID addIndexBuffer(const uint nIndices, const uint indexSize, const BufferAccess bufferAccess, const void *data = NULL);

	SamplerStateID addSamplerState(const Filter filter, const AddressMode s, const AddressMode t, const AddressMode r, const float lod = 0, const uint maxAniso = 16, const int compareFunc = 0, const float *border_color = NULL);
	BlendStateID addBlendState(const int srcFactorRGB, const int destFactorRGB, const int srcFactorAlpha, const int destFactorAlpha, const int blendModeRGB, const int blendModeAlpha, const int mask = ALL, const bool alphaToCoverage = false);
	DepthStateID addDepthState(const bool depthTest, const bool depthWrite, const int depthFunc, const bool stencilTest, const uint8 stencilReadMask, const uint8 stencilWriteMask,
		const int stencilFuncFront, const int stencilFuncBack, const int stencilFailFront, const int stencilFailBack,
		const int depthFailFront, const int depthFailBack, const int stencilPassFront, const int stencilPassBack);
	RasterizerStateID addRasterizerState(const int cullMode, const int fillMode = SOLID, const bool multiSample = true, const bool scissor = false);

//	int getTextureUnit(const ShaderID shader, const char *textureName) const;
	int getSamplerUnit(const ShaderID shader, const char *samplerName) const;

	void setTexture(const char *textureName, const TextureID texture);
	void setTexture(const char *textureName, const TextureID texture, const SamplerStateID samplerState);
	void setTextureSlice(const char *textureName, const TextureID texture, const int slice);
	void applyTextures();

	void setSamplerState(const char *samplerName, const SamplerStateID samplerState);
	void applySamplerStates();

	void setShaderConstantRaw(const char *name, const void *data, const int size);
	void applyConstants();

//	void changeTexture(const uint imageUnit, const TextureID texture);
	void changeRenderTargets(const TextureID *colorRTs, const uint nRenderTargets, const TextureID depthRT = TEXTURE_NONE, const int depthSlice = NO_SLICE, const int *slices = NULL);
	void changeToMainFramebuffer();
	void changeShader(const ShaderID shader);
	void changeVertexFormat(const VertexFormatID vertexFormat);
	void changeVertexBuffer(const int stream, const VertexBufferID vertexBuffer, const intptr offset = 0);
	void changeIndexBuffer(const IndexBufferID indexBuffer);

//	void changeSamplerState(const uint samplerUnit, const SamplerStateID samplerState);
	void changeBlendState(const BlendStateID blendState, const uint sampleMask = ~0);
	void changeDepthState(const DepthStateID depthState, const uint stencilRef = 0);
	void changeRasterizerState(const RasterizerStateID rasterizerState);


	void changeShaderConstant1i(const char *name, const int constant);
	void changeShaderConstant1f(const char *name, const float constant);
	void changeShaderConstant2f(const char *name, const vec2 &constant);
	void changeShaderConstant3f(const char *name, const vec3 &constant);
	void changeShaderConstant4f(const char *name, const vec4 &constant);
	void changeShaderConstant3x3f(const char *name, const mat3 &constant);
	void changeShaderConstant4x4f(const char *name, const mat4 &constant);
	void changeShaderConstantArray1f(const char *name, const float *constant, const uint count);
	void changeShaderConstantArray2f(const char *name, const vec2 *constant, const uint count);
	void changeShaderConstantArray3f(const char *name, const vec3 *constant, const uint count);
	void changeShaderConstantArray4f(const char *name, const vec4 *constant, const uint count);

	void clear(const bool clearColor, const bool clearDepth, const bool clearStencil, const float *color = NULL, const float depth = 1.0f, const uint stencil = 0);

	void drawArrays(const Primitives primitives, const int firstVertex, const int nVertices);
	void drawElements(const Primitives primitives, const int firstIndex, const int nIndices, const int firstVertex, const int nVertices);

	void setup2DMode(const float left, const float right, const float top, const float bottom);
	void drawPlain(const Primitives primitives, vec2 *vertices, const uint nVertices, const BlendStateID blendState, const DepthStateID depthState, const vec4 *color = NULL);
	void drawTextured(const Primitives primitives, TexVertex *vertices, const uint nVertices, const TextureID texture, const SamplerStateID samplerState, const BlendStateID blendState, const DepthStateID depthState, const vec4 *color = NULL);

	LPDIRECT3DBASETEXTURE9 getD3DTexture(const TextureID texture) const;

	void flush();
	void finish();

protected:
	bool createFrameBufferSurfaces();
	bool releaseFrameBufferSurfaces();

	LPDIRECT3DDEVICE9 dev;

//	TextureFilter *currentFilters;

	LPDIRECT3DSURFACE9 fbColor;
	LPDIRECT3DSURFACE9 fbDepth;

	TextureID currentTextures[MAX_TEXTUREUNIT], selectedTextures[MAX_TEXTUREUNIT];
	SamplerStateID currentSamplerStates[MAX_SAMPLERSTATE], selectedSamplerStates[MAX_SAMPLERSTATE];

	int currentSrcFactorRGB;
	int currentDstFactorRGB;
	int currentBlendModeRGB;
	int currentSrcFactorAlpha;
	int currentDstFactorAlpha;
	int currentBlendModeAlpha;
	int currentMask;
	bool currentBlendEnable;

	int currentDepthFunc;
	bool currentDepthTestEnable;
	bool currentDepthWriteEnable;


	bool currentMultiSampleEnable;
	bool currentScissorEnable;
	int currentCullMode;
	int currentFillMode;

	float4 vsRegs[256];
	float4 psRegs[224];
	int minVSDirty, maxVSDirty;
	int minPSDirty, maxPSDirty;

	ShaderID plainShader, texShader;
	VertexFormatID plainVF, texVF;
	float4 scaleBias2D;

	LPDIRECT3DQUERY9 eventQuery;
};

#endif // _DIRECT3DRENDERER_H_
