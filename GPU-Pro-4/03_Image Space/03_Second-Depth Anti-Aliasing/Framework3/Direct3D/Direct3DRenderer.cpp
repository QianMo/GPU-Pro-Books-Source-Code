
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

#include "Direct3DRenderer.h"
#include "../Util/String.h"

struct Texture {
	LPDIRECT3DBASETEXTURE9 texture;
	LPDIRECT3DSURFACE9 *surfaces;
	int width, height;
	uint flags;

	DWORD usage;
	FORMAT format;
	bool mipMapped;
//	TextureFilter filter;
//	float lod;
};

struct Constant {
	char *name;
	int vsReg;
	int psReg;
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
	LPDIRECT3DVERTEXSHADER9 vertexShader;
	LPDIRECT3DPIXELSHADER9 pixelShader;
	ID3DXConstantTable *vsConstants;
	ID3DXConstantTable *psConstants;

	Constant *constants;
	Sampler *samplers;

	uint nConstants;
	uint nSamplers;
};

struct VertexFormat {
	LPDIRECT3DVERTEXDECLARATION9 vertexDecl;
	uint vertexSize[MAX_VERTEXSTREAM];
};

struct VertexBuffer {
	LPDIRECT3DVERTEXBUFFER9 vertexBuffer;
	long size;
	DWORD usage;
};

struct IndexBuffer {
	LPDIRECT3DINDEXBUFFER9 indexBuffer;
	uint nIndices;
	uint indexSize;
	DWORD usage;
};

struct SamplerState {
	D3DTEXTUREFILTERTYPE minFilter;
	D3DTEXTUREFILTERTYPE magFilter;
	D3DTEXTUREFILTERTYPE mipFilter;
	D3DTEXTUREADDRESS wrapS;
	D3DTEXTUREADDRESS wrapT;
	D3DTEXTUREADDRESS wrapR;
	DWORD maxAniso;
	float lod;
	D3DCOLOR borderColor;
};

struct BlendState {
	int srcFactorRGB;
	int dstFactorRGB;
	int blendModeRGB;
	int srcFactorAlpha;
	int dstFactorAlpha;
	int blendModeAlpha;
	int mask;
	bool blendEnable;
};

struct DepthState {
	int depthFunc;
	bool depthTest;
	bool depthWrite;
};

struct RasterizerState {
	int cullMode;
	int fillMode;
	bool multiSample;
	bool scissor;
};

// Blending constants
const int ZERO                = D3DBLEND_ZERO;
const int ONE                 = D3DBLEND_ONE;
const int SRC_COLOR           = D3DBLEND_SRCCOLOR;
const int ONE_MINUS_SRC_COLOR = D3DBLEND_INVSRCCOLOR;
const int DST_COLOR           = D3DBLEND_DESTCOLOR;
const int ONE_MINUS_DST_COLOR = D3DBLEND_INVDESTCOLOR;
const int SRC_ALPHA           = D3DBLEND_SRCALPHA;
const int ONE_MINUS_SRC_ALPHA = D3DBLEND_INVSRCALPHA;
const int DST_ALPHA           = D3DBLEND_DESTALPHA;
const int ONE_MINUS_DST_ALPHA = D3DBLEND_INVDESTALPHA;
const int SRC_ALPHA_SATURATE  = D3DBLEND_SRCALPHASAT;

const int BM_ADD              = D3DBLENDOP_ADD;
const int BM_SUBTRACT         = D3DBLENDOP_SUBTRACT;
const int BM_REVERSE_SUBTRACT = D3DBLENDOP_REVSUBTRACT;
const int BM_MIN              = D3DBLENDOP_MIN;
const int BM_MAX              = D3DBLENDOP_MAX;

// Depth-test constants
const int NEVER    = D3DCMP_NEVER;
const int LESS     = D3DCMP_LESS;
const int EQUAL    = D3DCMP_EQUAL;
const int LEQUAL   = D3DCMP_LESSEQUAL;
const int GREATER  = D3DCMP_GREATER;
const int NOTEQUAL = D3DCMP_NOTEQUAL;
const int GEQUAL   = D3DCMP_GREATEREQUAL;
const int ALWAYS   = D3DCMP_ALWAYS;

// Stencil-test constants
const int KEEP     = D3DSTENCILOP_KEEP;
const int SET_ZERO = D3DSTENCILOP_ZERO;
const int REPLACE  = D3DSTENCILOP_REPLACE;
const int INVERT   = D3DSTENCILOP_INVERT;
const int INCR     = D3DSTENCILOP_INCR;
const int DECR     = D3DSTENCILOP_DECR;
const int INCR_SAT = D3DSTENCILOP_INCRSAT;
const int DECR_SAT = D3DSTENCILOP_DECRSAT;

// Culling constants
const int CULL_NONE  = D3DCULL_NONE;
const int CULL_BACK  = D3DCULL_CCW;
const int CULL_FRONT = D3DCULL_CW;

// Fillmode constants
const int SOLID = D3DFILL_SOLID;
const int WIREFRAME = D3DFILL_WIREFRAME;

static D3DFORMAT formats[] = {
	D3DFMT_UNKNOWN,

	// Unsigned formats
	D3DFMT_L8,
	D3DFMT_A8L8,
	D3DFMT_X8R8G8B8,
	D3DFMT_A8R8G8B8,

	D3DFMT_L16,
	D3DFMT_G16R16,
	D3DFMT_UNKNOWN, // RGB16 not directly supported
	D3DFMT_A16B16G16R16,

	// Signed formats
	D3DFMT_UNKNOWN,
	D3DFMT_V8U8,
	D3DFMT_UNKNOWN,
	D3DFMT_Q8W8V8U8,

	D3DFMT_UNKNOWN,
	D3DFMT_V16U16,
	D3DFMT_UNKNOWN,
	D3DFMT_Q16W16V16U16,

	// Float formats
	D3DFMT_R16F,
	D3DFMT_G16R16F,
	D3DFMT_UNKNOWN, // RGB16F not directly supported
	D3DFMT_A16B16G16R16F,

	D3DFMT_R32F,
	D3DFMT_G32R32F,
	D3DFMT_UNKNOWN, // RGB32F not directly supported
	D3DFMT_A32B32G32R32F,

	// Signed integer formats
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,

	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,

	// Unsigned integer formats
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,

	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,
	D3DFMT_UNKNOWN,

	// Packed formats
	D3DFMT_UNKNOWN, // RGBE8 not directly supported
	D3DFMT_UNKNOWN, // RGB9E5 not supported
	D3DFMT_UNKNOWN, // RG11B10F not supported
	D3DFMT_R5G6B5,
	D3DFMT_A4R4G4B4,
	D3DFMT_A2B10G10R10,

	// Depth formats
	D3DFMT_D16,
	D3DFMT_D24X8,
	D3DFMT_D24S8,
	D3DFMT_D32F_LOCKABLE,

	// Compressed formats
	D3DFMT_DXT1,
	D3DFMT_DXT3,
	D3DFMT_DXT5,
	(D3DFORMAT) '1ITA', // 3Dc 1 channel
	(D3DFORMAT) '2ITA', // 3Dc 2 channels
};

Direct3DRenderer::Direct3DRenderer(const LPDIRECT3DDEVICE9 d3ddev, const D3DCAPS9 &d3dcaps) : Renderer(){
	dev = d3ddev;

	createFrameBufferSurfaces();
	
	if (d3dcaps.PixelShaderVersion >= D3DPS_VERSION(1,0)){
		nImageUnits = 4;
		if (d3dcaps.PixelShaderVersion >= D3DPS_VERSION(1,4)) nImageUnits = 6;
		if (d3dcaps.PixelShaderVersion >= D3DPS_VERSION(2,0)) nImageUnits = 16;
	} else {
		nImageUnits = d3dcaps.MaxSimultaneousTextures;
	}

	maxAnisotropic = d3dcaps.MaxAnisotropy;


	nMRTs = d3dcaps.NumSimultaneousRTs;
	if (nMRTs > MAX_MRTS) nMRTs = MAX_MRTS;

	plainShader = SHADER_NONE;
	plainVF = VF_NONE;
	texShader = SHADER_NONE;
	texVF = VF_NONE;

	eventQuery = NULL;

	setD3Ddefaults();
	resetToDefaults();
}

Direct3DRenderer::~Direct3DRenderer(){
	if (eventQuery) eventQuery->Release();

	releaseFrameBufferSurfaces();


	// Delete shaders
	for (uint i = 0; i < shaders.getCount(); i++){
		if (shaders[i].vertexShader) shaders[i].vertexShader->Release();
		if (shaders[i].pixelShader) shaders[i].pixelShader->Release();
		if (shaders[i].vsConstants) shaders[i].vsConstants->Release();
		if (shaders[i].psConstants) shaders[i].psConstants->Release();

		for (uint j = 0; j < shaders[i].nSamplers; j++){
			delete shaders[i].samplers[j].name;
		}
		for (uint j = 0; j < shaders[i].nConstants; j++){
			delete shaders[i].constants[j].name;
		}
		delete shaders[i].samplers;
		delete shaders[i].constants;
	}

    // Delete vertex formats
	for (uint i = 0; i < vertexFormats.getCount(); i++){
		if (vertexFormats[i].vertexDecl) vertexFormats[i].vertexDecl->Release();
	}

    // Delete vertex buffers
	for (uint i = 0; i < vertexBuffers.getCount(); i++){
		if (vertexBuffers[i].vertexBuffer) vertexBuffers[i].vertexBuffer->Release();
	}

	// Delete index buffers
	for (uint i = 0; i < indexBuffers.getCount(); i++){
		if (indexBuffers[i].indexBuffer) indexBuffers[i].indexBuffer->Release();
	}

	// Delete textures
	for (uint i = 0; i < textures.getCount(); i++){
		removeTexture(i);
	}
}

bool Direct3DRenderer::createFrameBufferSurfaces(){
	if (dev->GetRenderTarget(0, &fbColor) != D3D_OK) return false;
	dev->GetDepthStencilSurface(&fbDepth);
	return true;
}

bool Direct3DRenderer::releaseFrameBufferSurfaces(){
	if (fbColor) fbColor->Release();
	if (fbDepth) fbDepth->Release();
	return true;
}

void Direct3DRenderer::resetToDefaults(){
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

	currentDepthFunc = LESS;
	currentDepthTestEnable = true;
	currentDepthWriteEnable = true;

	currentCullMode = CULL_NONE;
	currentFillMode = SOLID;
	currentMultiSampleEnable = true;
	currentScissorEnable = false;

	currentDepthRT = FB_DEPTH;

	memset(vsRegs, 0, sizeof(vsRegs));
	memset(psRegs, 0, sizeof(psRegs));
	minVSDirty = 256;
	maxVSDirty = -1;
	minPSDirty = 224;
	maxPSDirty = -1;
}

void Direct3DRenderer::reset(const uint flags){
	Renderer::reset(flags);

	if (flags & RESET_TEX){
		for (uint i = 0; i < MAX_TEXTUREUNIT; i++){
			selectedTextures[i] = TEXTURE_NONE;
		}
	}

	if (flags & RESET_SS){
		for (uint i = 0; i < MAX_SAMPLERSTATE; i++){
			selectedSamplerStates[i] = SS_NONE;
		}
	}
}

void Direct3DRenderer::setD3Ddefaults(){
	// Set some of my preferred defaults
	dev->SetRenderState(D3DRS_LIGHTING, FALSE);
	dev->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
	dev->SetRenderState(D3DRS_SEPARATEALPHABLENDENABLE, TRUE);

	dev->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_MODULATE);

	if (maxAnisotropic > 1){
		for (uint i = 0; i < nImageUnits; i++){
			dev->SetSamplerState(i, D3DSAMP_MAXANISOTROPY, maxAnisotropic);
		}
	}
}

bool createRenderTarget(LPDIRECT3DDEVICE9 dev, Texture &tex){
	if (isDepthFormat(tex.format)){
		if (tex.surfaces == NULL) tex.surfaces = new LPDIRECT3DSURFACE9;
		if (dev->CreateDepthStencilSurface(tex.width, tex.height, formats[tex.format], D3DMULTISAMPLE_NONE, 0, FALSE, tex.surfaces, NULL) != D3D_OK){
			delete tex.surfaces;

			ErrorMsg("Couldn't create depth surface");
			return false;
		}
	} else {
		if (tex.flags & CUBEMAP){
			if (dev->CreateCubeTexture(tex.width, tex.mipMapped? 0 : 1, tex.usage, formats[tex.format], D3DPOOL_DEFAULT, (LPDIRECT3DCUBETEXTURE9 *) &tex.texture, NULL) != D3D_OK){
				ErrorMsg("Couldn't create render target");
				return false;
			}

			if (tex.surfaces == NULL) tex.surfaces = new LPDIRECT3DSURFACE9[6];
			for (uint i = 0; i < 6; i++){
				((LPDIRECT3DCUBETEXTURE9) tex.texture)->GetCubeMapSurface((D3DCUBEMAP_FACES) i, 0, &tex.surfaces[i]);
			}
		} else {
			if (dev->CreateTexture(tex.width, tex.height, tex.mipMapped? 0 : 1, tex.usage, formats[tex.format], D3DPOOL_DEFAULT, (LPDIRECT3DTEXTURE9 *) &tex.texture, NULL) != D3D_OK){
				ErrorMsg("Couldn't create render target");
				return false;
			}
			if (tex.surfaces == NULL) tex.surfaces = new LPDIRECT3DSURFACE9;
			((LPDIRECT3DTEXTURE9) tex.texture)->GetSurfaceLevel(0, tex.surfaces);
		}
	}

	return true;
}

bool Direct3DRenderer::resetDevice(D3DPRESENT_PARAMETERS &d3dpp){
	for (uint i = 0; i < textures.getCount(); i++){
		if (textures[i].surfaces){
			int n = (textures[i].flags & CUBEMAP)? 6 : 1;

			if (textures[i].texture) textures[i].texture->Release();
			for (int k = 0; k < n; k++){
				textures[i].surfaces[k]->Release();
			}
		}
	}

	if (!releaseFrameBufferSurfaces()) return false;

	if (dev->Reset(&d3dpp) != D3D_OK){
		ErrorMsg("Device reset failed");
		return false;
	}

	if (!createFrameBufferSurfaces()) return false;

	resetToDefaults();
	setD3Ddefaults();

	for (uint i = 0; i < textures.getCount(); i++){
		if (textures[i].surfaces){
			createRenderTarget(dev, textures[i]);
		}
	}

	return true;
}

// Returns the closest power of two
int getPow2(const int x){
	int i = 1;
	while (i < x) i += i;

	if (4 * x < 3 * i) i >>= 1;
	return i;
}

TextureID Direct3DRenderer::addTexture(Image &img, const SamplerStateID samplerState, uint flags){
	Texture tex;
	memset(&tex, 0, sizeof(tex));
	tex.mipMapped = (img.getMipMapCount() > 1);

	FORMAT format = img.getFormat();
	if (img.isCube()){
		if (dev->CreateCubeTexture(img.getWidth(), img.getMipMapCount(), 0, formats[format], D3DPOOL_MANAGED, (LPDIRECT3DCUBETEXTURE9 *) &tex.texture, NULL) != D3D_OK){
			ErrorMsg("Couldn't create cubemap");
			return false;
		}
	} else if (img.is3D()){
		if (dev->CreateVolumeTexture(img.getWidth(), img.getHeight(), img.getDepth(), img.getMipMapCount(), 0, formats[format], D3DPOOL_MANAGED, (LPDIRECT3DVOLUMETEXTURE9 *) &tex.texture, NULL) != D3D_OK){
			ErrorMsg("Couldn't create volumetric texture");
			return false;
		}
	} else {
		D3DFORMAT form = formats[format];


		if (dev->CreateTexture(img.getWidth(), img.getHeight(), img.getMipMapCount(), 0, formats[format], D3DPOOL_MANAGED, (LPDIRECT3DTEXTURE9 *) &tex.texture, NULL) != D3D_OK){
			ErrorMsg("Couldn't create texture");
			return false;
		}
	}

	if (format == FORMAT_RGB8) img.convert(FORMAT_RGBA8);
	if (format == FORMAT_RGB8 || format == FORMAT_RGBA8) img.swap(0, 2);

	unsigned char *src;
	int mipMapLevel = 0;
	while ((src = img.getPixels(mipMapLevel)) != NULL){
		int size = img.getMipMappedSize(mipMapLevel, 1);

		if (img.is3D()){
			D3DLOCKED_BOX box;
			if (((LPDIRECT3DVOLUMETEXTURE9) tex.texture)->LockBox(mipMapLevel, &box, NULL, 0) == D3D_OK){
				memcpy(box.pBits, src, size);
				((LPDIRECT3DVOLUMETEXTURE9) tex.texture)->UnlockBox(mipMapLevel);
			}
		} else if (img.isCube()){
			size /= 6;

			D3DLOCKED_RECT rect;
			for (int i = 0; i < 6; i++){
				if (((LPDIRECT3DCUBETEXTURE9) tex.texture)->LockRect((D3DCUBEMAP_FACES) i, mipMapLevel, &rect, NULL, 0) == D3D_OK){
					memcpy(rect.pBits, src, size);
					((LPDIRECT3DCUBETEXTURE9) tex.texture)->UnlockRect((D3DCUBEMAP_FACES) i, mipMapLevel);
				}
				src += size;				
			}
		} else {
			D3DLOCKED_RECT rect;
			if (((LPDIRECT3DTEXTURE9) tex.texture)->LockRect(mipMapLevel, &rect, NULL, 0) == D3D_OK){
				memcpy(rect.pBits, src, size);
				((LPDIRECT3DTEXTURE9) tex.texture)->UnlockRect(mipMapLevel);
			}
		}
		mipMapLevel++;
	}

//	setupFilter(tex, filter, flags);
//	tex.lod = lod;

	return textures.add(tex);
}

TextureID Direct3DRenderer::addRenderTarget(const int width, const int height, const int depth, const int mipMapCount, const int arraySize, const FORMAT format, const int msaaSamples, const SamplerStateID samplerState, uint flags){
	if (depth > 1 || arraySize > 1) return TEXTURE_NONE;
	// For now ...
	if (msaaSamples > 1 || mipMapCount > 1) return TEXTURE_NONE;

	Texture tex;
	memset(&tex, 0, sizeof(tex));

	tex.width  = width;
	tex.height = height;
	tex.format = format;
	tex.usage  = D3DUSAGE_RENDERTARGET;
	tex.flags  = flags;
	tex.mipMapped = false;

	if (flags & R2VB) tex.usage |= D3DUSAGE_DMAP;

	if (createRenderTarget(dev, tex)){
//		setupFilter(tex, filter, flags);
		return textures.add(tex);
	} else {
		return TEXTURE_NONE;
	}
}

TextureID Direct3DRenderer::addRenderDepth(const int width, const int height, const int arraySize, const FORMAT format, const int msaaSamples, const SamplerStateID samplerState, uint flags){
	// For now ...
	if (msaaSamples > 1) return TEXTURE_NONE;

	Texture tex;
	memset(&tex, 0, sizeof(tex));

	tex.format = format;
	tex.width  = width;
	tex.height = height;

	if (createRenderTarget(dev, tex)){
		return textures.add(tex);
	} else {
		return TEXTURE_NONE;
	}
}

bool Direct3DRenderer::resizeRenderTarget(const TextureID renderTarget, const int width, const int height, const int depth, const int mipMapCount, const int arraySize){
	if (depth > 1 || arraySize > 1 || mipMapCount > 1) return false;

	if (textures[renderTarget].surfaces){
		int n = (textures[renderTarget].flags & CUBEMAP)? 6 : 1;

		if (textures[renderTarget].texture) textures[renderTarget].texture->Release();
		for (int k = 0; k < n; k++){
			textures[renderTarget].surfaces[k]->Release();
		}
	}
	textures[renderTarget].width  = width;
	textures[renderTarget].height = height;

	return createRenderTarget(dev, textures[renderTarget]);
}

bool Direct3DRenderer::generateMipMaps(const TextureID renderTarget){
	// TODO: Implement
	return false;
}

void Direct3DRenderer::removeTexture(const TextureID texture){
	if (textures[texture].surfaces){
		int n = (textures[texture].flags & CUBEMAP)? 6 : 1;
		for (int k = 0; k < n; k++){
			textures[texture].surfaces[k]->Release();
		}
		delete textures[texture].surfaces;
		textures[texture].surfaces = NULL;
	}
	if (textures[texture].texture){
		textures[texture].texture->Release();
		textures[texture].texture = NULL;
	}
}

ShaderID Direct3DRenderer::addShader(const char *vsText, const char *gsText, const char *fsText, const int vsLine, const int gsLine, const int fsLine,
                                     const char *header, const char *extra, const char *fileName, const char **attributeNames, const int nAttributes, const uint flags){

	if (vsText == NULL && fsText == NULL) return SHADER_NONE;

	LPD3DXBUFFER shaderBuf = NULL;
	LPD3DXBUFFER errorsBuf = NULL;

	Shader shader;
	shader.vertexShader = NULL;
	shader.pixelShader  = NULL;
	shader.vsConstants  = NULL;
	shader.psConstants  = NULL;

	if (vsText != NULL){
		String shaderString;
		if (extra  != NULL) shaderString += extra;
		if (header != NULL) shaderString += header;
		shaderString.sprintf("#line %d\n", vsLine + 1);
		shaderString += vsText;

		const char *profile = D3DXGetVertexShaderProfile(dev);

		if (D3DXCompileShader(shaderString, shaderString.getLength(), NULL, NULL, "main", profile, D3DXSHADER_DEBUG | D3DXSHADER_PACKMATRIX_ROWMAJOR, &shaderBuf, &errorsBuf, &shader.vsConstants) == D3D_OK){
			dev->CreateVertexShader((DWORD *) shaderBuf->GetBufferPointer(), &shader.vertexShader);
#ifdef DEBUG
			LPD3DXBUFFER disasm;
			D3DXDisassembleShader((DWORD *) shaderBuf->GetBufferPointer(), FALSE, NULL, &disasm);
			char *str = (char *) disasm->GetBufferPointer();

			while (*str){
				while (*str == '\n' || *str == '\r') *str++;

				char *endStr = str;
				while (*endStr && *endStr != '\n' && *endStr != '\r') *endStr++;
				if (*str != '#'){
					*endStr = '\0';
					outputDebugString(str);
				}

				str = endStr + 1;
			}

			if (disasm != NULL) disasm->Release();
#endif
		} else {
			ErrorMsg((const char *) errorsBuf->GetBufferPointer());
		}
		if (shaderBuf != NULL) shaderBuf->Release();
		if (errorsBuf != NULL) errorsBuf->Release();

		if (shader.vertexShader == NULL) return SHADER_NONE;
	}

	if (fsText != NULL){
		String shaderString;
		if (extra  != NULL) shaderString += extra;
		if (header != NULL) shaderString += header;
		shaderString.sprintf("#line %d\n", fsLine + 1);
		shaderString += fsText;

		const char *profile = D3DXGetPixelShaderProfile(dev);

		if (D3DXCompileShader(shaderString, shaderString.getLength(), NULL, NULL, "main", profile, D3DXSHADER_DEBUG | D3DXSHADER_PACKMATRIX_ROWMAJOR, &shaderBuf, &errorsBuf, &shader.psConstants) == D3D_OK){
			dev->CreatePixelShader((DWORD *) shaderBuf->GetBufferPointer(), &shader.pixelShader);
#ifdef DEBUG
			LPD3DXBUFFER disasm;
			D3DXDisassembleShader((DWORD *) shaderBuf->GetBufferPointer(), FALSE, NULL, &disasm);
			char *str = (char *) disasm->GetBufferPointer();

			while (*str){
				while (*str == '\n' || *str == '\r') *str++;

				char *endStr = str;
				while (*endStr && *endStr != '\n' && *endStr != '\r') *endStr++;
				if (*str != '#'){
					*endStr = '\0';
					outputDebugString(str);
				}

				str = endStr + 1;
			}

			if (disasm != NULL) disasm->Release();
#endif
		} else {
			ErrorMsg((const char *) errorsBuf->GetBufferPointer());
		}
		if (shaderBuf != NULL) shaderBuf->Release();
		if (errorsBuf != NULL) errorsBuf->Release();

		if (shader.pixelShader == NULL) return SHADER_NONE;
	}

	
	D3DXCONSTANTTABLE_DESC vsDesc, psDesc;
	shader.vsConstants->GetDesc(&vsDesc);
	shader.psConstants->GetDesc(&psDesc);

	uint count = vsDesc.Constants + psDesc.Constants;

	Sampler  *samplers  = (Sampler  *) malloc(count * sizeof(Sampler));
	Constant *constants = (Constant *) malloc(count * sizeof(Constant));

	uint nSamplers  = 0;
	uint nConstants = 0;

	D3DXCONSTANT_DESC cDesc;
	for (uint i = 0; i < vsDesc.Constants; i++){
		UINT count = 1;
		shader.vsConstants->GetConstantDesc(shader.vsConstants->GetConstant(NULL, i), &cDesc, &count);

		size_t length = strlen(cDesc.Name);
		if (cDesc.Type >= D3DXPT_SAMPLER && cDesc.Type <= D3DXPT_SAMPLERCUBE){
			// TODO: Vertex samplers not yet supported ...
		} else {
			constants[nConstants].name = new char[length + 1];
			strcpy(constants[nConstants].name, cDesc.Name);
			constants[nConstants].vsReg = cDesc.RegisterIndex;
			constants[nConstants].psReg = -1;
			//constants[nConstants].nElements = cDesc.RegisterCount;
			nConstants++;
		}
	}

	uint nVSConsts = nConstants;
	for (uint i = 0; i < psDesc.Constants; i++){
		UINT count = 1;
		shader.psConstants->GetConstantDesc(shader.psConstants->GetConstant(NULL, i), &cDesc, &count);

		size_t length = strlen(cDesc.Name);
		if (cDesc.Type >= D3DXPT_SAMPLER && cDesc.Type <= D3DXPT_SAMPLERCUBE){
			samplers[nSamplers].name = new char[length + 1];
			samplers[nSamplers].index = cDesc.RegisterIndex;
			strcpy(samplers[nSamplers].name, cDesc.Name);
			nSamplers++;
		} else {
			int merge = -1;
			for (uint i = 0; i < nVSConsts; i++){
				if (strcmp(constants[i].name, cDesc.Name) == 0){
					merge = i;
					break;
				}
			}

			if (merge < 0){
				constants[nConstants].name = new char[length + 1];
				strcpy(constants[nConstants].name, cDesc.Name);
				constants[nConstants].vsReg = -1;
				constants[nConstants].psReg = cDesc.RegisterIndex;
				//constants[nConstants].nElements = cDesc.RegisterCount;
			} else {
				constants[merge].psReg = cDesc.RegisterIndex;
			}
			nConstants++;
		}
	}

	// Shorten arrays to actual count
	samplers  = (Sampler  *) realloc(samplers,  nSamplers  * sizeof(Sampler));
	constants = (Constant *) realloc(constants, nConstants * sizeof(Constant));
	qsort(samplers,  nSamplers,  sizeof(Sampler),  samplerComp);
	qsort(constants, nConstants, sizeof(Constant), constantComp);

	shader.constants  = constants;
	shader.samplers   = samplers;
	shader.nConstants = nConstants;
	shader.nSamplers  = nSamplers;

	return shaders.add(shader);
}

VertexFormatID Direct3DRenderer::addVertexFormat(const FormatDesc *formatDesc, const uint nAttribs, const ShaderID shader){
	static const D3DDECLTYPE types[][4] = {
		D3DDECLTYPE_FLOAT1, D3DDECLTYPE_FLOAT2,    D3DDECLTYPE_FLOAT3, D3DDECLTYPE_FLOAT4,
		D3DDECLTYPE_UNUSED, D3DDECLTYPE_FLOAT16_2, D3DDECLTYPE_UNUSED, D3DDECLTYPE_FLOAT16_4,
		D3DDECLTYPE_UNUSED, D3DDECLTYPE_UNUSED,    D3DDECLTYPE_UNUSED, D3DDECLTYPE_UBYTE4N,
	};

	static const D3DDECLUSAGE usages[] = {
		(D3DDECLUSAGE) (-1),
		D3DDECLUSAGE_POSITION,
		D3DDECLUSAGE_TEXCOORD,
		D3DDECLUSAGE_NORMAL,
		D3DDECLUSAGE_TANGENT,
		D3DDECLUSAGE_BINORMAL,
	};

	int index[6];
	memset(index, 0, sizeof(index));

	VertexFormat vf;
	memset(vf.vertexSize, 0, sizeof(vf.vertexSize));

	D3DVERTEXELEMENT9 *vElem = new D3DVERTEXELEMENT9[nAttribs + 1];

	// Fill the vertex element array
	for (uint i = 0; i < nAttribs; i++){
		int stream = formatDesc[i].stream;
		int size = formatDesc[i].size;

		vElem[i].Stream = stream;
		vElem[i].Offset = vf.vertexSize[stream];
		vElem[i].Type = types[formatDesc[i].format][size - 1];
		vElem[i].Method = D3DDECLMETHOD_DEFAULT;
		vElem[i].Usage = usages[formatDesc[i].type];
		vElem[i].UsageIndex = index[formatDesc[i].type]++;

		vf.vertexSize[stream] += size * getFormatSize(formatDesc[i].format);
	}
	// Terminating element
	memset(vElem + nAttribs, 0, sizeof(D3DVERTEXELEMENT9));
	vElem[nAttribs].Stream = 0xFF;
	vElem[nAttribs].Type = D3DDECLTYPE_UNUSED;

	HRESULT hr = dev->CreateVertexDeclaration(vElem, &vf.vertexDecl);
	delete vElem;

	if (hr != D3D_OK){
		ErrorMsg("Couldn't create vertex declaration");
		return VF_NONE;
	}

	return vertexFormats.add(vf);
}

DWORD usages[] = {
	0,
	D3DUSAGE_DYNAMIC,
	D3DUSAGE_DYNAMIC,
};

VertexBufferID Direct3DRenderer::addVertexBuffer(const long size, const BufferAccess bufferAccess, const void *data){
	VertexBuffer vb;
	vb.size = size;
	vb.usage = usages[bufferAccess];

	bool dynamic = (vb.usage & D3DUSAGE_DYNAMIC) != 0;

	if (dev->CreateVertexBuffer(size, vb.usage, 0, dynamic? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &vb.vertexBuffer, NULL) != D3D_OK){
        ErrorMsg("Couldn't create vertex buffer");
		return VB_NONE;
	}

	if (data != NULL){
		void *dest;
		if (vb.vertexBuffer->Lock(0, size, &dest, dynamic? D3DLOCK_DISCARD : 0) == D3D_OK){
			memcpy(dest, data, size);
			vb.vertexBuffer->Unlock();
		}
	}

	return vertexBuffers.add(vb);
}

IndexBufferID Direct3DRenderer::addIndexBuffer(const uint nIndices, const uint indexSize, const BufferAccess bufferAccess, const void *data){
	IndexBuffer ib;
	ib.nIndices = nIndices;
	ib.indexSize = indexSize;
	ib.usage = usages[bufferAccess];

	bool dynamic = (ib.usage & D3DUSAGE_DYNAMIC) != 0;

	uint size = nIndices * indexSize;
	if (dev->CreateIndexBuffer(size, ib.usage, indexSize == 2? D3DFMT_INDEX16 : D3DFMT_INDEX32, dynamic? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &ib.indexBuffer, NULL) != D3D_OK){
        ErrorMsg("Couldn't create index buffer");
		return IB_NONE;
	}

	// Upload the provided index data if any
	if (data != NULL){
		void *dest;
		if (ib.indexBuffer->Lock(0, size, &dest, dynamic? D3DLOCK_DISCARD : 0) == D3D_OK){
			memcpy(dest, data, size);
			ib.indexBuffer->Unlock();
		} else {
            ErrorMsg("Couldn't lock index buffer");
		}
	}

	return indexBuffers.add(ib);
}

SamplerStateID Direct3DRenderer::addSamplerState(const Filter filter, const AddressMode s, const AddressMode t, const AddressMode r, const float lod, const uint maxAniso, const int compareFunc, const float *border_color){
	SamplerState samplerState;

	samplerState.minFilter = hasAniso(filter)? D3DTEXF_ANISOTROPIC : (filter != NEAREST)? D3DTEXF_LINEAR : D3DTEXF_POINT;
	samplerState.magFilter = hasAniso(filter)? D3DTEXF_ANISOTROPIC : (filter != NEAREST)? D3DTEXF_LINEAR : D3DTEXF_POINT;
	samplerState.mipFilter = (filter == TRILINEAR || filter == TRILINEAR_ANISO)? D3DTEXF_LINEAR : hasMipmaps(filter)? D3DTEXF_POINT : D3DTEXF_NONE;
	samplerState.wrapS = (s == WRAP)? D3DTADDRESS_WRAP : (s == CLAMP)? D3DTADDRESS_CLAMP : D3DTADDRESS_BORDER;
	samplerState.wrapT = (t == WRAP)? D3DTADDRESS_WRAP : (t == CLAMP)? D3DTADDRESS_CLAMP : D3DTADDRESS_BORDER;
	samplerState.wrapR = (r == WRAP)? D3DTADDRESS_WRAP : (r == CLAMP)? D3DTADDRESS_CLAMP : D3DTADDRESS_BORDER;
	samplerState.maxAniso = min((uint) maxAnisotropic, maxAniso);
	samplerState.lod = lod;
	if (border_color)
	{
		samplerState.borderColor = D3DCOLOR_ARGB(
			uint32(border_color[3] * 255.0f + 0.5f),
			uint32(border_color[0] * 255.0f + 0.5f),
			uint32(border_color[1] * 255.0f + 0.5f),
			uint32(border_color[2] * 255.0f + 0.5f));
	}
	else
	{
		samplerState.borderColor = 0;
	}

	return samplerStates.add(samplerState);
}

BlendStateID Direct3DRenderer::addBlendState(const int srcFactorRGB, const int destFactorRGB, const int srcFactorAlpha, const int destFactorAlpha, const int blendModeRGB, const int blendModeAlpha, const int mask, const bool alphaToCoverage){
	BlendState blendState;

	blendState.srcFactorRGB = srcFactorRGB;
	blendState.dstFactorRGB = destFactorRGB;
	blendState.blendModeRGB = blendModeRGB;
	blendState.srcFactorAlpha = srcFactorAlpha;
	blendState.dstFactorAlpha = destFactorAlpha;
	blendState.blendModeAlpha = blendModeAlpha;
	blendState.mask = mask;
	blendState.blendEnable = (srcFactorRGB != ONE || destFactorRGB != ZERO || srcFactorAlpha != ONE || destFactorAlpha != ZERO);

	return blendStates.add(blendState);
}

DepthStateID Direct3DRenderer::addDepthState(const bool depthTest, const bool depthWrite, const int depthFunc, const bool stencilTest, const uint8 stencilReadMask, const uint8 stencilWriteMask,
		const int stencilFuncFront, const int stencilFuncBack, const int stencilFailFront, const int stencilFailBack,
		const int depthFailFront, const int depthFailBack, const int stencilPassFront, const int stencilPassBack){

	// TODO: Add stencil support...
	DepthState depthState;

	depthState.depthTest  = depthTest;
	depthState.depthWrite = depthWrite;
	depthState.depthFunc  = depthFunc;

	return depthStates.add(depthState);
}

RasterizerStateID Direct3DRenderer::addRasterizerState(const int cullMode, const int fillMode, const bool multiSample, const bool scissor){
	RasterizerState rasterizerState;

	rasterizerState.cullMode = cullMode;
	rasterizerState.fillMode = fillMode;
	rasterizerState.multiSample = multiSample;
	rasterizerState.scissor = scissor;

	return rasterizerStates.add(rasterizerState);
}

int Direct3DRenderer::getSamplerUnit(const ShaderID shader, const char *samplerName) const {
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

void Direct3DRenderer::setTexture(const char *textureName, const TextureID texture){
	int unit = getSamplerUnit(selectedShader, textureName);
	if (unit >= 0){
		selectedTextures[unit] = texture;
	}
}

void Direct3DRenderer::setTexture(const char *textureName, const TextureID texture, const SamplerStateID samplerState){
	int unit = getSamplerUnit(selectedShader, textureName);
	if (unit >= 0){
		selectedTextures[unit] = texture;
		selectedSamplerStates[unit] = samplerState;
	}
}

void Direct3DRenderer::setTextureSlice(const char *textureName, const TextureID texture, const int slice){
	ASSERT(0);
}

void Direct3DRenderer::applyTextures(){
	for (uint i = 0; i < MAX_TEXTUREUNIT; i++){
		TextureID texture = selectedTextures[i];
		if (texture != currentTextures[i]){
			if (texture == TEXTURE_NONE){
				dev->SetTexture(i, NULL);
			} else {
				dev->SetTexture(i, textures[texture].texture);
			}
			currentTextures[i] = texture;
		}
	}
}

void Direct3DRenderer::setSamplerState(const char *samplerName, const SamplerStateID samplerState){
	int unit = getSamplerUnit(selectedShader, samplerName);
	if (unit >= 0){
		selectedSamplerStates[unit] = samplerState;
	}
}

void Direct3DRenderer::applySamplerStates(){
	for (uint i = 0; i < MAX_SAMPLERSTATE; i++){
		SamplerStateID samplerState = selectedSamplerStates[i];
		if (samplerState != currentSamplerStates[i]){
			SamplerState ss, css;

			if (samplerState == SS_NONE){
				ss.magFilter = D3DTEXF_POINT;
				ss.minFilter = D3DTEXF_POINT;
				ss.mipFilter = D3DTEXF_NONE;
				ss.wrapS = D3DTADDRESS_WRAP;
				ss.wrapT = D3DTADDRESS_WRAP;
				ss.wrapR = D3DTADDRESS_WRAP;
				ss.maxAniso = 1;
				ss.lod = 0.0f;
				ss.borderColor = 0xFFFFFFFF;
			} else {
				ss = samplerStates[samplerState];
			}
			if (currentSamplerStates[i] == SS_NONE){
				css.magFilter = D3DTEXF_POINT;
				css.minFilter = D3DTEXF_POINT;
				css.mipFilter = D3DTEXF_NONE;
				css.wrapS = D3DTADDRESS_WRAP;
				css.wrapT = D3DTADDRESS_WRAP;
				css.wrapR = D3DTADDRESS_WRAP;
				css.maxAniso = 1;
				css.lod = 0.0f;
				css.borderColor = 0xFFFFFFFF;
			} else {
				css = samplerStates[currentSamplerStates[i]];
			}

			if (ss.minFilter != css.minFilter) dev->SetSamplerState(i, D3DSAMP_MINFILTER, ss.minFilter);
			if (ss.magFilter != css.magFilter) dev->SetSamplerState(i, D3DSAMP_MAGFILTER, ss.magFilter);
			if (ss.mipFilter != css.mipFilter) dev->SetSamplerState(i, D3DSAMP_MIPFILTER, ss.mipFilter);

			if (ss.wrapS != css.wrapS) dev->SetSamplerState(i, D3DSAMP_ADDRESSU, ss.wrapS);
			if (ss.wrapT != css.wrapT) dev->SetSamplerState(i, D3DSAMP_ADDRESSV, ss.wrapT);
			if (ss.wrapR != css.wrapR) dev->SetSamplerState(i, D3DSAMP_ADDRESSW, ss.wrapR);

			if (ss.maxAniso != css.maxAniso) dev->SetSamplerState(i, D3DSAMP_MAXANISOTROPY, ss.maxAniso);

			if (ss.lod != css.lod) dev->SetSamplerState(i, D3DSAMP_MIPMAPLODBIAS, *(DWORD *) &ss.lod);

			if (ss.borderColor != css.borderColor) dev->SetSamplerState(i, D3DSAMP_BORDERCOLOR, ss.borderColor);

			currentSamplerStates[i] = samplerState;
		}
	}
}


void Direct3DRenderer::setShaderConstantRaw(const char *name, const void *data, const int size){
	int minConstant = 0;
	int maxConstant = shaders[selectedShader].nConstants - 1;
	Constant *constants = shaders[selectedShader].constants;

	// Do a quick lookup in the sorted table with a binary search
	while (minConstant <= maxConstant){
		int currConstant = (minConstant + maxConstant) >> 1;
		int res = strcmp(name, constants[currConstant].name);
		if (res == 0){
			Constant *c = constants + currConstant;

			if (c->vsReg >= 0){
				if (memcmp(vsRegs + c->vsReg, data, size)){
					memcpy(vsRegs + c->vsReg, data, size);
					
					int r0 = c->vsReg;
					int r1 = c->vsReg + ((size + 15) >> 4);
					
					if (r0 < minVSDirty) minVSDirty = r0;
					if (r1 > maxVSDirty) maxVSDirty = r1;
				}
			}

			if (c->psReg >= 0){
				if (memcmp(psRegs + c->psReg, data, size)){
					memcpy(psRegs + c->psReg, data, size);
					
					int r0 = c->psReg;
					int r1 = c->psReg + ((size + 15) >> 4);
					
					if (r0 < minPSDirty) minPSDirty = r0;
					if (r1 > maxPSDirty) maxPSDirty = r1;
				}
			}
			return;

		} else if (res > 0){
			minConstant = currConstant + 1;
		} else {
			maxConstant = currConstant - 1;
		}
	}

#ifdef _DEBUG
	char str[256];
	sprintf(str, "Invalid constant \"%s\"", name);
	outputDebugString(str);
#endif
}

void Direct3DRenderer::applyConstants(){
//	if (currentShader != SHADER_NONE){
		if (minVSDirty < maxVSDirty){
			dev->SetVertexShaderConstantF(minVSDirty, (const float *) (vsRegs + minVSDirty), maxVSDirty - minVSDirty);
			minVSDirty = 256;
			maxVSDirty = -1;
		}
		if (minPSDirty < maxPSDirty){
			dev->SetPixelShaderConstantF(minPSDirty, (const float *) (psRegs + minPSDirty), maxPSDirty - minPSDirty);
			minPSDirty = 224;
			maxPSDirty = -1;
		}
//	}
}

void Direct3DRenderer::changeRenderTargets(const TextureID *colorRTs, const uint nRenderTargets, const TextureID depthRT, const int depthSlice, const int *slices){
	for (uint i = 0; i < nRenderTargets; i++){
		TextureID rt = colorRTs[i];
		int face = (slices != NULL)? slices[i] : 0;

		if (rt != currentColorRT[i] || face != currentColorRTSlice[i]){
			if (face == NO_SLICE){
				dev->SetRenderTarget(i, textures[rt].surfaces[0]);
			} else {
				dev->SetRenderTarget(i, textures[rt].surfaces[face]);
			}

			currentColorRT[i] = rt;
			currentColorRTSlice[i] = face;
		}
	}

	for (uint i = nRenderTargets; i < nMRTs; i++){
		if (currentColorRT[i] != TEXTURE_NONE){
			dev->SetRenderTarget(i, NULL);
			currentColorRT[i] = TEXTURE_NONE;
		}
	}

	if (depthRT != currentDepthRT){
		if (depthRT == TEXTURE_NONE){
			dev->SetDepthStencilSurface(NULL);
		} else if (depthRT == FB_DEPTH){
			dev->SetDepthStencilSurface(fbDepth);
		} else {
			dev->SetDepthStencilSurface(textures[depthRT].surfaces[0]);
		}
		currentDepthRT = depthRT;
	}
}

void Direct3DRenderer::changeToMainFramebuffer(){
	if (currentColorRT[0] != TEXTURE_NONE){
		dev->SetRenderTarget(0, fbColor);
		currentColorRT[0] = TEXTURE_NONE;
	}

	for (uint i = 1; i < nMRTs; i++){
		if (currentColorRT[i] != TEXTURE_NONE){
			dev->SetRenderTarget(i, NULL);
			currentColorRT[i] = TEXTURE_NONE;
		}
	}

	if (currentDepthRT != FB_DEPTH){
		dev->SetDepthStencilSurface(fbDepth);
		currentDepthRT = FB_DEPTH;
	}
}

void Direct3DRenderer::changeShader(const ShaderID shader){
	if (shader != currentShader){
		if (shader == SHADER_NONE){
			dev->SetVertexShader(NULL);
			dev->SetPixelShader(NULL);
		} else {
			dev->SetVertexShader(shaders[shader].vertexShader);
			dev->SetPixelShader(shaders[shader].pixelShader);
		}
		currentShader = shader;
	}
}

void Direct3DRenderer::changeVertexFormat(const VertexFormatID vertexFormat){
	if (vertexFormat != currentVertexFormat){
		if (vertexFormat != VF_NONE){
			dev->SetVertexDeclaration(vertexFormats[vertexFormat].vertexDecl);

			if (currentVertexFormat != VF_NONE){
				for (int i = 0; i < MAX_VERTEXSTREAM; i++){
					if (vertexFormats[vertexFormat].vertexSize[i] != vertexFormats[currentVertexFormat].vertexSize[i]){
						currentVertexBuffers[i] = VB_INVALID;
					}
				}
			}
		}

		currentVertexFormat = vertexFormat;
	}
}

void Direct3DRenderer::changeVertexBuffer(const int stream, const VertexBufferID vertexBuffer, const intptr offset){
	if (vertexBuffer != currentVertexBuffers[stream] || offset != currentOffsets[stream]){
		if (vertexBuffer == VB_NONE){
			dev->SetStreamSource(stream, NULL, 0, 0);
		} else {
			dev->SetStreamSource(stream, vertexBuffers[vertexBuffer].vertexBuffer, (UINT) offset, vertexFormats[currentVertexFormat].vertexSize[stream]);
		}

		currentVertexBuffers[stream] = vertexBuffer;
		currentOffsets[stream] = offset;
	}
}

void Direct3DRenderer::changeIndexBuffer(const IndexBufferID indexBuffer){
	if (indexBuffer != currentIndexBuffer){
		if (indexBuffer == IB_NONE){
			dev->SetIndices(NULL);
		} else {
			dev->SetIndices(indexBuffers[indexBuffer].indexBuffer);
		}

		currentIndexBuffer = indexBuffer;
	}
}

void Direct3DRenderer::changeBlendState(const BlendStateID blendState, const uint sampleMask){
	if (blendState != currentBlendState){
		if (blendState == BS_NONE || !blendStates[blendState].blendEnable){
			if (currentBlendEnable){
				dev->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
				currentBlendEnable = false;
			}
		} else {
			if (blendStates[blendState].blendEnable){
				if (!currentBlendEnable){
					dev->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
					currentBlendEnable = true;
				}
				if (blendStates[blendState].srcFactorRGB != currentSrcFactorRGB){
					dev->SetRenderState(D3DRS_SRCBLEND, currentSrcFactorRGB = blendStates[blendState].srcFactorRGB);
				}
				if (blendStates[blendState].dstFactorRGB != currentDstFactorRGB){
					dev->SetRenderState(D3DRS_DESTBLEND, currentDstFactorRGB = blendStates[blendState].dstFactorRGB);
				}
				if (blendStates[blendState].blendModeRGB != currentBlendModeRGB){
					dev->SetRenderState(D3DRS_BLENDOP, currentBlendModeRGB = blendStates[blendState].blendModeRGB);
				}
				if (blendStates[blendState].srcFactorAlpha != currentSrcFactorAlpha){
					dev->SetRenderState(D3DRS_SRCBLENDALPHA, currentSrcFactorAlpha = blendStates[blendState].srcFactorRGB);
				}
				if (blendStates[blendState].dstFactorAlpha != currentDstFactorAlpha){
					dev->SetRenderState(D3DRS_DESTBLENDALPHA, currentDstFactorAlpha = blendStates[blendState].dstFactorRGB);
				}
				if (blendStates[blendState].blendModeAlpha != currentBlendModeAlpha){
					dev->SetRenderState(D3DRS_BLENDOPALPHA, currentBlendModeAlpha = blendStates[blendState].blendModeRGB);
				}
			}
		}

		int mask = ALL;
		if (blendState != BS_NONE){
			mask = blendStates[blendState].mask;
		}

		if (mask != currentMask){
			dev->SetRenderState(D3DRS_COLORWRITEENABLE, currentMask = mask);
		}

		currentBlendState = blendState;
	}
	if (sampleMask != currentSampleMask){
		dev->SetRenderState(D3DRS_MULTISAMPLEMASK, sampleMask);
		currentSampleMask = sampleMask;
	}
}

void Direct3DRenderer::changeDepthState(const DepthStateID depthState, const uint stencilRef){
	if (depthState != currentDepthState){
		if (depthState == DS_NONE){
			if (!currentDepthTestEnable){
				dev->SetRenderState(D3DRS_ZENABLE, TRUE);
				currentDepthTestEnable = true;
			}

			if (!currentDepthWriteEnable){
				dev->SetRenderState(D3DRS_ZWRITEENABLE, TRUE);
				currentDepthWriteEnable = true;
			}

			if (currentDepthFunc != LESS){
				dev->SetRenderState(D3DRS_ZFUNC, currentDepthFunc = LESS);
			}
		} else {
			if (depthStates[depthState].depthTest){
				if (!currentDepthTestEnable){
					dev->SetRenderState(D3DRS_ZENABLE, TRUE);
					currentDepthTestEnable = true;
				}
				if (depthStates[depthState].depthWrite != currentDepthWriteEnable){
					dev->SetRenderState(D3DRS_ZWRITEENABLE, (currentDepthWriteEnable = depthStates[depthState].depthWrite)? TRUE : FALSE);
				}
				if (depthStates[depthState].depthFunc != currentDepthFunc){
					dev->SetRenderState(D3DRS_ZFUNC, currentDepthFunc = depthStates[depthState].depthFunc);
				}
			} else {
				if (currentDepthTestEnable){
					dev->SetRenderState(D3DRS_ZENABLE, FALSE);
					currentDepthTestEnable = false;
				}
			}
		}

		currentDepthState = depthState;
	}

	if (stencilRef != currentStencilRef){
		dev->SetRenderState(D3DRS_STENCILREF, stencilRef);
		currentStencilRef = stencilRef;
	}
}

void Direct3DRenderer::changeRasterizerState(const RasterizerStateID rasterizerState){
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
			dev->SetRenderState(D3DRS_CULLMODE, currentCullMode = state.cullMode);
		}

		if (state.fillMode != currentFillMode){
			dev->SetRenderState(D3DRS_FILLMODE, currentFillMode = state.fillMode);
		}

		if (state.multiSample != currentMultiSampleEnable){
			dev->SetRenderState(D3DRS_MULTISAMPLEANTIALIAS, currentMultiSampleEnable = state.multiSample);
		}

		if (state.scissor != currentScissorEnable){
			dev->SetRenderState(D3DRS_SCISSORTESTENABLE, currentScissorEnable = state.scissor);
		}

		currentRasterizerState = rasterizerState;
	}
}
/*
void Direct3DRenderer::changeShaderConstant1i(const char *name, const int constant){
	ASSERT(0);
}

void Direct3DRenderer::changeShaderConstant1f(const char *name, const float constant){
	changeShaderConstant4f(name, vec4(constant, constant, constant, constant));
}

void Direct3DRenderer::changeShaderConstant2f(const char *name, const vec2 &constant){
	changeShaderConstant4f(name, vec4(constant, 0, 1));
}

void Direct3DRenderer::changeShaderConstant3f(const char *name, const vec3 &constant){
	changeShaderConstant4f(name, vec4(constant, 1));
}

void Direct3DRenderer::changeShaderConstant4f(const char *name, const vec4 &constant){
	ASSERT(currentShader != SHADER_NONE);

	D3DXHANDLE handle;
	if (shaders[currentShader].vertexShader != NULL){
		if (shaders[currentShader].vsConstants != NULL){
			if (handle = shaders[currentShader].vsConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].vsConstants->SetVector(dev, handle, (D3DXVECTOR4 *) &constant);
			}
		}
	}

	if (shaders[currentShader].pixelShader != NULL){
		if (shaders[currentShader].psConstants != NULL){
			if (handle = shaders[currentShader].psConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].psConstants->SetVector(dev, handle, (D3DXVECTOR4 *) &constant);
			}
		}
	}
}

void Direct3DRenderer::changeShaderConstant3x3f(const char *name, const mat3 &constant){
	ASSERT(currentShader != SHADER_NONE);

	D3DXHANDLE handle;
	if (shaders[currentShader].vertexShader != NULL){
		if (shaders[currentShader].vsConstants != NULL){
			if (handle = shaders[currentShader].vsConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].vsConstants->SetFloatArray(dev, handle, constant, 9);
			}
		}
	}

	if (shaders[currentShader].pixelShader != NULL){
		if (shaders[currentShader].psConstants != NULL){
			if (handle = shaders[currentShader].psConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].psConstants->SetFloatArray(dev, handle, constant, 9);
			}
		}
	}
}

void Direct3DRenderer::changeShaderConstant4x4f(const char *name, const mat4 &constant){
	ASSERT(currentShader != SHADER_NONE);

	D3DXHANDLE handle;
	if (shaders[currentShader].vertexShader != NULL){
		if (shaders[currentShader].vsConstants != NULL){
			if (handle = shaders[currentShader].vsConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].vsConstants->SetMatrix(dev, handle, (D3DXMATRIX *) &constant);
			}
		}
	}

	if (shaders[currentShader].pixelShader != NULL){
		if (shaders[currentShader].psConstants != NULL){
			if (handle = shaders[currentShader].psConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].psConstants->SetMatrix(dev, handle, (D3DXMATRIX *) &constant);
			}
		}
	}
}

void Direct3DRenderer::changeShaderConstantArray1f(const char *name, const float *constant, const uint count){
	ASSERT(currentShader != SHADER_NONE);

	D3DXHANDLE handle;
	if (shaders[currentShader].vertexShader != NULL){
		if (shaders[currentShader].vsConstants != NULL){
			if (handle = shaders[currentShader].vsConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].vsConstants->SetFloatArray(dev, handle, constant, count);
			}
		}
	}

	if (shaders[currentShader].pixelShader != NULL){
		if (shaders[currentShader].psConstants != NULL){
			if (handle = shaders[currentShader].psConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].psConstants->SetFloatArray(dev, handle, constant, count);
			}
		}
	}
}

void Direct3DRenderer::changeShaderConstantArray2f(const char *name, const vec2 *constant, const uint count){
	ASSERT(0);
}

void Direct3DRenderer::changeShaderConstantArray3f(const char *name, const vec3 *constant, const uint count){
	ASSERT(0);
}

void Direct3DRenderer::changeShaderConstantArray4f(const char *name, const vec4 *constant, const uint count){
	ASSERT(currentShader != SHADER_NONE);

	D3DXHANDLE handle;
	if (shaders[currentShader].vertexShader != NULL){
		if (shaders[currentShader].vsConstants != NULL){
			if (handle = shaders[currentShader].vsConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].vsConstants->SetVectorArray(dev, handle, (D3DXVECTOR4 *) constant, count);
			}
		}
	}

	if (shaders[currentShader].pixelShader != NULL){
		if (shaders[currentShader].psConstants != NULL){
			if (handle = shaders[currentShader].psConstants->GetConstantByName(NULL, name)){
				shaders[currentShader].psConstants->SetVectorArray(dev, handle, (D3DXVECTOR4 *) constant, count);
			}
		}
	}
}
*/

void Direct3DRenderer::clear(const bool clearColor, const bool clearDepth, const bool clearStencil, const float *color, const float depth, const uint stencil){
	DWORD clearCol = 0, flags = 0;
	if (clearColor){
		flags |= D3DCLEAR_TARGET;
		if (color) clearCol = toBGRA(*(float4 *) color);
	}
	if (clearDepth)   flags |= D3DCLEAR_ZBUFFER;
	if (clearStencil) flags |= D3DCLEAR_STENCIL;

	dev->Clear(0, NULL, flags, clearCol, depth, stencil);
}

const D3DPRIMITIVETYPE d3dPrim[] = {
	D3DPT_TRIANGLELIST,
	D3DPT_TRIANGLEFAN,
	D3DPT_TRIANGLESTRIP,
	(D3DPRIMITIVETYPE) 0, // Quads not supported
	D3DPT_LINELIST,
	D3DPT_LINESTRIP,
	(D3DPRIMITIVETYPE) 0, // Line loops not supported
	D3DPT_POINTLIST,
};

int getPrimitiveCount(const Primitives primitives, const int count){
	switch (primitives){
		case PRIM_TRIANGLES:
			return count / 3;
		case PRIM_TRIANGLE_FAN:
		case PRIM_TRIANGLE_STRIP:
			return count - 2;
		case PRIM_LINES:
			return count / 2;
		case PRIM_LINE_STRIP:
			return count - 1;
		case PRIM_POINTS:
			return count;
	};
	return 0;
}

void Direct3DRenderer::drawArrays(const Primitives primitives, const int firstVertex, const int nVertices){
	dev->DrawPrimitive(d3dPrim[primitives], firstVertex, nVertices);
}

void Direct3DRenderer::drawElements(const Primitives primitives, const int firstIndex, const int nIndices, const int firstVertex, const int nVertices){
	int indexSize = indexBuffers[currentIndexBuffer].indexSize;

	dev->DrawIndexedPrimitive(d3dPrim[primitives], 0, firstVertex, nVertices, firstIndex, getPrimitiveCount(primitives, nIndices));
}

void Direct3DRenderer::setup2DMode(const float left, const float right, const float top, const float bottom){
	scaleBias2D.x = 2.0f / (right - left);
	scaleBias2D.y = 2.0f / (top - bottom);
	scaleBias2D.z = -1.0f;
	scaleBias2D.w =  1.0f;
}

const char *plainVS =
"float4 scaleBias;"
"float4 main(float4 position: POSITION): POSITION {"
"	position.xy = position.xy * scaleBias.xy + scaleBias.zw;"
"	return position;"
"}";

const char *plainPS =
"float4 color;"
"float4 main(): COLOR {"
"	return color;"
"}";

const char *texVS =
"struct VsIn {"
"	float4 position: POSITION;"
"	float2 texCoord: TEXCOORD;"
"};"
"float4 scaleBias;"
"VsIn main(VsIn In){"
"	In.position.xy = In.position.xy * scaleBias.xy + scaleBias.zw;"
"	return In;"
"}";

const char *texPS =
"sampler2D Base: register(s0);"
"float4 color;"
"float4 main(float2 texCoord: TEXCOORD): COLOR {"
"	return tex2D(Base, texCoord) * color;"
"}";

void Direct3DRenderer::drawPlain(const Primitives primitives, vec2 *vertices, const uint nVertices, const BlendStateID blendState, const DepthStateID depthState, const vec4 *color){
	if (plainShader == SHADER_NONE){
		plainShader = addShader(plainVS, NULL, plainPS, 0, 0, 0);
		
		FormatDesc format[] = { 0, TYPE_VERTEX, FORMAT_FLOAT, 2 };
		plainVF = addVertexFormat(format, elementsOf(format), plainShader);
	}

	float4 col = float4(1, 1, 1, 1);
	if (color) col = *color;

	reset();
	setShader(plainShader);
	setShaderConstant4f("scaleBias", scaleBias2D);
	setShaderConstant4f("color", col);
	setBlendState(blendState);
	setDepthState(depthState);
	setVertexFormat(plainVF);
	apply();

	dev->DrawPrimitiveUP(d3dPrim[primitives], getPrimitiveCount(primitives, nVertices), vertices, sizeof(vec2));
}

void Direct3DRenderer::drawTextured(const Primitives primitives, TexVertex *vertices, const uint nVertices, const TextureID texture, const SamplerStateID samplerState, const BlendStateID blendState, const DepthStateID depthState, const vec4 *color){
	if (texShader == SHADER_NONE){
		texShader = addShader(texVS, NULL, texPS, 0, 0, 0);

		FormatDesc format[] = {
			0, TYPE_VERTEX,   FORMAT_FLOAT, 2,
			0, TYPE_TEXCOORD, FORMAT_FLOAT, 2,
		};
		texVF = addVertexFormat(format, elementsOf(format), texShader);
	}

	float4 col = float4(1, 1, 1, 1);
	if (color) col = *color;

	reset();
	setShader(texShader);
	setShaderConstant4f("scaleBias", scaleBias2D);
	setShaderConstant4f("color", col);
	setTexture("Base", texture, samplerState);
	setBlendState(blendState);
	setDepthState(depthState);
	setVertexFormat(texVF);
	apply();

	dev->DrawPrimitiveUP(d3dPrim[primitives], getPrimitiveCount(primitives, nVertices), vertices, sizeof(TexVertex));
}

LPDIRECT3DBASETEXTURE9 Direct3DRenderer::getD3DTexture(const TextureID texture) const {
	return textures[texture].texture;
}

void Direct3DRenderer::flush(){
	if (eventQuery == NULL){
		dev->CreateQuery(D3DQUERYTYPE_EVENT, &eventQuery);
	}

	eventQuery->Issue(D3DISSUE_END);
	eventQuery->GetData(NULL, 0, D3DGETDATA_FLUSH);
}

void Direct3DRenderer::finish(){
	if (eventQuery == NULL){
		dev->CreateQuery(D3DQUERYTYPE_EVENT, &eventQuery);
	}

	eventQuery->Issue(D3DISSUE_END);

	while (eventQuery->GetData(NULL, 0, D3DGETDATA_FLUSH) == S_FALSE){
		// Spin-wait
	}
}
