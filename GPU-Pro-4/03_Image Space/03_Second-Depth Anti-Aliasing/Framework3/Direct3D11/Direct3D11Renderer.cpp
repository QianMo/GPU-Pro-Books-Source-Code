
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

#include "Direct3D11Renderer.h"
#include "../Util/String.h"

#define INITGUID
#include <D3DCompiler.h>
#pragma comment (lib, "d3dcompiler.lib")
#pragma comment (lib, "dxguid.lib")

struct Texture
{
	ID3D11Resource *texture;
	ID3D11ShaderResourceView *srv;
	ID3D11RenderTargetView   *rtv;
	ID3D11DepthStencilView   *dsv;
	ID3D11ShaderResourceView **srvArray;
	ID3D11RenderTargetView   **rtvArray;
	ID3D11DepthStencilView   **dsvArray;
	DXGI_FORMAT texFormat;
	DXGI_FORMAT srvFormat;
	DXGI_FORMAT rtvFormat;
	DXGI_FORMAT dsvFormat;
	int width, height, depth;
	int arraySize;
	uint flags;
};

struct Constant
{
	char *name;
	ubyte *vsData;
	ubyte *gsData;
	ubyte *psData;
	int vsBuffer;
	int gsBuffer;
	int psBuffer;
};

int constantComp(const void *s0, const void *s1)
{
	return strcmp(((Constant *) s0)->name, ((Constant *) s1)->name);
}

struct Sampler
{
	char *name;
	int vsIndex;
	int gsIndex;
	int psIndex;
};

int samplerComp(const void *s0, const void *s1)
{
	return strcmp(((Sampler *) s0)->name, ((Sampler *) s1)->name);
}

struct Shader
{
	ID3D11VertexShader *vertexShader;
	ID3D11PixelShader *pixelShader;
	ID3D11GeometryShader *geometryShader;
	ID3DBlob *inputSignature;

	ID3D11Buffer **vsConstants;
	ID3D11Buffer **gsConstants;
	ID3D11Buffer **psConstants;
	uint nVSCBuffers;
	uint nGSCBuffers;
	uint nPSCBuffers;

	Constant *constants;
	Sampler *textures;
	Sampler *samplers;

	uint nConstants;
	uint nTextures;
	uint nSamplers;

	ubyte **vsConstMem;
	ubyte **gsConstMem;
	ubyte **psConstMem;

	bool *vsDirty;
	bool *gsDirty;
	bool *psDirty;
};

struct VertexFormat
{
	ID3D11InputLayout *inputLayout;
	uint vertexSize[MAX_VERTEXSTREAM];
};

struct VertexBuffer
{
	ID3D11Buffer *vertexBuffer;
	long size;
};

struct IndexBuffer
{
	ID3D11Buffer *indexBuffer;
	uint nIndices;
	uint indexSize;
};

struct SamplerState
{
	ID3D11SamplerState *samplerState;
};

struct BlendState
{
	ID3D11BlendState *blendState;
};

struct DepthState
{
	ID3D11DepthStencilState *dsState;
};

struct RasterizerState
{
	ID3D11RasterizerState *rsState;
};

// Blending constants
const int ZERO                = D3D11_BLEND_ZERO;
const int ONE                 = D3D11_BLEND_ONE;
const int SRC_COLOR           = D3D11_BLEND_SRC_COLOR;
const int ONE_MINUS_SRC_COLOR = D3D11_BLEND_INV_SRC_COLOR;
const int DST_COLOR           = D3D11_BLEND_DEST_COLOR;
const int ONE_MINUS_DST_COLOR = D3D11_BLEND_INV_DEST_COLOR;
const int SRC_ALPHA           = D3D11_BLEND_SRC_ALPHA;
const int ONE_MINUS_SRC_ALPHA = D3D11_BLEND_INV_SRC_ALPHA;
const int DST_ALPHA           = D3D11_BLEND_DEST_ALPHA;
const int ONE_MINUS_DST_ALPHA = D3D11_BLEND_INV_DEST_ALPHA;
const int SRC_ALPHA_SATURATE  = D3D11_BLEND_SRC_ALPHA_SAT;

const int BM_ADD              = D3D11_BLEND_OP_ADD;
const int BM_SUBTRACT         = D3D11_BLEND_OP_SUBTRACT;
const int BM_REVERSE_SUBTRACT = D3D11_BLEND_OP_REV_SUBTRACT;
const int BM_MIN              = D3D11_BLEND_OP_MIN;
const int BM_MAX              = D3D11_BLEND_OP_MAX;

// Depth-test constants
const int NEVER    = D3D11_COMPARISON_NEVER;
const int LESS     = D3D11_COMPARISON_LESS;
const int EQUAL    = D3D11_COMPARISON_EQUAL;
const int LEQUAL   = D3D11_COMPARISON_LESS_EQUAL;
const int GREATER  = D3D11_COMPARISON_GREATER;
const int NOTEQUAL = D3D11_COMPARISON_NOT_EQUAL;
const int GEQUAL   = D3D11_COMPARISON_GREATER_EQUAL;
const int ALWAYS   = D3D11_COMPARISON_ALWAYS;

// Stencil-test constants
const int KEEP     = D3D11_STENCIL_OP_KEEP;
const int SET_ZERO = D3D11_STENCIL_OP_ZERO;
const int REPLACE  = D3D11_STENCIL_OP_REPLACE;
const int INVERT   = D3D11_STENCIL_OP_INVERT;
const int INCR     = D3D11_STENCIL_OP_INCR;
const int DECR     = D3D11_STENCIL_OP_DECR;
const int INCR_SAT = D3D11_STENCIL_OP_INCR_SAT;
const int DECR_SAT = D3D11_STENCIL_OP_DECR_SAT;

// Culling constants
const int CULL_NONE  = D3D11_CULL_NONE;
const int CULL_BACK  = D3D11_CULL_BACK;
const int CULL_FRONT = D3D11_CULL_FRONT;

// Fillmode constants
const int SOLID = D3D11_FILL_SOLID;
const int WIREFRAME = D3D11_FILL_WIREFRAME;


static DXGI_FORMAT formats[] =
{
	DXGI_FORMAT_UNKNOWN,

	DXGI_FORMAT_R8_UNORM,
	DXGI_FORMAT_R8G8_UNORM,
	DXGI_FORMAT_UNKNOWN, // RGB8 not directly supported
	DXGI_FORMAT_R8G8B8A8_UNORM,

	DXGI_FORMAT_R16_UNORM,
	DXGI_FORMAT_R16G16_UNORM,
	DXGI_FORMAT_UNKNOWN, // RGB16 not directly supported
	DXGI_FORMAT_R16G16B16A16_UNORM,

	DXGI_FORMAT_R8_SNORM,
	DXGI_FORMAT_R8G8_SNORM,
	DXGI_FORMAT_UNKNOWN, // RGB8S not directly supported
	DXGI_FORMAT_R8G8B8A8_SNORM,

	DXGI_FORMAT_R16_SNORM,
	DXGI_FORMAT_R16G16_SNORM,
	DXGI_FORMAT_UNKNOWN, // RGB16S not directly supported
	DXGI_FORMAT_R16G16B16A16_SNORM,

	DXGI_FORMAT_R16_FLOAT,
	DXGI_FORMAT_R16G16_FLOAT,
	DXGI_FORMAT_UNKNOWN, // RGB16F not directly supported
	DXGI_FORMAT_R16G16B16A16_FLOAT,

	DXGI_FORMAT_R32_FLOAT,
	DXGI_FORMAT_R32G32_FLOAT,
	DXGI_FORMAT_R32G32B32_FLOAT,
	DXGI_FORMAT_R32G32B32A32_FLOAT,

	DXGI_FORMAT_R16_SINT,
	DXGI_FORMAT_R16G16_SINT,
	DXGI_FORMAT_UNKNOWN, // RGB16I not directly supported
	DXGI_FORMAT_R16G16B16A16_SINT,

	DXGI_FORMAT_R32_SINT,
	DXGI_FORMAT_R32G32_SINT,
	DXGI_FORMAT_R32G32B32_SINT,
	DXGI_FORMAT_R32G32B32A32_SINT,

	DXGI_FORMAT_R16_UINT,
	DXGI_FORMAT_R16G16_UINT,
	DXGI_FORMAT_UNKNOWN, // RGB16UI not directly supported
	DXGI_FORMAT_R16G16B16A16_UINT,

	DXGI_FORMAT_R32_UINT,
	DXGI_FORMAT_R32G32_UINT,
	DXGI_FORMAT_R32G32B32_UINT,
	DXGI_FORMAT_R32G32B32A32_UINT,

	DXGI_FORMAT_UNKNOWN, // RGBE8 not directly supported
	DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
	DXGI_FORMAT_R11G11B10_FLOAT,
	DXGI_FORMAT_B5G6R5_UNORM,
	DXGI_FORMAT_UNKNOWN, // RGBA4 not directly supported
	DXGI_FORMAT_R10G10B10A2_UNORM,

	DXGI_FORMAT_D16_UNORM,
	DXGI_FORMAT_D24_UNORM_S8_UINT,
	DXGI_FORMAT_D24_UNORM_S8_UINT,
	DXGI_FORMAT_D32_FLOAT,

	DXGI_FORMAT_BC1_UNORM,
	DXGI_FORMAT_BC2_UNORM,
	DXGI_FORMAT_BC3_UNORM,
	DXGI_FORMAT_BC4_UNORM,
	DXGI_FORMAT_BC5_UNORM,
};

Direct3D11Renderer::Direct3D11Renderer(ID3D11Device *d3ddev, ID3D11DeviceContext *ctx) : Renderer()
{
	device = d3ddev;
	context = ctx;

	eventQuery = NULL;

	nImageUnits = 16;
	maxAnisotropic = 16;


//	textureLod = new float[nImageUnits];

	nMRTs = 8;
	if (nMRTs > MAX_MRTS)
		nMRTs = MAX_MRTS;

	plainShader = SHADER_NONE;
	plainVF = VF_NONE;
	texShader = SHADER_NONE;
	texVF = VF_NONE;
//	rollingVB = NULL;
	rollingVB = VB_NONE;
	rollingVBOffset = 0;

	backBufferRTV = NULL;
	depthBufferDSV = NULL;

	resetToDefaults();
}

Direct3D11Renderer::~Direct3D11Renderer()
{
	context->ClearState();

	if (eventQuery)
		eventQuery->Release();


/*
	releaseFrameBufferSurfaces();
*/

	// Delete shaders
	for (uint i = 0; i < shaders.getCount(); i++)
	{
		if (shaders[i].vertexShader  ) shaders[i].vertexShader->Release();
		if (shaders[i].geometryShader) shaders[i].geometryShader->Release();
		if (shaders[i].pixelShader   ) shaders[i].pixelShader->Release();
		if (shaders[i].inputSignature) shaders[i].inputSignature->Release();

		for (uint k = 0; k < shaders[i].nVSCBuffers; k++)
		{
			shaders[i].vsConstants[k]->Release();
			delete [] shaders[i].vsConstMem[k];
		}
		for (uint k = 0; k < shaders[i].nGSCBuffers; k++)
		{
			shaders[i].gsConstants[k]->Release();
			delete [] shaders[i].gsConstMem[k];
		}
		for (uint k = 0; k < shaders[i].nPSCBuffers; k++)
		{
			shaders[i].psConstants[k]->Release();
			delete [] shaders[i].psConstMem[k];
		}
		delete [] shaders[i].vsConstants;
		delete [] shaders[i].gsConstants;
		delete [] shaders[i].psConstants;
		delete [] shaders[i].vsConstMem;
		delete [] shaders[i].gsConstMem;
		delete [] shaders[i].psConstMem;

		for (uint k = 0; k < shaders[i].nConstants; k++)
		{
			delete [] shaders[i].constants[k].name;
		}
		delete shaders[i].constants;

		for (uint k = 0; k < shaders[i].nTextures; k++)
		{
			delete [] shaders[i].textures[k].name;
		}
		free(shaders[i].textures);

		for (uint k = 0; k < shaders[i].nSamplers; k++)
		{
			delete [] shaders[i].samplers[k].name;
		}
		free(shaders[i].samplers);

		delete [] shaders[i].vsDirty;
		delete [] shaders[i].gsDirty;
		delete [] shaders[i].psDirty;
	}

    // Delete vertex formats
	for (uint i = 0; i < vertexFormats.getCount(); i++)
	{
		if (vertexFormats[i].inputLayout)
			vertexFormats[i].inputLayout->Release();
	}

    // Delete vertex buffers
	for (uint i = 0; i < vertexBuffers.getCount(); i++)
	{
		if (vertexBuffers[i].vertexBuffer)
			vertexBuffers[i].vertexBuffer->Release();
	}

	// Delete index buffers
	for (uint i = 0; i < indexBuffers.getCount(); i++)
	{
		if (indexBuffers[i].indexBuffer)
			indexBuffers[i].indexBuffer->Release();
	}

	// Delete samplerstates
	for (uint i = 0; i < samplerStates.getCount(); i++)
	{
		if (samplerStates[i].samplerState)
			samplerStates[i].samplerState->Release();
	}

	// Delete blendstates
	for (uint i = 0; i < blendStates.getCount(); i++)
	{
		if (blendStates[i].blendState)
			blendStates[i].blendState->Release();
	}

	// Delete depthstates
	for (uint i = 0; i < depthStates.getCount(); i++)
	{
		if (depthStates[i].dsState)
			depthStates[i].dsState->Release();
	}

	// Delete rasterizerstates
	for (uint i = 0; i < rasterizerStates.getCount(); i++)
	{
		if (rasterizerStates[i].rsState)
			rasterizerStates[i].rsState->Release();
	}

	// Delete textures
	for (uint i = 0; i < textures.getCount(); i++)
	{
		removeTexture(i);
	}

//	if (rollingVB) rollingVB->Release();
}

void Direct3D11Renderer::reset(const uint flags)
{
	Renderer::reset(flags);

	if (flags & RESET_TEX)
	{
		for (uint i = 0; i < MAX_TEXTUREUNIT; i++)
		{
			selectedTexturesVS[i] = TEXTURE_NONE;
			selectedTexturesGS[i] = TEXTURE_NONE;
			selectedTexturesPS[i] = TEXTURE_NONE;
			selectedTextureSlicesVS[i] = NO_SLICE;
			selectedTextureSlicesGS[i] = NO_SLICE;
			selectedTextureSlicesPS[i] = NO_SLICE;
		}
	}

	if (flags & RESET_SS)
	{
		for (uint i = 0; i < MAX_SAMPLERSTATE; i++)
		{
			selectedSamplerStatesVS[i] = SS_NONE;
			selectedSamplerStatesGS[i] = SS_NONE;
			selectedSamplerStatesPS[i] = SS_NONE;
		}
	}
}

void Direct3D11Renderer::resetToDefaults()
{
	Renderer::resetToDefaults();

	for (uint i = 0; i < MAX_TEXTUREUNIT; i++)
	{
		currentTexturesVS[i] = TEXTURE_NONE;
		currentTexturesGS[i] = TEXTURE_NONE;
		currentTexturesPS[i] = TEXTURE_NONE;
		currentTextureSlicesVS[i] = NO_SLICE;
		currentTextureSlicesGS[i] = NO_SLICE;
		currentTextureSlicesPS[i] = NO_SLICE;
	}

	for (uint i = 0; i < MAX_SAMPLERSTATE; i++)
	{
		currentSamplerStatesVS[i] = SS_NONE;
		currentSamplerStatesGS[i] = SS_NONE;
		currentSamplerStatesPS[i] = SS_NONE;
	}

	// TODO: Fix ...
	currentRasterizerState = -2;


/*
	currentDepthRT = FB_DEPTH;
*/
}

TextureID Direct3D11Renderer::addTexture(ID3D11Resource *resource, uint flags)
{
	Texture tex;
	memset(&tex, 0, sizeof(tex));

	tex.texture = resource;
	tex.srv = createSRV(resource);
	tex.flags = flags;

	D3D11_RESOURCE_DIMENSION type;
	resource->GetType(&type);

	switch (type)
	{
		case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
			D3D11_TEXTURE1D_DESC desc1d;
			((ID3D11Texture1D *) resource)->GetDesc(&desc1d);

			tex.width  = desc1d.Width;
			tex.height = 1;
			tex.depth  = 1;
			break;
		case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
			D3D11_TEXTURE2D_DESC desc2d;
			((ID3D11Texture2D *) resource)->GetDesc(&desc2d);

			tex.width  = desc2d.Width;
			tex.height = desc2d.Height;
			tex.depth  = 1;
			break;
		case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
			D3D11_TEXTURE3D_DESC desc3d;
			((ID3D11Texture3D *) resource)->GetDesc(&desc3d);

			tex.width  = desc3d.Width;
			tex.height = desc3d.Height;
			tex.depth  = desc3d.Depth;
			break;
	}

	return textures.add(tex);
}


TextureID Direct3D11Renderer::addTexture(Image &img, const SamplerStateID samplerState, uint flags)
{
	Texture tex;
	memset(&tex, 0, sizeof(tex));

	switch (img.getFormat())
	{
		case FORMAT_RGB8:
			img.convert(FORMAT_RGBA8);
//			img.convert(FORMAT_RGB10A2);
			break;
		case FORMAT_RGB16:
			img.convert(FORMAT_RGBA16);
			break;
		case FORMAT_RGB16F:
			img.convert(FORMAT_RGBA16F);
			break;
		case FORMAT_RGB32F:
			img.convert(FORMAT_RGBA32F);
			break;
	}

	FORMAT format = img.getFormat();
	uint nMipMaps = img.getMipMapCount();
	uint nSlices = img.isCube()? 6 : 1;
	uint arraySize = img.getArraySize();

	static D3D11_SUBRESOURCE_DATA texData[1024];
	D3D11_SUBRESOURCE_DATA *dest = texData;
	for (uint n = 0; n < arraySize; n++)
	{
		for (uint k = 0; k < nSlices; k++)
		{
			for (uint i = 0; i < nMipMaps; i++)
			{
				uint pitch, slicePitch;
				if (isCompressedFormat(format))
				{
					pitch = ((img.getWidth(i) + 3) >> 2) * getBytesPerBlock(format);
					slicePitch = pitch * ((img.getHeight(i) + 3) >> 2);
				}
				else
				{
					pitch = img.getWidth(i) * getBytesPerPixel(format);
					slicePitch = pitch * img.getHeight(i);
				}

				dest->pSysMem = img.getPixels(i, n) + k * slicePitch;
				dest->SysMemPitch = pitch;
				dest->SysMemSlicePitch = slicePitch;
				dest++;
			}
		}
	}

	tex.texFormat = formats[format];
	if (flags & SRGB)
	{
		// Change to the matching sRGB format
		switch (tex.texFormat)
		{
			case DXGI_FORMAT_R8G8B8A8_UNORM: tex.texFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; break;
			case DXGI_FORMAT_BC1_UNORM: tex.texFormat = DXGI_FORMAT_BC1_UNORM_SRGB; break;
			case DXGI_FORMAT_BC2_UNORM: tex.texFormat = DXGI_FORMAT_BC2_UNORM_SRGB; break;
			case DXGI_FORMAT_BC3_UNORM: tex.texFormat = DXGI_FORMAT_BC3_UNORM_SRGB; break;
		}
	}

	HRESULT hr;
	if (img.is1D())
	{
		D3D11_TEXTURE1D_DESC desc;
		desc.Width  = img.getWidth();
		desc.Format = tex.texFormat;
		desc.MipLevels = nMipMaps;
		desc.Usage = D3D11_USAGE_IMMUTABLE;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		desc.CPUAccessFlags = 0;
		desc.ArraySize = 1;
		desc.MiscFlags = 0;

		hr = device->CreateTexture1D(&desc, texData, (ID3D11Texture1D **) &tex.texture);
	}
	else if (img.is2D() || img.isCube())
	{
		D3D11_TEXTURE2D_DESC desc;
		desc.Width  = img.getWidth();
		desc.Height = img.getHeight();
		desc.Format = tex.texFormat;
		desc.MipLevels = nMipMaps;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D11_USAGE_IMMUTABLE;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		desc.CPUAccessFlags = 0;
		if (img.isCube())
		{
			desc.ArraySize = 6 * arraySize;
			desc.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;
		}
		else
		{
			desc.ArraySize = 1;
			desc.MiscFlags = 0;
		}

		hr = device->CreateTexture2D(&desc, texData, (ID3D11Texture2D **) &tex.texture);
	}
	else if (img.is3D())
	{
		D3D11_TEXTURE3D_DESC desc;
		desc.Width  = img.getWidth();
		desc.Height = img.getHeight();
		desc.Depth  = img.getDepth();
		desc.Format = tex.texFormat;
		desc.MipLevels = nMipMaps;
		desc.Usage = D3D11_USAGE_IMMUTABLE;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = 0;

		hr = device->CreateTexture3D(&desc, texData, (ID3D11Texture3D **) &tex.texture);
	}

	if (FAILED(hr))
	{
		ErrorMsg("Couldn't create texture");
		return TEXTURE_NONE;
	}

	tex.srvFormat = tex.texFormat;
	tex.srv = createSRV(tex.texture, tex.srvFormat);

	return textures.add(tex);
}

TextureID Direct3D11Renderer::addRenderTarget(const int width, const int height, const int depth, const int mipMapCount, const int arraySize, const FORMAT format, const int msaaSamples, const SamplerStateID samplerState, uint flags)
{
	Texture tex;
	memset(&tex, 0, sizeof(tex));

	tex.width  = width;
	tex.height = height;
	tex.depth  = depth;
	tex.arraySize = arraySize;
	tex.flags  = flags;
	tex.texFormat = formats[format];
	if (flags & SRGB)
	{
		// Change to the matching sRGB format
		switch (tex.texFormat)
		{
			case DXGI_FORMAT_R8G8B8A8_UNORM: tex.texFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; break;
			case DXGI_FORMAT_BC1_UNORM: tex.texFormat = DXGI_FORMAT_BC1_UNORM_SRGB; break;
			case DXGI_FORMAT_BC2_UNORM: tex.texFormat = DXGI_FORMAT_BC2_UNORM_SRGB; break;
			case DXGI_FORMAT_BC3_UNORM: tex.texFormat = DXGI_FORMAT_BC3_UNORM_SRGB; break;
		}
	}


	if (depth == 1)
	{
		D3D11_TEXTURE2D_DESC desc;
		desc.Width  = width;
		desc.Height = height;
		desc.Format = tex.texFormat;
		desc.MipLevels = mipMapCount;
		desc.SampleDesc.Count = msaaSamples;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		desc.CPUAccessFlags = 0;
		if (flags & CUBEMAP)
		{
			desc.ArraySize = 6;
			desc.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;
		}
		else
		{
			desc.ArraySize = arraySize;
			desc.MiscFlags = 0;
		}
		if (flags & USE_MIPGEN)
		{
			desc.MiscFlags |= D3D11_RESOURCE_MISC_GENERATE_MIPS;
		}

		if (FAILED(device->CreateTexture2D(&desc, NULL, (ID3D11Texture2D **) &tex.texture)))
		{
			ErrorMsg("Couldn't create render target");
			return TEXTURE_NONE;
		}
	}
	else
	{
		D3D11_TEXTURE3D_DESC desc;
		desc.Width  = width;
		desc.Height = height;
		desc.Depth  = depth;
		desc.Format = tex.texFormat;
		desc.MipLevels = mipMapCount;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = 0;

		if (flags & USE_MIPGEN)
		{
			desc.MiscFlags |= D3D11_RESOURCE_MISC_GENERATE_MIPS;
		}

		if (FAILED(device->CreateTexture3D(&desc, NULL, (ID3D11Texture3D **) &tex.texture)))
		{
			ErrorMsg("Couldn't create render target");
			return TEXTURE_NONE;
		}
	}


	tex.srvFormat = tex.texFormat;
	tex.rtvFormat = tex.texFormat;
	tex.srv = createSRV(tex.texture, tex.srvFormat);
	tex.rtv = createRTV(tex.texture, tex.rtvFormat);

	int sliceCount = (depth == 1)? arraySize : depth;

	if (flags & SAMPLE_SLICES)
	{
		tex.srvArray = new ID3D11ShaderResourceView *[sliceCount];
		for (int i = 0; i < sliceCount; i++)
		{
			tex.srvArray[i] = createSRV(tex.texture, tex.srvFormat, i);
		}
	}

	if (flags & RENDER_SLICES)
	{
		tex.rtvArray = new ID3D11RenderTargetView *[sliceCount];

		for (int i = 0; i < sliceCount; i++)
		{
			tex.rtvArray[i] = createRTV(tex.texture, tex.rtvFormat, i);
		}
	}

	return textures.add(tex);
}

TextureID Direct3D11Renderer::addRenderDepth(const int width, const int height, const int arraySize, const FORMAT format, const int msaaSamples, const SamplerStateID samplerState, uint flags)
{
	Texture tex;
	memset(&tex, 0, sizeof(tex));

	tex.width  = width;
	tex.height = height;
	tex.depth  = 1;
	tex.arraySize = arraySize;
	tex.flags  = flags;
	tex.texFormat = tex.dsvFormat = formats[format];

	D3D11_TEXTURE2D_DESC desc;
	desc.Width  = width;
	desc.Height = height;
	desc.MipLevels = 1;
	desc.SampleDesc.Count = msaaSamples;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	desc.CPUAccessFlags = 0;
	if (flags & CUBEMAP)
	{
		desc.ArraySize = 6;
		desc.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;
	}
	else
	{
		desc.ArraySize = arraySize;
		desc.MiscFlags = 0;
	}

	if (flags & SAMPLE_DEPTH)
	{
		switch (tex.dsvFormat)
		{
			case DXGI_FORMAT_D16_UNORM:
				tex.texFormat = DXGI_FORMAT_R16_TYPELESS;
				tex.srvFormat = DXGI_FORMAT_R16_UNORM;
				break;
			case DXGI_FORMAT_D24_UNORM_S8_UINT:
				tex.texFormat = DXGI_FORMAT_R24G8_TYPELESS;
				tex.srvFormat = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
				break;
			case DXGI_FORMAT_D32_FLOAT:
				tex.texFormat = DXGI_FORMAT_R32_TYPELESS;
				tex.srvFormat = DXGI_FORMAT_R32_FLOAT;
				break;
		}
		desc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
	}
	desc.Format = tex.texFormat;

	if (FAILED(device->CreateTexture2D(&desc, NULL, (ID3D11Texture2D **) &tex.texture)))
	{
		ErrorMsg("Couldn't create depth target");
		return TEXTURE_NONE;
	}

	tex.dsv = createDSV(tex.texture, tex.dsvFormat);
	if (flags & RENDER_SLICES)
	{
		tex.dsvArray = new ID3D11DepthStencilView *[arraySize];
		for (int i = 0; i < arraySize; i++)
		{
			tex.dsvArray[i] = createDSV(tex.texture, tex.dsvFormat, i);
		}
	}

	if (flags & SAMPLE_DEPTH)
	{
		tex.srv = createSRV(tex.texture, tex.srvFormat);

		if (flags & SAMPLE_SLICES)
		{
			tex.srvArray = new ID3D11ShaderResourceView *[arraySize];
			for (int i = 0; i < arraySize; i++)
			{
				tex.srvArray[i] = createSRV(tex.texture, tex.srvFormat, i);
			}
		}
	}

	return textures.add(tex);
}

bool Direct3D11Renderer::resizeRenderTarget(const TextureID renderTarget, const int width, const int height, const int depth, const int mipMapCount, const int arraySize)
{
	D3D11_RESOURCE_DIMENSION type;
	textures[renderTarget].texture->GetType(&type);

	switch (type)
	{
/*		case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
			D3D11_TEXTURE1D_DESC desc1d;
			((ID3D11Texture1D *) textures[renderTarget].texture)->GetDesc(&desc1d);

			desc1d.Width     = width;
			desc1d.ArraySize = arraySize;
			break;*/
		case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
			D3D11_TEXTURE2D_DESC desc2d;
			((ID3D11Texture2D *) textures[renderTarget].texture)->GetDesc(&desc2d);

			desc2d.Width     = width;
			desc2d.Height    = height;
			desc2d.ArraySize = arraySize;
			desc2d.MipLevels = mipMapCount;

			textures[renderTarget].texture->Release();
			if (FAILED(device->CreateTexture2D(&desc2d, NULL, (ID3D11Texture2D **) &textures[renderTarget].texture)))
			{
				ErrorMsg("Couldn't create render target");
				return false;
			}
			break;
/*		case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
			D3D11_TEXTURE3D_DESC desc3d;
			((ID3D11Texture3D *) textures[renderTarget].texture)->GetDesc(&desc3d);

			desc3d.Width  = width;
			desc3d.Height = height;
			desc3d.Depth  = depth;
			break;*/
		default:
			return false;
	}

	if (textures[renderTarget].rtv)
	{
		textures[renderTarget].rtv->Release();
		textures[renderTarget].rtv = createRTV(textures[renderTarget].texture, textures[renderTarget].rtvFormat);
	}
	if (textures[renderTarget].dsv)
	{
		textures[renderTarget].dsv->Release();
		textures[renderTarget].dsv = createDSV(textures[renderTarget].texture, textures[renderTarget].dsvFormat);
	}
	if (textures[renderTarget].srv)
	{
		textures[renderTarget].srv->Release();
		textures[renderTarget].srv = createSRV(textures[renderTarget].texture, textures[renderTarget].srvFormat);
	}
	if (textures[renderTarget].rtvArray)
	{
		for (int i = 0; i < textures[renderTarget].arraySize; i++)
		{
			textures[renderTarget].rtvArray[i]->Release();
		}
		if (arraySize != textures[renderTarget].arraySize)
		{
			delete [] textures[renderTarget].rtvArray;
			textures[renderTarget].rtvArray = new ID3D11RenderTargetView *[arraySize];
		}
		for (int i = 0; i < arraySize; i++)
		{
			textures[renderTarget].rtvArray[i] = createRTV(textures[renderTarget].texture, textures[renderTarget].rtvFormat, i);
		}
	}
	if (textures[renderTarget].dsvArray)
	{
		for (int i = 0; i < textures[renderTarget].arraySize; i++)
		{
			textures[renderTarget].dsvArray[i]->Release();
		}
		if (arraySize != textures[renderTarget].arraySize)
		{
			delete [] textures[renderTarget].dsvArray;
			textures[renderTarget].dsvArray = new ID3D11DepthStencilView *[arraySize];
		}
		for (int i = 0; i < arraySize; i++)
		{
			textures[renderTarget].dsvArray[i] = createDSV(textures[renderTarget].texture, textures[renderTarget].dsvFormat, i);
		}
	}
	if (textures[renderTarget].srvArray)
	{
		for (int i = 0; i < textures[renderTarget].arraySize; i++)
		{
			textures[renderTarget].srvArray[i]->Release();
		}
		if (arraySize != textures[renderTarget].arraySize)
		{
			delete [] textures[renderTarget].srvArray;
			textures[renderTarget].srvArray = new ID3D11ShaderResourceView *[arraySize];
		}
		for (int i = 0; i < arraySize; i++)
		{
			textures[renderTarget].srvArray[i] = createSRV(textures[renderTarget].texture, textures[renderTarget].srvFormat, i);
		}
	}

	textures[renderTarget].width  = width;
	textures[renderTarget].height = height;
	textures[renderTarget].depth  = depth;
	textures[renderTarget].arraySize = arraySize;

	return true;
}

bool Direct3D11Renderer::generateMipMaps(const TextureID renderTarget)
{
	context->GenerateMips(textures[renderTarget].srv);

	return true;
}

void Direct3D11Renderer::removeTexture(const TextureID texture)
{
	SAFE_RELEASE(textures[texture].texture);
	SAFE_RELEASE(textures[texture].srv);
	SAFE_RELEASE(textures[texture].rtv);
	SAFE_RELEASE(textures[texture].dsv);

	int sliceCount = (textures[texture].depth == 1)? textures[texture].arraySize : textures[texture].depth;

	if (textures[texture].srvArray)
	{
		for (int i = 0; i < sliceCount; i++)
		{
			textures[texture].srvArray[i]->Release();
		}
		delete [] textures[texture].srvArray;
		textures[texture].srvArray = NULL;
	}
	if (textures[texture].rtvArray)
	{
		for (int i = 0; i < sliceCount; i++)
		{
			textures[texture].rtvArray[i]->Release();
		}
		delete [] textures[texture].rtvArray;
		textures[texture].rtvArray = NULL;
	}
	if (textures[texture].dsvArray)
	{
		for (int i = 0; i < sliceCount; i++)
		{
			textures[texture].dsvArray[i]->Release();
		}
		delete [] textures[texture].dsvArray;
		textures[texture].dsvArray = NULL;
	}
}

ShaderID Direct3D11Renderer::addShader(const char *vsText, const char *gsText, const char *fsText, const int vsLine, const int gsLine, const int fsLine,
									   const char *header, const char *extra, const char *fileName, const char **attributeNames, const int nAttributes, const uint flags)
{
	if (vsText == NULL && gsText == NULL && fsText == NULL)
		return SHADER_NONE;

	D3D_FEATURE_LEVEL feature_level = device->GetFeatureLevel();

	Shader shader;
	memset(&shader, 0, sizeof(shader));

	ID3DBlob *shaderBuf = NULL;
	ID3DBlob *errorsBuf = NULL;

	ID3D11ShaderReflection *vsRefl = NULL;
	ID3D11ShaderReflection *gsRefl = NULL;
	ID3D11ShaderReflection *psRefl = NULL;

	UINT compileFlags = D3DCOMPILE_PACK_MATRIX_ROW_MAJOR | D3DCOMPILE_ENABLE_STRICTNESS;// | D3D11_SHADER_DEBUG | D3D11_SHADER_SKIP_OPTIMIZATION;

	if (vsText != NULL)
	{
		String shaderString;
		if (extra != NULL) shaderString += extra;
		if (header != NULL) shaderString += header;
		shaderString.sprintf("#line %d\n", vsLine + 1);
		shaderString += vsText;

		const char *target = (feature_level == D3D_FEATURE_LEVEL_11_0)? "vs_5_0" : (feature_level == D3D_FEATURE_LEVEL_10_1)? "vs_4_1" : "vs_4_0";

		if (SUCCEEDED(D3DCompile(shaderString, shaderString.getLength(), fileName, NULL, NULL, "main", target, compileFlags, 0, &shaderBuf, &errorsBuf)))
		{
			if (SUCCEEDED(device->CreateVertexShader(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), NULL, &shader.vertexShader)))
			{
				D3DGetInputSignatureBlob(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), &shader.inputSignature);
				D3DReflect(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), IID_ID3D11ShaderReflection, (void **) &vsRefl);

#ifdef DEBUG
				if (flags & ASSEMBLY)
				{
					ID3DBlob *disasm = NULL;
					if (SUCCEEDED(D3DDisassemble(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), 0, NULL, &disasm)))
					{
						outputDebugString((const char *) disasm->GetBufferPointer());
					}
					SAFE_RELEASE(disasm);
				}
#endif
			}
		}
		else
		{
			ErrorMsg((const char *) errorsBuf->GetBufferPointer());
		}
		SAFE_RELEASE(shaderBuf);
		SAFE_RELEASE(errorsBuf);

		if (shader.vertexShader == NULL)
			return SHADER_NONE;
	}

	if (gsText != NULL)
	{
		String shaderString;
		if (extra != NULL) shaderString += extra;
		if (header != NULL) shaderString += header;
		shaderString.sprintf("#line %d\n", gsLine + 1);
		shaderString += gsText;

		const char *target = (feature_level == D3D_FEATURE_LEVEL_11_0)? "gs_5_0" : (feature_level == D3D_FEATURE_LEVEL_10_1)? "gs_4_1" : "gs_4_0";

		if (SUCCEEDED(D3DCompile(shaderString, shaderString.getLength(), fileName, NULL, NULL, "main", target, compileFlags, 0, &shaderBuf, &errorsBuf)))
		{
			if (SUCCEEDED(device->CreateGeometryShader(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), NULL, &shader.geometryShader)))
			{
				D3DReflect(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), IID_ID3D11ShaderReflection, (void **) &gsRefl);
#ifdef DEBUG
				if (flags & ASSEMBLY)
				{
					ID3DBlob *disasm = NULL;
					if (SUCCEEDED(D3DDisassemble(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), 0, NULL, &disasm)))
					{
						outputDebugString((const char *) disasm->GetBufferPointer());
					}
					SAFE_RELEASE(disasm);
				}
#endif
			}
		}
		else
		{
			ErrorMsg((const char *) errorsBuf->GetBufferPointer());
		}
		SAFE_RELEASE(shaderBuf);
		SAFE_RELEASE(errorsBuf);

		if (shader.geometryShader == NULL)
			return SHADER_NONE;
	}

	if (fsText != NULL)
	{
		String shaderString;
		if (extra != NULL) shaderString += extra;
		if (header != NULL) shaderString += header;
		shaderString.sprintf("#line %d\n", fsLine + 1);
		shaderString += fsText;

		const char *target = (feature_level == D3D_FEATURE_LEVEL_11_0)? "ps_5_0" : (feature_level == D3D_FEATURE_LEVEL_10_1)? "ps_4_1" : "ps_4_0";

		if (SUCCEEDED(D3DCompile(shaderString, shaderString.getLength(), fileName, NULL, NULL, "main", target, compileFlags, 0, &shaderBuf, &errorsBuf)))
		{
			if (SUCCEEDED(device->CreatePixelShader(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), NULL, &shader.pixelShader)))
			{
				D3DReflect(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), IID_ID3D11ShaderReflection, (void **) &psRefl);
#ifdef DEBUG
				if (flags & ASSEMBLY)
				{
					ID3DBlob *disasm = NULL;
					if (SUCCEEDED(D3DDisassemble(shaderBuf->GetBufferPointer(), shaderBuf->GetBufferSize(), 0, NULL, &disasm)))
					{
						outputDebugString((const char *) disasm->GetBufferPointer());
					}
					SAFE_RELEASE(disasm);
				}
#endif
			}
		}
		else
		{
			ErrorMsg((const char *) errorsBuf->GetBufferPointer());
		}
		SAFE_RELEASE(shaderBuf);
		SAFE_RELEASE(errorsBuf);

		if (shader.pixelShader == NULL)
			return SHADER_NONE;
	}

	D3D11_SHADER_DESC vsDesc, gsDesc, psDesc;
	if (vsRefl)
	{
		vsRefl->GetDesc(&vsDesc);

		if (vsDesc.ConstantBuffers)
		{
			shader.nVSCBuffers = vsDesc.ConstantBuffers;
			shader.vsConstants = new ID3D11Buffer *[shader.nVSCBuffers];
			shader.vsConstMem = new ubyte *[shader.nVSCBuffers];
			shader.vsDirty = new bool[shader.nVSCBuffers];
		}
	}
	if (gsRefl)
	{
		gsRefl->GetDesc(&gsDesc);

		if (gsDesc.ConstantBuffers)
		{
			shader.nGSCBuffers = gsDesc.ConstantBuffers;
			shader.gsConstants = new ID3D11Buffer *[shader.nGSCBuffers];
			shader.gsConstMem = new ubyte *[shader.nGSCBuffers];
			shader.gsDirty = new bool[shader.nGSCBuffers];
		}
	}
	if (psRefl)
	{
		psRefl->GetDesc(&psDesc);

		if (psDesc.ConstantBuffers)
		{
			shader.nPSCBuffers = psDesc.ConstantBuffers;
			shader.psConstants = new ID3D11Buffer *[shader.nPSCBuffers];
			shader.psConstMem = new ubyte *[shader.nPSCBuffers];
			shader.psDirty = new bool[shader.nPSCBuffers];
		}
	}

	D3D11_SHADER_BUFFER_DESC sbDesc;

	D3D11_BUFFER_DESC cbDesc;
	cbDesc.Usage = D3D11_USAGE_DEFAULT;//D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = 0;//D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;

	Array <Constant> constants;

	for (uint i = 0; i < shader.nVSCBuffers; i++)
	{
		vsRefl->GetConstantBufferByIndex(i)->GetDesc(&sbDesc);

		cbDesc.ByteWidth = sbDesc.Size;
		device->CreateBuffer(&cbDesc, NULL, &shader.vsConstants[i]);

		shader.vsConstMem[i] = new ubyte[sbDesc.Size];
		for (uint k = 0; k < sbDesc.Variables; k++)
		{
			D3D11_SHADER_VARIABLE_DESC vDesc;
			vsRefl->GetConstantBufferByIndex(i)->GetVariableByIndex(k)->GetDesc(&vDesc);

			Constant constant;
			size_t length = strlen(vDesc.Name);
			constant.name = new char[length + 1];
			strcpy(constant.name, vDesc.Name);
			constant.vsData = shader.vsConstMem[i] + vDesc.StartOffset;
			constant.gsData = NULL;
			constant.psData = NULL;
			constant.vsBuffer = i;
			constant.gsBuffer = -1;
			constant.psBuffer = -1;
			constants.add(constant);
		}

		shader.vsDirty[i] = false;
	}

	uint maxConst = constants.getCount();
	for (uint i = 0; i < shader.nGSCBuffers; i++)
	{
		gsRefl->GetConstantBufferByIndex(i)->GetDesc(&sbDesc);

		cbDesc.ByteWidth = sbDesc.Size;
		device->CreateBuffer(&cbDesc, NULL, &shader.gsConstants[i]);

		shader.gsConstMem[i] = new ubyte[sbDesc.Size];
		for (uint k = 0; k < sbDesc.Variables; k++)
		{
			D3D11_SHADER_VARIABLE_DESC vDesc;
			gsRefl->GetConstantBufferByIndex(i)->GetVariableByIndex(k)->GetDesc(&vDesc);

			int merge = -1;
			for (uint i = 0; i < maxConst; i++)
			{
				if (strcmp(constants[i].name, vDesc.Name) == 0)
				{
					merge = i;
					break;
				}
			}

			if (merge < 0)
			{
				Constant constant;
				size_t length = strlen(vDesc.Name);
				constant.name = new char[length + 1];
				strcpy(constant.name, vDesc.Name);
				constant.vsData = NULL;
				constant.gsData = shader.gsConstMem[i] + vDesc.StartOffset;
				constant.psData = NULL;
				constant.vsBuffer = -1;
				constant.gsBuffer = i;
				constant.psBuffer = -1;
				constants.add(constant);
			}
			else
			{
				constants[merge].gsData = shader.gsConstMem[i] + vDesc.StartOffset;
				constants[merge].gsBuffer = i;
			}
		}

		shader.gsDirty[i] = false;
	}

	maxConst = constants.getCount();
	for (uint i = 0; i < shader.nPSCBuffers; i++)
	{
		psRefl->GetConstantBufferByIndex(i)->GetDesc(&sbDesc);

		cbDesc.ByteWidth = sbDesc.Size;
		device->CreateBuffer(&cbDesc, NULL, &shader.psConstants[i]);

		shader.psConstMem[i] = new ubyte[sbDesc.Size];
		for (uint k = 0; k < sbDesc.Variables; k++)
		{
			D3D11_SHADER_VARIABLE_DESC vDesc;
			psRefl->GetConstantBufferByIndex(i)->GetVariableByIndex(k)->GetDesc(&vDesc);

			int merge = -1;
			for (uint i = 0; i < maxConst; i++)
			{
				if (strcmp(constants[i].name, vDesc.Name) == 0)
				{
					merge = i;
					break;
				}
			}

			if (merge < 0)
			{
				Constant constant;
				size_t length = strlen(vDesc.Name);
				constant.name = new char[length + 1];
				strcpy(constant.name, vDesc.Name);
				constant.vsData = NULL;
				constant.gsData = NULL;
				constant.psData = shader.psConstMem[i] + vDesc.StartOffset;
				constant.vsBuffer = -1;
				constant.gsBuffer = -1;
				constant.psBuffer = i;
				constants.add(constant);
			}
			else
			{
				constants[merge].psData = shader.psConstMem[i] + vDesc.StartOffset;
				constants[merge].psBuffer = i;
			}
		}

		shader.psDirty[i] = false;
	}

	shader.nConstants = constants.getCount();
	shader.constants = new Constant[shader.nConstants];
	memcpy(shader.constants, constants.getArray(), shader.nConstants * sizeof(Constant));
	qsort(shader.constants, shader.nConstants, sizeof(Constant), constantComp);

	uint nMaxVSRes = vsRefl? vsDesc.BoundResources : 0;
	uint nMaxGSRes = gsRefl? gsDesc.BoundResources : 0;
	uint nMaxPSRes = psRefl? psDesc.BoundResources : 0;

	int maxResources = nMaxVSRes + nMaxGSRes + nMaxPSRes;
	if (maxResources)
	{
		shader.textures = (Sampler *) malloc(maxResources * sizeof(Sampler));
		shader.samplers = (Sampler *) malloc(maxResources * sizeof(Sampler));
		shader.nTextures = 0;
		shader.nSamplers = 0;

		D3D11_SHADER_INPUT_BIND_DESC siDesc;
		for (uint i = 0; i < nMaxVSRes; i++)
		{
			vsRefl->GetResourceBindingDesc(i, &siDesc);

			if (siDesc.Type == D3D10_SIT_TEXTURE)
			{
				size_t length = strlen(siDesc.Name);
				shader.textures[shader.nTextures].name = new char[length + 1];
				strcpy(shader.textures[shader.nTextures].name, siDesc.Name);
				shader.textures[shader.nTextures].vsIndex = siDesc.BindPoint;
				shader.textures[shader.nTextures].gsIndex = -1;
				shader.textures[shader.nTextures].psIndex = -1;
				shader.nTextures++;
			}
			else if (siDesc.Type == D3D10_SIT_SAMPLER)
			{
				size_t length = strlen(siDesc.Name);
				shader.samplers[shader.nSamplers].name = new char[length + 1];
				strcpy(shader.samplers[shader.nSamplers].name, siDesc.Name);
				shader.samplers[shader.nSamplers].vsIndex = siDesc.BindPoint;
				shader.samplers[shader.nSamplers].gsIndex = -1;
				shader.samplers[shader.nSamplers].psIndex = -1;
				shader.nSamplers++;
			}
		}
		uint maxTexture = shader.nTextures;
		uint maxSampler = shader.nSamplers;
		for (uint i = 0; i < nMaxGSRes; i++)
		{
			gsRefl->GetResourceBindingDesc(i, &siDesc);

			if (siDesc.Type == D3D10_SIT_TEXTURE)
			{
				int merge = -1;
				for (uint i = 0; i < maxTexture; i++)
				{
					if (strcmp(shader.textures[i].name, siDesc.Name) == 0)
					{
						merge = i;
						break;
					}
				}
				if (merge < 0)
				{
					size_t length = strlen(siDesc.Name);
					shader.textures[shader.nTextures].name = new char[length + 1];
					strcpy(shader.textures[shader.nTextures].name, siDesc.Name);
					shader.textures[shader.nTextures].vsIndex = -1;
					shader.textures[shader.nTextures].gsIndex = siDesc.BindPoint;
					shader.textures[shader.nTextures].psIndex = -1;
					shader.nTextures++;
				}
				else
				{
					shader.textures[merge].gsIndex = siDesc.BindPoint;
				}
			}
			else if (siDesc.Type == D3D10_SIT_SAMPLER)
			{
				int merge = -1;
				for (uint i = 0; i < maxSampler; i++)
				{
					if (strcmp(shader.samplers[i].name, siDesc.Name) == 0)
					{
						merge = i;
						break;
					}
				}
				if (merge < 0)
				{
					size_t length = strlen(siDesc.Name);
					shader.samplers[shader.nSamplers].name = new char[length + 1];
					strcpy(shader.samplers[shader.nSamplers].name, siDesc.Name);
					shader.samplers[shader.nSamplers].vsIndex = -1;
					shader.samplers[shader.nSamplers].gsIndex = siDesc.BindPoint;
					shader.samplers[shader.nSamplers].psIndex = -1;
					shader.nSamplers++;
				}
				else
				{
					shader.samplers[merge].gsIndex = siDesc.BindPoint;
				}
			}
		}
		maxTexture = shader.nTextures;
		maxSampler = shader.nSamplers;
		for (uint i = 0; i < nMaxPSRes; i++)
		{
			psRefl->GetResourceBindingDesc(i, &siDesc);

			if (siDesc.Type == D3D10_SIT_TEXTURE)
			{
				int merge = -1;
				for (uint i = 0; i < maxTexture; i++)
				{
					if (strcmp(shader.textures[i].name, siDesc.Name) == 0)
					{
						merge = i;
						break;
					}
				}
				if (merge < 0)
				{
					size_t length = strlen(siDesc.Name);
					shader.textures[shader.nTextures].name = new char[length + 1];
					strcpy(shader.textures[shader.nTextures].name, siDesc.Name);
					shader.textures[shader.nTextures].vsIndex = -1;
					shader.textures[shader.nTextures].gsIndex = -1;
					shader.textures[shader.nTextures].psIndex = siDesc.BindPoint;
					shader.nTextures++;
				}
				else
				{
					shader.textures[merge].psIndex = siDesc.BindPoint;
				}
			}
			else if (siDesc.Type == D3D10_SIT_SAMPLER)
			{
				int merge = -1;
				for (uint i = 0; i < maxSampler; i++)
				{
					if (strcmp(shader.samplers[i].name, siDesc.Name) == 0)
					{
						merge = i;
						break;
					}
				}
				if (merge < 0)
				{
					size_t length = strlen(siDesc.Name);
					shader.samplers[shader.nSamplers].name = new char[length + 1];
					strcpy(shader.samplers[shader.nSamplers].name, siDesc.Name);
					shader.samplers[shader.nSamplers].vsIndex = -1;
					shader.samplers[shader.nSamplers].gsIndex = -1;
					shader.samplers[shader.nSamplers].psIndex = siDesc.BindPoint;
					shader.nSamplers++;
				}
				else
				{
					shader.samplers[merge].psIndex = siDesc.BindPoint;
				}
			}
		}
		shader.textures = (Sampler *) realloc(shader.textures, shader.nTextures * sizeof(Sampler));
		shader.samplers = (Sampler *) realloc(shader.samplers, shader.nSamplers * sizeof(Sampler));
		qsort(shader.textures, shader.nTextures, sizeof(Sampler), samplerComp);
		qsort(shader.samplers, shader.nSamplers, sizeof(Sampler), samplerComp);
	}

	if (vsRefl) vsRefl->Release();
	if (gsRefl) gsRefl->Release();
	if (psRefl) psRefl->Release();

	return shaders.add(shader);
}

VertexFormatID Direct3D11Renderer::addVertexFormat(const FormatDesc *formatDesc, const uint nAttribs, const ShaderID shader)
{
	static const DXGI_FORMAT formats[][4] =
	{
		{ DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT },
		{ DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_UNKNOWN,         DXGI_FORMAT_R16G16B16A16_FLOAT },
		{ DXGI_FORMAT_R8_UNORM,  DXGI_FORMAT_R8G8_UNORM,   DXGI_FORMAT_UNKNOWN,         DXGI_FORMAT_R8G8B8A8_UNORM     },
	};

	static const char *semantics[] =
	{
		NULL,
		"Position",
		"Texcoord",
		"Normal",
		"Tangent",
		"Binormal",
	};


	int index[6];
	memset(index, 0, sizeof(index));

	VertexFormat vf;
	memset(vf.vertexSize, 0, sizeof(vf.vertexSize));

	D3D11_INPUT_ELEMENT_DESC *desc = new D3D11_INPUT_ELEMENT_DESC[nAttribs];

	// Fill the vertex element array
	for (uint i = 0; i < nAttribs; i++)
	{
		int stream = formatDesc[i].stream;
		int size = formatDesc[i].size;
		desc[i].InputSlot = stream;
		desc[i].AlignedByteOffset = vf.vertexSize[stream];
		desc[i].SemanticName = semantics[formatDesc[i].type];
		desc[i].SemanticIndex = index[formatDesc[i].type]++;
		desc[i].Format = formats[formatDesc[i].format][size - 1];
		desc[i].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		desc[i].InstanceDataStepRate = 0;

		vf.vertexSize[stream] += size * getFormatSize(formatDesc[i].format);
	}

	HRESULT hr = device->CreateInputLayout(desc, nAttribs, shaders[shader].inputSignature->GetBufferPointer(), shaders[shader].inputSignature->GetBufferSize(), &vf.inputLayout);
	delete [] desc;

	if (FAILED(hr))
	{
		ErrorMsg("Couldn't create vertex declaration");
		return VF_NONE;
	}

	return vertexFormats.add(vf);
}

static const D3D11_USAGE usage[] =
{
	D3D11_USAGE_IMMUTABLE,
	D3D11_USAGE_DEFAULT,
	D3D11_USAGE_DYNAMIC,
};

VertexBufferID Direct3D11Renderer::addVertexBuffer(const long size, const BufferAccess bufferAccess, const void *data)
{
	VertexBuffer vb;
	vb.size = size;

	D3D11_BUFFER_DESC desc;
	desc.Usage = usage[bufferAccess];
	desc.ByteWidth = size;
	desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	desc.CPUAccessFlags = (bufferAccess == DYNAMIC)? D3D11_CPU_ACCESS_WRITE : 0;
	desc.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA vbData;
	vbData.pSysMem = data;
	vbData.SysMemPitch = 0;
	vbData.SysMemSlicePitch = 0;

	if (FAILED(device->CreateBuffer(&desc, data? &vbData : NULL, &vb.vertexBuffer)))
	{
        ErrorMsg("Couldn't create vertex buffer");
		return VB_NONE;
	}

	return vertexBuffers.add(vb);
}

IndexBufferID Direct3D11Renderer::addIndexBuffer(const uint nIndices, const uint indexSize, const BufferAccess bufferAccess, const void *data)
{
	IndexBuffer ib;
	ib.indexSize = indexSize;
	ib.nIndices = nIndices;

	D3D11_BUFFER_DESC desc;
	desc.Usage = usage[bufferAccess];
	desc.ByteWidth = nIndices * indexSize;
	desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	desc.CPUAccessFlags = (bufferAccess == DYNAMIC)? D3D11_CPU_ACCESS_WRITE : 0;
	desc.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA ibData;
	ibData.pSysMem = data;
	ibData.SysMemPitch = 0;
	ibData.SysMemSlicePitch = 0;

	if (FAILED(device->CreateBuffer(&desc, data? &ibData : NULL, &ib.indexBuffer)))
	{
        ErrorMsg("Couldn't create vertex buffer");
		return IB_NONE;
	}

	return indexBuffers.add(ib);
}

static const D3D11_FILTER filters[] =
{
	D3D11_FILTER_MIN_MAG_MIP_POINT,
	D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT,
	D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT,
	D3D11_FILTER_MIN_MAG_MIP_LINEAR,
	D3D11_FILTER_ANISOTROPIC,
	D3D11_FILTER_ANISOTROPIC,
};

static const D3D11_TEXTURE_ADDRESS_MODE address_modes[] =
{
	D3D11_TEXTURE_ADDRESS_WRAP,
	D3D11_TEXTURE_ADDRESS_CLAMP,
	D3D11_TEXTURE_ADDRESS_BORDER,
};

SamplerStateID Direct3D11Renderer::addSamplerState(const Filter filter, const AddressMode s, const AddressMode t, const AddressMode r, const float lod, const uint maxAniso, const int compareFunc, const float *border_color)
{
	SamplerState samplerState;

	D3D11_SAMPLER_DESC desc;
	desc.Filter = filters[filter];
	if (compareFunc)
		desc.Filter = (D3D11_FILTER) (desc.Filter | 0x80);
	desc.ComparisonFunc = (D3D11_COMPARISON_FUNC) compareFunc;
	desc.AddressU = address_modes[s];
	desc.AddressV = address_modes[t];
	desc.AddressW = address_modes[r];
	desc.MipLODBias = lod;
	desc.MaxAnisotropy = hasAniso(filter)? maxAniso : 1;
	if (border_color)
	{
		desc.BorderColor[0] = border_color[0];
		desc.BorderColor[1] = border_color[1];
		desc.BorderColor[2] = border_color[2];
		desc.BorderColor[3] = border_color[3];
	}
	else
	{
		desc.BorderColor[0] = 0;
		desc.BorderColor[1] = 0;
		desc.BorderColor[2] = 0;
		desc.BorderColor[3] = 0;
	}
	desc.MinLOD = 0;
	desc.MaxLOD = hasMipmaps(filter)? D3D11_FLOAT32_MAX : 0;

	if (FAILED(device->CreateSamplerState(&desc, &samplerState.samplerState)))
	{
		ErrorMsg("Couldn't create samplerstate");
		return SS_NONE;
	}

	return samplerStates.add(samplerState);
}

BlendStateID Direct3D11Renderer::addBlendState(const int srcFactorRGB, const int destFactorRGB, const int srcFactorAlpha, const int destFactorAlpha, const int blendModeRGB, const int blendModeAlpha, const int mask, const bool alphaToCoverage)
{
	BlendState blendState;

	BOOL blendEnable =
		srcFactorRGB   != D3D11_BLEND_ONE || destFactorRGB   != D3D11_BLEND_ZERO ||
		srcFactorAlpha != D3D11_BLEND_ONE || destFactorAlpha != D3D11_BLEND_ZERO;

	D3D11_BLEND_DESC desc;
	desc.AlphaToCoverageEnable = (BOOL) alphaToCoverage;
	desc.IndependentBlendEnable = FALSE;
	for (int i = 0; i < elementsOf(desc.RenderTarget); i++)
	{
		D3D11_RENDER_TARGET_BLEND_DESC &rt = desc.RenderTarget[i];

		rt.BlendEnable = blendEnable;
		rt.SrcBlend = (D3D11_BLEND) srcFactorRGB;
		rt.DestBlend = (D3D11_BLEND) destFactorRGB;
		rt.BlendOp = (D3D11_BLEND_OP) blendModeAlpha;
		rt.SrcBlendAlpha = (D3D11_BLEND) srcFactorAlpha;
		rt.DestBlendAlpha = (D3D11_BLEND) destFactorAlpha;
		rt.BlendOpAlpha = (D3D11_BLEND_OP) blendModeAlpha;
		rt.RenderTargetWriteMask = mask;
	}

	if (FAILED(device->CreateBlendState(&desc, &blendState.blendState)))
	{
		ErrorMsg("Couldn't create blendstate");
		return BS_NONE;
	}

	return blendStates.add(blendState);
}

DepthStateID Direct3D11Renderer::addDepthState(const bool depthTest, const bool depthWrite, const int depthFunc, const bool stencilTest, const uint8 stencilReadMask, const uint8 stencilWriteMask,
		const int stencilFuncFront, const int stencilFuncBack, const int stencilFailFront, const int stencilFailBack,
		const int depthFailFront, const int depthFailBack, const int stencilPassFront, const int stencilPassBack)
{
	DepthState depthState;

	D3D11_DEPTH_STENCIL_DESC desc;
	desc.DepthEnable = (BOOL) depthTest;
	desc.DepthWriteMask = depthWrite? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
	desc.DepthFunc = (D3D11_COMPARISON_FUNC) depthFunc;
	desc.StencilEnable = (BOOL) stencilTest;
	desc.StencilReadMask  = stencilReadMask;
	desc.StencilWriteMask = stencilWriteMask;
	desc.BackFace. StencilFunc = (D3D11_COMPARISON_FUNC) stencilFuncBack;
	desc.FrontFace.StencilFunc = (D3D11_COMPARISON_FUNC) stencilFuncFront;
	desc.BackFace. StencilDepthFailOp = (D3D11_STENCIL_OP) depthFailBack;
	desc.FrontFace.StencilDepthFailOp = (D3D11_STENCIL_OP) depthFailFront;
	desc.BackFace. StencilFailOp = (D3D11_STENCIL_OP) stencilFailBack;
	desc.FrontFace.StencilFailOp = (D3D11_STENCIL_OP) stencilFailFront;
	desc.BackFace. StencilPassOp = (D3D11_STENCIL_OP) stencilPassBack;
	desc.FrontFace.StencilPassOp = (D3D11_STENCIL_OP) stencilPassFront;

	if (FAILED(device->CreateDepthStencilState(&desc, &depthState.dsState)))
	{
		ErrorMsg("Couldn't create depthstate");
		return DS_NONE;
	}

	return depthStates.add(depthState);	
}

RasterizerStateID Direct3D11Renderer::addRasterizerState(const int cullMode, const int fillMode, const bool multiSample, const bool scissor, const float depthBias, const float slopeDepthBias)
{
	RasterizerState rasterizerState;

	D3D11_RASTERIZER_DESC desc;
	desc.CullMode = (D3D11_CULL_MODE) cullMode;
	desc.FillMode = (D3D11_FILL_MODE) fillMode;
	desc.FrontCounterClockwise = FALSE;
	desc.DepthBias = (INT) depthBias;
	desc.DepthBiasClamp = 0.0f;
	desc.SlopeScaledDepthBias = slopeDepthBias;
	desc.AntialiasedLineEnable = FALSE;
	desc.DepthClipEnable = TRUE;
	desc.MultisampleEnable = (BOOL) multiSample;
	desc.ScissorEnable = (BOOL) scissor;

	if (FAILED(device->CreateRasterizerState(&desc, &rasterizerState.rsState)))
	{
		ErrorMsg("Couldn't create rasterizerstate");
		return RS_NONE;
	}

	return rasterizerStates.add(rasterizerState);
}

const Sampler *getSampler(const Sampler *samplers, const int count, const char *name)
{
	int minSampler = 0;
	int maxSampler = count - 1;

	// Do a quick lookup in the sorted table with a binary search
	while (minSampler <= maxSampler)
	{
		int currSampler = (minSampler + maxSampler) >> 1;
        int res = strcmp(name, samplers[currSampler].name);
		if (res == 0)
			return samplers + currSampler;
		else if (res > 0)
            minSampler = currSampler + 1;
		else
            maxSampler = currSampler - 1;
	}

	return NULL;
}

void Direct3D11Renderer::setTexture(const char *textureName, const TextureID texture)
{
	ASSERT(selectedShader != SHADER_NONE);

	const Sampler *s = getSampler(shaders[selectedShader].textures, shaders[selectedShader].nTextures, textureName);
	if (s)
	{
		if (s->vsIndex >= 0)
		{
			selectedTexturesVS[s->vsIndex] = texture;
			selectedTextureSlicesVS[s->vsIndex] = NO_SLICE;
		}
		if (s->gsIndex >= 0)
		{
			selectedTexturesGS[s->gsIndex] = texture;
			selectedTextureSlicesGS[s->gsIndex] = NO_SLICE;
		}
		if (s->psIndex >= 0)
		{
			selectedTexturesPS[s->psIndex] = texture;
			selectedTextureSlicesPS[s->psIndex] = NO_SLICE;
		}
	}
	else
	{
#ifdef DEBUG
		char str[256];
		sprintf(str, "Invalid texture \"%s\"", textureName);
		outputDebugString(str);
#endif
	}
}

void Direct3D11Renderer::setTexture(const char *textureName, const TextureID texture, const SamplerStateID samplerState)
{
	ASSERT(selectedShader != SHADER_NONE);

	const Sampler *s = getSampler(shaders[selectedShader].textures, shaders[selectedShader].nTextures, textureName);
	if (s)
	{
		if (s->vsIndex >= 0)
		{
			selectedTexturesVS[s->vsIndex] = texture;
			selectedTextureSlicesVS[s->vsIndex] = NO_SLICE;
			selectedSamplerStatesVS[s->vsIndex] = samplerState;
		}
		if (s->gsIndex >= 0)
		{
			selectedTexturesGS[s->gsIndex] = texture;
			selectedTextureSlicesGS[s->gsIndex] = NO_SLICE;
			selectedSamplerStatesGS[s->gsIndex] = samplerState;
		}
		if (s->psIndex >= 0)
		{
			selectedTexturesPS[s->psIndex] = texture;
			selectedTextureSlicesPS[s->psIndex] = NO_SLICE;
			selectedSamplerStatesPS[s->psIndex] = samplerState;
		}
	}
	else
	{
#ifdef DEBUG
		char str[256];
		sprintf(str, "Invalid texture \"%s\"", textureName);
		outputDebugString(str);
#endif
	}
}

void Direct3D11Renderer::setTextureSlice(const char *textureName, const TextureID texture, const int slice)
{
	ASSERT(selectedShader != SHADER_NONE);

	const Sampler *s = getSampler(shaders[selectedShader].textures, shaders[selectedShader].nTextures, textureName);
	if (s)
	{
		if (s->vsIndex >= 0)
		{
			selectedTexturesVS[s->vsIndex] = texture;
			selectedTextureSlicesVS[s->vsIndex] = slice;
		}
		if (s->gsIndex >= 0)
		{
			selectedTexturesGS[s->gsIndex] = texture;
			selectedTextureSlicesGS[s->gsIndex] = slice;
		}
		if (s->psIndex >= 0)
		{
			selectedTexturesPS[s->psIndex] = texture;
			selectedTextureSlicesPS[s->psIndex] = slice;
		}
	}
	else
	{
#ifdef DEBUG
		char str[256];
		sprintf(str, "Invalid texture \"%s\"", textureName);
		outputDebugString(str);
#endif
	}
}


bool fillSRV(ID3D11ShaderResourceView **dest, int &min, int &max, const TextureID selectedTextures[], TextureID currentTextures[], const TextureID selectedTextureSlices[], TextureID currentTextureSlices[], const Texture *textures)
{
	min = 0;
	do
	{
		if (selectedTextures[min] != currentTextures[min] || selectedTextureSlices[min] != currentTextureSlices[min])
		{
			max = MAX_TEXTUREUNIT;
			do
			{
				max--;
			} while (selectedTextures[max] == currentTextures[max] && selectedTextureSlices[max] != currentTextureSlices[max]);

			for (int i = min; i <= max; i++)
			{
				if (selectedTextures[i] != TEXTURE_NONE)
				{
					if (selectedTextureSlices[i] == NO_SLICE)
					{
						*dest++ = textures[selectedTextures[i]].srv;
					}
					else
					{
						*dest++ = textures[selectedTextures[i]].srvArray[selectedTextureSlices[i]];
					}
				}
				else
				{
					*dest++ = NULL;
				}
				currentTextures[i] = selectedTextures[i];
				currentTextureSlices[i] = selectedTextureSlices[i];
			}
			return true;
		}
		min++;
	} while (min < MAX_TEXTUREUNIT);

	return false;
}

void Direct3D11Renderer::applyTextures()
{
	ID3D11ShaderResourceView *srViews[MAX_TEXTUREUNIT];

	int min, max;
	if (fillSRV(srViews, min, max, selectedTexturesVS, currentTexturesVS, selectedTextureSlicesVS, currentTextureSlicesVS, textures.getArray()))
		context->VSSetShaderResources(min, max - min + 1, srViews);

	if (fillSRV(srViews, min, max, selectedTexturesGS, currentTexturesGS, selectedTextureSlicesGS, currentTextureSlicesGS, textures.getArray()))
		context->GSSetShaderResources(min, max - min + 1, srViews);

	if (fillSRV(srViews, min, max, selectedTexturesPS, currentTexturesPS, selectedTextureSlicesPS, currentTextureSlicesPS, textures.getArray()))
		context->PSSetShaderResources(min, max - min + 1, srViews);
}

void Direct3D11Renderer::setSamplerState(const char *samplerName, const SamplerStateID samplerState)
{
	ASSERT(selectedShader != SHADER_NONE);

	const Sampler *s = getSampler(shaders[selectedShader].samplers, shaders[selectedShader].nSamplers, samplerName);
	if (s)
	{
		if (s->vsIndex >= 0) selectedSamplerStatesVS[s->vsIndex] = samplerState;
		if (s->gsIndex >= 0) selectedSamplerStatesGS[s->gsIndex] = samplerState;
		if (s->psIndex >= 0) selectedSamplerStatesPS[s->psIndex] = samplerState;
	}
	else
	{
#ifdef DEBUG
		char str[256];
		sprintf(str, "Invalid samplerstate \"%s\"", samplerName);
		outputDebugString(str);
#endif
	}
}

bool fillSS(ID3D11SamplerState **dest, int &min, int &max, const SamplerStateID selectedSamplerStates[], SamplerStateID currentSamplerStates[], const SamplerState *samplerStates)
{
	min = 0;
	do
	{
		if (selectedSamplerStates[min] != currentSamplerStates[min])
		{
			max = MAX_SAMPLERSTATE;
			do
			{
				max--;
			} while (selectedSamplerStates[max] == currentSamplerStates[max]);

			for (int i = min; i <= max; i++)
			{
				if (selectedSamplerStates[i] != SS_NONE)
				{
					*dest++ = samplerStates[selectedSamplerStates[i]].samplerState;
				}
				else
				{
					*dest++ = NULL;
				}
				currentSamplerStates[i] = selectedSamplerStates[i];
			}
			return true;
		}
		min++;
	} while (min < MAX_SAMPLERSTATE);

	return false;
}

void Direct3D11Renderer::applySamplerStates()
{
	ID3D11SamplerState *samplers[MAX_SAMPLERSTATE];

	int min, max;
	if (fillSS(samplers, min, max, selectedSamplerStatesVS, currentSamplerStatesVS, samplerStates.getArray()))
		context->VSSetSamplers(min, max - min + 1, samplers);

	if (fillSS(samplers, min, max, selectedSamplerStatesGS, currentSamplerStatesGS, samplerStates.getArray()))
		context->GSSetSamplers(min, max - min + 1, samplers);

	if (fillSS(samplers, min, max, selectedSamplerStatesPS, currentSamplerStatesPS, samplerStates.getArray()))
		context->PSSetSamplers(min, max - min + 1, samplers);
}

void Direct3D11Renderer::setShaderConstantRaw(const char *name, const void *data, const int size)
{
	int minConstant = 0;
	int maxConstant = shaders[selectedShader].nConstants - 1;
	Constant *constants = shaders[selectedShader].constants;

	// Do a quick lookup in the sorted table with a binary search
	while (minConstant <= maxConstant)
	{
		int currConstant = (minConstant + maxConstant) >> 1;
		int res = strcmp(name, constants[currConstant].name);
		if (res == 0)
		{
			Constant *c = constants + currConstant;

			if (c->vsData)
			{
				if (memcmp(c->vsData, data, size))
				{
					memcpy(c->vsData, data, size);
					shaders[selectedShader].vsDirty[c->vsBuffer] = true;
				}
			}
			if (c->gsData)
			{
				if (memcmp(c->gsData, data, size))
				{
					memcpy(c->gsData, data, size);
					shaders[selectedShader].gsDirty[c->gsBuffer] = true;
				}
			}
			if (c->psData)
			{
				if (memcmp(c->psData, data, size))
				{
					memcpy(c->psData, data, size);
					shaders[selectedShader].psDirty[c->psBuffer] = true;
				}
			}
			return;

		}
		else if (res > 0)
		{
			minConstant = currConstant + 1;
		}
		else
		{
			maxConstant = currConstant - 1;
		}
	}

#ifdef DEBUG
	char str[256];
	sprintf(str, "Invalid constant \"%s\"", name);
	outputDebugString(str);
#endif
}

void Direct3D11Renderer::applyConstants()
{
	if (currentShader != SHADER_NONE)
	{
		Shader *shader = &shaders[currentShader];

		for (uint i = 0; i < shader->nVSCBuffers; i++)
		{
			if (shader->vsDirty[i])
			{
				context->UpdateSubresource(shader->vsConstants[i], 0, NULL, shader->vsConstMem[i], 0, 0);
				shader->vsDirty[i] = false;
			}
		}
		for (uint i = 0; i < shader->nGSCBuffers; i++)
		{
			if (shader->gsDirty[i])
			{
				context->UpdateSubresource(shader->gsConstants[i], 0, NULL, shader->gsConstMem[i], 0, 0);
				shader->gsDirty[i] = false;
			}
		}
		for (uint i = 0; i < shader->nPSCBuffers; i++)
		{
			if (shader->psDirty[i])
			{
				context->UpdateSubresource(shader->psConstants[i], 0, NULL, shader->psConstMem[i], 0, 0);
				shader->psDirty[i] = false;
			}
		}
	}
}

void Direct3D11Renderer::changeRenderTargets(const TextureID *colorRTs, const uint nRenderTargets, const TextureID depthRT, const int depthSlice, const int *slices)
{
	// Reset bound textures
	for (int i = 0; i < MAX_TEXTUREUNIT; i++)
	{
		selectedTexturesVS[i] = TEXTURE_NONE;
		selectedTexturesGS[i] = TEXTURE_NONE;
		selectedTexturesPS[i] = TEXTURE_NONE;
	}
	applyTextures();

	ID3D11RenderTargetView *rtv[16];
	ID3D11DepthStencilView *dsv;

	if (depthRT == FB_DEPTH)
		dsv = depthBufferDSV;
	else if (depthRT == TEXTURE_NONE)
		dsv = NULL;
	else if (depthSlice == NO_SLICE)
		dsv = textures[depthRT].dsv;
	else
		dsv = textures[depthRT].dsvArray[depthSlice];

	currentDepthRT = depthRT;
	currentDepthSlice = depthSlice;

	for (uint i = 0; i < nRenderTargets; i++)
	{
		TextureID rt = colorRTs[i];
		int slice = NO_SLICE;
		if (slices == NULL || slices[i] == NO_SLICE)
		{
			if (rt == FB_COLOR)
				rtv[i] = backBufferRTV;
			else
				rtv[i] = textures[rt].rtv;
		}
		else
		{
			slice = slices[i];
			rtv[i] = textures[rt].rtvArray[slice];
		}

		currentColorRT[i] = rt;
		currentColorRTSlice[i] = slice;
	}

	for (uint i = nRenderTargets; i < MAX_MRTS; i++)
	{
		currentColorRT[i] = TEXTURE_NONE;
		currentColorRTSlice[i] = NO_SLICE;
	}

	context->OMSetRenderTargets(nRenderTargets, rtv, dsv);


	int w, h;
	if (nRenderTargets > 0)
	{
		TextureID rt = colorRTs[0];
		if (rt == FB_COLOR)
		{
			w = viewportWidth;
			h = viewportHeight;
		}
		else
		{
			w = textures[rt].width;
			h = textures[rt].height;
		}
	}
	else
	{
		w = textures[depthRT].width;
		h = textures[depthRT].height;
	}

	D3D11_VIEWPORT vp;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	vp.MinDepth = 0;
	vp.MaxDepth = 1;
	vp.Width  = (float) w;
	vp.Height = (float) h;
	context->RSSetViewports(1, &vp);
}

void Direct3D11Renderer::changeToMainFramebuffer()
{
	// Reset bound textures
	for (int i = 0; i < MAX_TEXTUREUNIT; i++)
	{
		selectedTexturesVS[i] = TEXTURE_NONE;
		selectedTexturesGS[i] = TEXTURE_NONE;
		selectedTexturesPS[i] = TEXTURE_NONE;
	}
	applyTextures();

	context->OMSetRenderTargets(1, &backBufferRTV, depthBufferDSV);

	D3D11_VIEWPORT vp = { 0, 0, (float) viewportWidth, (float) viewportHeight, 0.0f, 1.0f };
	context->RSSetViewports(1, &vp);

	currentColorRT[0] = FB_COLOR;
	currentColorRTSlice[0] = NO_SLICE;

	for (uint i = 1; i < MAX_MRTS; i++)
	{
		currentColorRT[i] = TEXTURE_NONE;
		currentColorRTSlice[i] = NO_SLICE;
	}
	currentDepthRT = FB_DEPTH;
	currentDepthSlice = NO_SLICE;
}

void Direct3D11Renderer::changeShader(const ShaderID shaderID)
{
	if (shaderID != currentShader)
	{
		if (shaderID == SHADER_NONE)
		{
			context->VSSetShader(NULL, NULL, 0);
			context->GSSetShader(NULL, NULL, 0);
			context->PSSetShader(NULL, NULL, 0);
		}
		else
		{
			context->VSSetShader(shaders[shaderID].vertexShader,   NULL, 0);
			context->GSSetShader(shaders[shaderID].geometryShader, NULL, 0);
			context->PSSetShader(shaders[shaderID].pixelShader,    NULL, 0);

			if (shaders[shaderID].nVSCBuffers) context->VSSetConstantBuffers(0, shaders[shaderID].nVSCBuffers, shaders[shaderID].vsConstants);
			if (shaders[shaderID].nGSCBuffers) context->GSSetConstantBuffers(0, shaders[shaderID].nGSCBuffers, shaders[shaderID].gsConstants);
			if (shaders[shaderID].nPSCBuffers) context->PSSetConstantBuffers(0, shaders[shaderID].nPSCBuffers, shaders[shaderID].psConstants);
		}

		currentShader = shaderID;
	}
}

void Direct3D11Renderer::changeVertexFormat(const VertexFormatID vertexFormatID)
{
	if (vertexFormatID != currentVertexFormat)
	{
		if (vertexFormatID == VF_NONE)
		{
			context->IASetInputLayout(NULL);
		}
		else
		{
			context->IASetInputLayout(vertexFormats[vertexFormatID].inputLayout);

			/*if (currentVertexFormat != VF_NONE){
				for (int i = 0; i < MAX_VERTEXSTREAM; i++){
					if (vertexFormats[vertexFormatID].vertexSize[i] != vertexFormats[currentVertexFormat].vertexSize[i]){
						currentVertexBuffers[i] = VB_INVALID;
					}
				}
			}*/
		}

		currentVertexFormat = vertexFormatID;
	}
}

void Direct3D11Renderer::changeVertexBuffer(const int stream, const VertexBufferID vertexBufferID, const intptr offset)
{
	if (vertexBufferID != currentVertexBuffers[stream] || offset != currentOffsets[stream])
	{
		UINT strides[1];
		UINT offsets[1];
		if (vertexBufferID == VB_NONE)
		{
			strides[0] = 0;
			offsets[0] = 0;
			ID3D11Buffer *null[] = { NULL };
			context->IASetVertexBuffers(stream, 1, null, strides, offsets);
		}
		else
		{
			strides[0] = vertexFormats[currentVertexFormat].vertexSize[stream];
			offsets[0] = (UINT) offset;
			context->IASetVertexBuffers(stream, 1, &vertexBuffers[vertexBufferID].vertexBuffer, strides, offsets);
		}

		currentVertexBuffers[stream] = vertexBufferID;
		currentOffsets[stream] = offset;
	}
}

void Direct3D11Renderer::changeIndexBuffer(const IndexBufferID indexBufferID)
{
	if (indexBufferID != currentIndexBuffer)
	{
		if (indexBufferID == IB_NONE)
		{
			context->IASetIndexBuffer(NULL, DXGI_FORMAT_UNKNOWN, 0);
		}
		else
		{
			DXGI_FORMAT format = indexBuffers[indexBufferID].indexSize < 4? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
			context->IASetIndexBuffer(indexBuffers[indexBufferID].indexBuffer, format, 0);
		}

		currentIndexBuffer = indexBufferID;
	}
}

void Direct3D11Renderer::changeBlendState(const BlendStateID blendState, const uint sampleMask)
{
	if (blendState != currentBlendState || sampleMask != currentSampleMask)
	{
		if (blendState == BS_NONE)
			context->OMSetBlendState(NULL, float4(0, 0, 0, 0), sampleMask);
		else
			context->OMSetBlendState(blendStates[blendState].blendState, float4(0, 0, 0, 0), sampleMask);

		currentBlendState = blendState;
		currentSampleMask = sampleMask;
	}
}

void Direct3D11Renderer::changeDepthState(const DepthStateID depthState, const uint stencilRef)
{
	if (depthState != currentDepthState || stencilRef != currentStencilRef)
	{
		if (depthState == DS_NONE)
			context->OMSetDepthStencilState(NULL, stencilRef);
		else
			context->OMSetDepthStencilState(depthStates[depthState].dsState, stencilRef);

		currentDepthState = depthState;
		currentStencilRef = stencilRef;
	}
}

void Direct3D11Renderer::changeRasterizerState(const RasterizerStateID rasterizerState)
{
	if (rasterizerState != currentRasterizerState)
	{
		if (rasterizerState == RS_NONE)
			context->RSSetState(rasterizerStates[0].rsState);
		else
			context->RSSetState(rasterizerStates[rasterizerState].rsState);

		currentRasterizerState = rasterizerState;
	}
}

void Direct3D11Renderer::clear(const bool clearColor, const bool clearDepth, const bool clearStencil, const float *color, const float depth, const uint stencil)
{
	if (clearColor)
	{
		if (currentColorRT[0] == FB_COLOR)
		{
			context->ClearRenderTargetView(backBufferRTV, color);
		}

		for (int i = 0; i < MAX_MRTS; i++)
		{
			if (currentColorRT[i] >= 0)
			{
				if (currentColorRTSlice[i] == NO_SLICE)
					context->ClearRenderTargetView(textures[currentColorRT[i]].rtv, color);
				else
					context->ClearRenderTargetView(textures[currentColorRT[i]].rtvArray[currentColorRTSlice[i]], color);
			}
		}
	}

	if (clearDepth || clearStencil)
	{
		UINT clearFlags = 0;
		if (clearDepth)   clearFlags |= D3D11_CLEAR_DEPTH;
		if (clearStencil) clearFlags |= D3D11_CLEAR_STENCIL;

		if (currentDepthRT == FB_DEPTH)
		{
			context->ClearDepthStencilView(depthBufferDSV, clearFlags, depth, stencil);
		}
		else if (currentDepthRT >= 0)
		{
			if (currentDepthSlice == NO_SLICE)
				context->ClearDepthStencilView(textures[currentDepthRT].dsv, clearFlags, depth, stencil);
			else
				context->ClearDepthStencilView(textures[currentDepthRT].dsvArray[currentDepthSlice], clearFlags, depth, stencil);
		}
	}
}

void Direct3D11Renderer::clearRenderTarget(const TextureID renderTarget, const float4 &color, const int slice)
{
	if (slice == NO_SLICE)
		context->ClearRenderTargetView(textures[renderTarget].rtv, color);
	else
		context->ClearRenderTargetView(textures[renderTarget].rtvArray[slice], color);
}

void Direct3D11Renderer::clearDepthTarget(const TextureID depthTarget, const float depth, const int slice)
{
	if (slice == NO_SLICE)
		context->ClearDepthStencilView(textures[depthTarget].dsv, D3D11_CLEAR_DEPTH, depth, 0);
	else
		context->ClearDepthStencilView(textures[depthTarget].dsvArray[slice], D3D11_CLEAR_DEPTH, depth, 0);
}


static const D3D11_PRIMITIVE_TOPOLOGY d3dPrim[] =
{
	D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
	D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED, // Triangle fans not supported
	D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
	D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED, // Quads not supported
	D3D11_PRIMITIVE_TOPOLOGY_LINELIST,
	D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP,
	D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED, // Line loops not supported
	D3D11_PRIMITIVE_TOPOLOGY_POINTLIST,
};

void Direct3D11Renderer::drawArrays(const Primitives primitives, const int firstVertex, const int nVertices)
{
	context->IASetPrimitiveTopology(d3dPrim[primitives]);
	context->Draw(nVertices, firstVertex);
}

void Direct3D11Renderer::drawElements(const Primitives primitives, const int firstIndex, const int nIndices, const int firstVertex, const int nVertices)
{
	context->IASetPrimitiveTopology(d3dPrim[primitives]);
	context->DrawIndexed(nIndices, firstIndex, 0);
}

void Direct3D11Renderer::setup2DMode(const float left, const float right, const float top, const float bottom)
{
	scaleBias2D.x = 2.0f / (right - left);
	scaleBias2D.y = 2.0f / (top - bottom);
	scaleBias2D.z = -1.0f;
	scaleBias2D.w =  1.0f;
}

const char *plainVS =
"float4 scaleBias;"
"float4 main(float4 position: Position): SV_Position {"
"	position.xy = position.xy * scaleBias.xy + scaleBias.zw;"
"	return position;"
"}";

const char *plainPS =
"float4 color;"
"float4 main() : SV_Target {"
"	return color;"
"}";

const char *texDefs =
"struct VsIn {"
"	float4 position: Position;"
"	float2 texCoord: TexCoord;"
"};"
"struct PsIn {"
"	float4 position: SV_Position;"
"	float2 texCoord: TexCoord;"
"};\n";

const char *texVS =
"float4 scaleBias;"
"PsIn main(VsIn In){"
"	PsIn Out;"
"	Out.position = In.position;"
"	Out.position.xy = Out.position.xy * scaleBias.xy + scaleBias.zw;"
"	Out.texCoord = In.texCoord;"
"	return Out;"
"}";

const char *texPS =
"Texture2D Base: register(t0);"
"SamplerState base: register(s0);"
"float4 color;"
"float4 main(PsIn In) : SV_Target {"
"	return Base.Sample(base, In.texCoord) * color;"
"}";

void Direct3D11Renderer::drawPlain(const Primitives primitives, vec2 *vertices, const uint nVertices, const BlendStateID blendState, const DepthStateID depthState, const vec4 *color)
{
	int size = nVertices * sizeof(vec2);

	UINT stride = sizeof(vec2);
	UINT offset = copyToRollingVB(vertices, size);

	if (plainShader == SHADER_NONE)
	{
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
	setVertexBuffer(0, rollingVB, offset);
	apply();

	context->IASetPrimitiveTopology(d3dPrim[primitives]);
	context->Draw(nVertices, 0);
}

void Direct3D11Renderer::drawTextured(const Primitives primitives, TexVertex *vertices, const uint nVertices, const TextureID texture, const SamplerStateID samplerState, const BlendStateID blendState, const DepthStateID depthState, const vec4 *color)
{
	int size = nVertices * sizeof(TexVertex);

	UINT stride = sizeof(TexVertex);
	UINT offset = copyToRollingVB(vertices, size);

	if (texShader == SHADER_NONE)
	{
		texShader = addShader(texVS, NULL, texPS, 0, 0, 0, texDefs);

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
	setTexture("Base", texture);
	setSamplerState("base", samplerState);
	setBlendState(blendState);
	setDepthState(depthState);
	setVertexFormat(texVF);
	setVertexBuffer(0, rollingVB, offset);
	apply();

	context->IASetPrimitiveTopology(d3dPrim[primitives]);
	context->Draw(nVertices, 0);
}

ID3D11ShaderResourceView *Direct3D11Renderer::createSRV(ID3D11Resource *resource, DXGI_FORMAT format, const int firstSlice, const int sliceCount)
{
	D3D11_RESOURCE_DIMENSION type;
	resource->GetType(&type);

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	ID3D11ShaderResourceView *srv;

	switch (type)
	{
		case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
			D3D11_TEXTURE1D_DESC desc1d;
			((ID3D11Texture1D *) resource)->GetDesc(&desc1d);

			srvDesc.Format = (format != DXGI_FORMAT_UNKNOWN)? format : desc1d.Format;
			if (desc1d.ArraySize > 1)
			{
				srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
				srvDesc.Texture1DArray.FirstArraySlice = 0;
				srvDesc.Texture1DArray.ArraySize = desc1d.ArraySize;
				srvDesc.Texture1DArray.MostDetailedMip = 0;
				srvDesc.Texture1DArray.MipLevels = desc1d.MipLevels;
			}
			else
			{
				srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1D;
				srvDesc.Texture1D.MostDetailedMip = 0;
				srvDesc.Texture1D.MipLevels = desc1d.MipLevels;
			}
			break;
		case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
			D3D11_TEXTURE2D_DESC desc2d;
			((ID3D11Texture2D *) resource)->GetDesc(&desc2d);

			srvDesc.Format = (format != DXGI_FORMAT_UNKNOWN)? format : desc2d.Format;
			if (desc2d.ArraySize > 1)
			{
				if (desc2d.MiscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE)
				{
					if (desc2d.ArraySize == 6)
					{
						srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
						srvDesc.TextureCube.MostDetailedMip = 0;
						srvDesc.TextureCube.MipLevels = desc2d.MipLevels;
					}
					else
					{
						srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBEARRAY;
						srvDesc.TextureCubeArray.MostDetailedMip = 0;
						srvDesc.TextureCubeArray.MipLevels = desc2d.MipLevels;
						srvDesc.TextureCubeArray.First2DArrayFace = 0;
						srvDesc.TextureCubeArray.NumCubes = desc2d.ArraySize / 6;
					}
				}
				else
				{
					srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
					if (firstSlice < 0)
					{
						srvDesc.Texture2DArray.FirstArraySlice = 0;
						srvDesc.Texture2DArray.ArraySize = desc2d.ArraySize;
					}
					else
					{
						srvDesc.Texture2DArray.FirstArraySlice = firstSlice;
						if (sliceCount < 0)
							srvDesc.Texture2DArray.ArraySize = 1;
						else
							srvDesc.Texture2DArray.ArraySize = sliceCount;
					}
					srvDesc.Texture2DArray.MostDetailedMip = 0;
					srvDesc.Texture2DArray.MipLevels = desc2d.MipLevels;
				}
			}
			else
			{
				srvDesc.ViewDimension = (desc2d.SampleDesc.Count > 1)? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
				srvDesc.Texture2D.MostDetailedMip = 0;
				srvDesc.Texture2D.MipLevels = desc2d.MipLevels;
			}
			break;
		case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
			D3D11_TEXTURE3D_DESC desc3d;
			((ID3D11Texture3D *) resource)->GetDesc(&desc3d);

			srvDesc.Format = (format != DXGI_FORMAT_UNKNOWN)? format : desc3d.Format;
			srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
			srvDesc.Texture3D.MostDetailedMip = 0;
			srvDesc.Texture3D.MipLevels = desc3d.MipLevels;

			break;
		default:
			ErrorMsg("Unsupported type");
			return NULL;
	}

	if (FAILED(device->CreateShaderResourceView(resource, &srvDesc, &srv)))
	{
		ErrorMsg("CreateShaderResourceView failed");
		return NULL;
	}

	return srv;
}

ID3D11RenderTargetView *Direct3D11Renderer::createRTV(ID3D11Resource *resource, DXGI_FORMAT format, const int firstSlice, const int sliceCount)
{
	D3D11_RESOURCE_DIMENSION type;
	resource->GetType(&type);

	D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
	ID3D11RenderTargetView *rtv;

	switch (type)
	{
		case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
			D3D11_TEXTURE2D_DESC desc2d;
			((ID3D11Texture2D *) resource)->GetDesc(&desc2d);

			rtvDesc.Format = (format != DXGI_FORMAT_UNKNOWN)? format : desc2d.Format;
			if (desc2d.ArraySize > 1)
			{
				rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
				if (firstSlice < 0)
				{
					rtvDesc.Texture2DArray.FirstArraySlice = 0;
					rtvDesc.Texture2DArray.ArraySize = desc2d.ArraySize;
				}
				else
				{
					rtvDesc.Texture2DArray.FirstArraySlice = firstSlice;
					if (sliceCount < 0)
						rtvDesc.Texture2DArray.ArraySize = 1;
					else
						rtvDesc.Texture2DArray.ArraySize = sliceCount;
				}
				rtvDesc.Texture2DArray.MipSlice = 0;
			}
			else
			{
				rtvDesc.ViewDimension = (desc2d.SampleDesc.Count > 1)? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
				rtvDesc.Texture2D.MipSlice = 0;
			}
			break;
		case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
			D3D11_TEXTURE3D_DESC desc3d;
			((ID3D11Texture3D *) resource)->GetDesc(&desc3d);

			rtvDesc.Format = (format != DXGI_FORMAT_UNKNOWN)? format : desc3d.Format;
			rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
			if (firstSlice < 0)
			{
				rtvDesc.Texture3D.FirstWSlice = 0;
				rtvDesc.Texture3D.WSize = desc3d.Depth;
			}
			else
			{
				rtvDesc.Texture3D.FirstWSlice = firstSlice;
				if (sliceCount < 0)
					rtvDesc.Texture3D.WSize = 1;
				else
					rtvDesc.Texture3D.WSize = sliceCount;
			}
			rtvDesc.Texture3D.MipSlice = 0;
			break;
		default:
			ErrorMsg("Unsupported type");
			return NULL;
	}

	if (FAILED(device->CreateRenderTargetView(resource, &rtvDesc, &rtv)))
	{
		ErrorMsg("CreateRenderTargetView failed");
		return NULL;
	}

	return rtv;
}

ID3D11DepthStencilView *Direct3D11Renderer::createDSV(ID3D11Resource *resource, DXGI_FORMAT format, const int firstSlice, const int sliceCount)
{
	D3D11_RESOURCE_DIMENSION type;
	resource->GetType(&type);

	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
	ID3D11DepthStencilView *dsv;

	switch (type)
	{
		case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
			D3D11_TEXTURE2D_DESC desc2d;
			((ID3D11Texture2D *) resource)->GetDesc(&desc2d);

			dsvDesc.Format = (format != DXGI_FORMAT_UNKNOWN)? format : desc2d.Format;
			if (desc2d.ArraySize > 1)
			{
				dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
				if (firstSlice < 0)
				{
					dsvDesc.Texture2DArray.FirstArraySlice = 0;
					dsvDesc.Texture2DArray.ArraySize = desc2d.ArraySize;
					dsvDesc.Texture2DArray.MipSlice = 0;
				}
				else
				{
					dsvDesc.Texture2DArray.FirstArraySlice = firstSlice;
					if (sliceCount < 0)
						dsvDesc.Texture2DArray.ArraySize = 1;
					else
						dsvDesc.Texture2DArray.ArraySize = sliceCount;
					dsvDesc.Texture2DArray.MipSlice = 0;
				}
			}
			else
			{
				dsvDesc.ViewDimension = (desc2d.SampleDesc.Count > 1)? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
				dsvDesc.Texture2D.MipSlice = 0;
			}
			break;
		default:
			ErrorMsg("Unsupported type");
			return NULL;
	}

	dsvDesc.Flags = 0;

	if (FAILED(device->CreateDepthStencilView(resource, &dsvDesc, &dsv)))
	{
		ErrorMsg("CreateDepthStencilView failed");
		return NULL;
	}

	return dsv;
}

ubyte *Direct3D11Renderer::mapRollingVB(const uint size)
{
	ASSERT(size <= ROLLING_VB_SIZE);

	if (rollingVB == VB_NONE)
		rollingVB = addVertexBuffer(ROLLING_VB_SIZE, DEFAULT);

	ubyte *data = NULL;
	D3D11_MAP flag = D3D11_MAP_WRITE_NO_OVERWRITE;
	if (rollingVBOffset + size > ROLLING_VB_SIZE)
	{
		flag = D3D11_MAP_WRITE_DISCARD;
		rollingVBOffset = 0;
	}

	D3D11_MAPPED_SUBRESOURCE ms;
	context->Map(vertexBuffers[rollingVB].vertexBuffer, 0, flag, 0, &ms);

	return ((uint8 *) ms.pData) + rollingVBOffset;
}

void Direct3D11Renderer::unmapRollingVB(const uint size)
{
	context->Unmap(vertexBuffers[rollingVB].vertexBuffer, 0);

	rollingVBOffset += size;
}

uint Direct3D11Renderer::copyToRollingVB(const void *src, const uint size)
{
	ASSERT(size <= ROLLING_VB_SIZE);

	if (rollingVB == VB_NONE)
		rollingVB = addVertexBuffer(ROLLING_VB_SIZE, DYNAMIC);

	ubyte *data = NULL;
	D3D11_MAP flag = D3D11_MAP_WRITE_NO_OVERWRITE;
	if (rollingVBOffset + size > ROLLING_VB_SIZE)
	{
		flag = D3D11_MAP_WRITE_DISCARD;
		rollingVBOffset = 0;
	}

	uint offset = rollingVBOffset;

	D3D11_MAPPED_SUBRESOURCE ms;
	context->Map(vertexBuffers[rollingVB].vertexBuffer, 0, flag, 0, &ms);
		memcpy(((uint8 *) ms.pData) + offset, src, size);
	context->Unmap(vertexBuffers[rollingVB].vertexBuffer, 0);

	rollingVBOffset += size;

	return offset;
}

ID3D11Resource *Direct3D11Renderer::getResource(const TextureID texture) const
{
	return textures[texture].texture;
}

void Direct3D11Renderer::flush()
{
	context->Flush();
}

void Direct3D11Renderer::finish()
{
	if (eventQuery == NULL)
	{
		D3D11_QUERY_DESC desc;
		desc.Query = D3D11_QUERY_EVENT;
		desc.MiscFlags = 0;
		device->CreateQuery(&desc, &eventQuery);
	}

	context->End(eventQuery);

	context->Flush();

	BOOL result = FALSE;
	do
	{
		context->GetData(eventQuery, &result, sizeof(BOOL), 0);
	} while (!result);
}
