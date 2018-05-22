#pragma once

#include <setjmp.h>
#include <istream>
#include <d3d11.h>

#include "CoreError.h"
#include "CoreColor.h"
#include "ICoreBase.h"
#include "CoreLog.h"
#include <vector>

using namespace std;

extern "C" 
{ 
#include "..\jpeglib\jpeglib.h" 
}
#include "..\libpng\png.h"


class Core;
 
#define JPEG_INPUT_BUF_SIZE  4096

// TGA structures
#pragma pack(push,1)
typedef struct
{
	unsigned char idlength;
	unsigned char cmaptype;
	unsigned char imagetype;

	struct
	{
		unsigned short firstentryindex;
		unsigned short length;
		unsigned char entrysize;
	} cmapspec;

	struct
	{
		unsigned short x0;
		unsigned short y0;
		unsigned short w;
		unsigned short h;
		unsigned char bpp;
		unsigned char desc;
	} imgspec;

} TGAHeader;
#pragma pack(pop)

// For making jpeglib reading from streams
typedef struct 
{
  struct jpeg_source_mgr pub;		// public fields 
  std::istream *In;					// source stream
  void* Buffer;						// jpeglib's requested buffer
} StreamSourceManager;

typedef struct 
{
  struct jpeg_error_mgr Pub;	// public fields 
  jmp_buf JumpBuffer;			// for return to caller
} LogErrorManager;



class CoreTexture2D : public ICoreBase
{
	friend Core;
	protected:
		// Init constructor
		CoreTexture2D();

		// Load a texture from a stream
		CoreResult init(Core* core, std::istream *in[], UINT textureCount, UINT mipLevels,
				  	    UINT cpuAccessFlag, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
					    UINT sampleCount, UINT sampleQuality, bool bSRGB = false);

		// Load a texture from a stream
		CoreResult init(Core* core, const std::vector <std::istream *> &in, UINT mipLevels,
				  	    UINT cpuAccessFlag, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
					    UINT sampleCount, UINT sampleQuality, bool bSRGB = false);

		// Create a texture from memory
		CoreResult init(Core* core, BYTE** data, UINT width, UINT height, UINT textureCount, UINT mipLevels,
					    DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
					    UINT sampleCount, UINT sampleQuality);
		
		// Directly use an ID3D11Texture2D object
		CoreResult init(Core* core, ID3D11Texture2D* texture);
		
		
		ID3D11Texture2D* texture;
		Core* core;
		
		UINT width;
		UINT height;
		UINT mipLevels;
		UINT sampleCount;
		UINT sampleQuality;
		UINT textureCount;
		UINT cpuAccessFlags;
		UINT miscFlags;
		D3D11_USAGE usage;
		UINT bindFlags;
		DXGI_FORMAT format;
		bool sRGB;
		
		// Creates and fills the texture with the supplied data
		CoreResult createAndFillTexture(BYTE** data);
		// Loads a Bitmap from a stream
		CoreResult loadBitmap(std::istream& in, BYTE** data);
		// Loads a TGA from a stream
		CoreResult loadTga(std::istream& in, BYTE** data);
		// Loads a JPG from a stream
		CoreResult loadJpg(std::istream& in, BYTE** data);
		// Loads a PNG from a stream
		CoreResult loadPng(std::istream& in, BYTE** data);

		// CleanUp
		virtual void finalRelease();

		

	public:
		// Create a RenderTargetView
		CoreResult CoreTexture2D::CreateRenderTargetView(D3D11_RENDER_TARGET_VIEW_DESC* rtvDesc, ID3D11RenderTargetView** rtv);
		// Create a DepthStencilView 
		CoreResult CoreTexture2D::CreateDepthStencilView(D3D11_DEPTH_STENCIL_VIEW_DESC* dsvDesc, ID3D11DepthStencilView** dsv);
		// Create a ShaderResourceView
		CoreResult CoreTexture2D::CreateShaderResourceView(D3D11_SHADER_RESOURCE_VIEW_DESC* srvDesc, ID3D11ShaderResourceView** srv);

		inline DXGI_FORMAT GetFormat()						{ return format; }
		inline UINT GetWidth()								{ return width; }
		inline UINT GetHeight()								{ return height; }
		inline UINT GetMipLevels()							{ return mipLevels; }
		inline ID3D11Resource* GetResource()				{ return texture; }
		inline UINT GetSampleCount()						{ return sampleCount; }
		inline UINT GetSampleQuality()						{ return sampleQuality; }
};