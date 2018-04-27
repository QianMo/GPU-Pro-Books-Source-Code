#include "Precompiled.h"

#if BM_PROV_TEXTURELOADER == BM_PROV_DXT_TEXTURELOADER

#include <Windows.h>

#include <d3d9.h>
#include <d3dx9.h>

#include "Wrap3D/Src/Format.h"
#include "Wrap3D/Src/TextureConfig.h"
#include "Wrap3D/Src/Usages.h"
#include "Wrap3D/Src/Device.h"

#include "TextureLoaderConfig.h"
#include "D3DXTextureLoader.h"

#pragma comment(lib,"d3d9")

#ifdef _DEBUG
#pragma comment(lib,"d3dx9d")
#else
#pragma comment(lib,"d3dx9")
#endif

namespace Mod
{
#define MD_D3DV(expr) if( (expr) != D3D_OK ) { MD_FERROR(L#expr L" failed!" ); }

	//------------------------------------------------------------------------

	namespace
	{
		IDirect3DBaseTexture9*	CreateD3D9Texture( IDirect3DDevice9* dev, const D3DXIMAGE_INFO& imageInfo, const void* dataPtr, UINT64 dataSize );
		TextureConfigPtr		CreateTextureConfig( IDirect3DBaseTexture9* baseTexture, const D3D9FormatMap& fmtMap );
	}	

	//------------------------------------------------------------------------

	/*explicit*/
	D3DXTextureLoader::D3DXTextureLoader( const TextureLoaderConfig& cfg ) :
	Parent( cfg ),
	mD3D9( NULL ),
	mD3D9Device( NULL ),
	mD3D9FormatMap( NULL )
	{
		mD3D9 = Direct3DCreate9( D3D_SDK_VERSION );
		MD_FERROR_ON_FALSE( mD3D9 );

	    D3DDISPLAYMODE Mode;
		MD_D3DV( mD3D9->GetAdapterDisplayMode( 0, &Mode ) );

		D3DPRESENT_PARAMETERS pp;

		memset( &pp, 0, sizeof pp );

		pp.BackBufferWidth	= 1;
		pp.BackBufferHeight	= 1;

		pp.BackBufferFormat		= Mode.Format;
		pp.BackBufferCount		= 1;

		pp.MultiSampleType		= D3DMULTISAMPLE_NONE;
		pp.MultiSampleQuality	= 0;

		pp.SwapEffect		= D3DSWAPEFFECT_COPY;
		pp.hDeviceWindow	= NULL;
		pp.Windowed			= TRUE;
		pp.EnableAutoDepthStencil	= FALSE;
		pp.AutoDepthStencilFormat	= D3DFMT_D16;
		pp.Flags					= 0;
    
		pp.FullScreen_RefreshRateInHz	= 0;
		pp.PresentationInterval			= 0;

		MD_D3DV( mD3D9->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_NULLREF, NULL, D3DCREATE_HARDWARE_VERTEXPROCESSING, &pp, &mD3D9Device ) );

		mD3D9FormatMap.reset( new D3D9FormatMap( cfg.dev->GetFormats() ) );
	}

	//------------------------------------------------------------------------

	D3DXTextureLoader::~D3DXTextureLoader()
	{
		if( mD3D9Device ) 
			mD3D9Device->Release();

		if( mD3D9 ) 
			mD3D9->Release();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	TextureConfigPtr
	D3DXTextureLoader::LoadImpl( const Bytes& data, const String& fileExtension ) /*OVERRIDE*/
	{
		fileExtension;

		MD_FERROR_ON_FALSE( data.GetRawPtr() );

		D3DXIMAGE_INFO  imageInfo;

		const void* dataPtr = data.GetRawPtr();
		UINT64	dataSize	= data.GetSize();

		MD_D3DV( D3DXGetImageInfoFromFileInMemory( dataPtr, (UINT)dataSize, &imageInfo ) );

		IDirect3DBaseTexture9* texture = CreateD3D9Texture( mD3D9Device, imageInfo, dataPtr, dataSize );
		struct ReleaseResources
		{
			~ReleaseResources()
			{
				if( texture )
				{
					texture->Release();
				}
			}

			IDirect3DBaseTexture9* texture;
		} releaseReources = { texture }; &releaseReources;

		TextureConfigPtr texCfg = CreateTextureConfig( texture, *mD3D9FormatMap );

		GetConfig().dev->GetUsages().AssignImmutable( *texCfg );

		return texCfg;		
	}

	//------------------------------------------------------------------------

	TextureLoaderPtr CreateTextureLoader( const TextureLoaderConfig& cfg )
	{
		return TextureLoaderPtr( new D3DXTextureLoader( cfg ) );
	}

	//------------------------------------------------------------------------

	namespace
	{
		IDirect3DBaseTexture9* CreateD3D9Texture( IDirect3DDevice9* dev, const D3DXIMAGE_INFO& imageInfo, const void* dataPtr, UINT64 dataSize )
		{
			switch( imageInfo.ResourceType )
			{
			case D3DRTYPE_TEXTURE:
				{
					IDirect3DTexture9* result;
					MD_D3DV( D3DXCreateTextureFromFileInMemory( dev, dataPtr, (UINT)dataSize, &result ) );

					return result;
				}

			case D3DRTYPE_VOLUMETEXTURE:
				{
					IDirect3DVolumeTexture9* result;
					MD_D3DV( D3DXCreateVolumeTextureFromFileInMemory( dev, dataPtr, (UINT)dataSize, &result ) );

					return result;
				}
				

			case D3DRTYPE_CUBETEXTURE:
				{
					IDirect3DCubeTexture9* result;
					MD_D3DV( D3DXCreateCubeTextureFromFileInMemory( dev, dataPtr, (UINT)dataSize, &result ) );

					return result;
				}
			}

			MD_FERROR( L"CreateTexture: unsupported format!" );

			return NULL;
		}

		//------------------------------------------------------------------------

		MD_INLINE
		UINT32 GetFormatHeightDivider( D3DFORMAT fmt )
		{
			switch( fmt )
			{
			case D3DFMT_DXT1:
			case D3DFMT_DXT2:
			case D3DFMT_DXT3:
			case D3DFMT_DXT4:
			case D3DFMT_DXT5:
				return 4;

			default:
				return 1;
			}
		}

		//------------------------------------------------------------------------

		void DeriveFaceAndMip( IDirect3DCubeTexture9* tex, UINT32 i, D3DCUBEMAP_FACES& oFace, UINT32& oMip )
		{
			UINT32 numMips = tex->GetLevelCount();
			oFace = D3DCUBEMAP_FACES( i / numMips );
			oMip = i % numMips;
		}

		//------------------------------------------------------------------------

		void Lock2D( IDirect3DTexture9* tex, UINT32 i, D3DLOCKED_RECT& oLockedLevel )
		{
			MD_D3DV( tex->LockRect( i, &oLockedLevel, NULL, D3DLOCK_READONLY ) );
		}

		void Lock3D( IDirect3DVolumeTexture9* tex, UINT32 i, D3DLOCKED_BOX& oLockedLevel )
		{
			MD_D3DV( tex->LockBox( i, &oLockedLevel, NULL, D3DLOCK_READONLY ) );
		}

		void LockCube( IDirect3DCubeTexture9* tex, UINT32 i, D3DLOCKED_RECT& oLockedLevel )
		{
			D3DCUBEMAP_FACES faceIdx;
			UINT32 mipIdx;

			DeriveFaceAndMip( tex, i, faceIdx, mipIdx );

			MD_D3DV( tex->LockRect( faceIdx, mipIdx, &oLockedLevel, NULL, D3DLOCK_READONLY ) );
		}

		//------------------------------------------------------------------------

		void Unlock2D( IDirect3DTexture9* tex, UINT32 i )
		{
			MD_D3DV( tex->UnlockRect( i ) );
		}

		void Unlock3D( IDirect3DVolumeTexture9* tex, UINT32 i )
		{
			MD_D3DV( tex->UnlockBox( i ) );
		}

		void UnlockCube( IDirect3DCubeTexture9* tex, UINT32 i )
		{
			D3DCUBEMAP_FACES faceIdx;
			UINT32 mipIdx;

			DeriveFaceAndMip( tex, i, faceIdx, mipIdx );

			MD_D3DV( tex->UnlockRect( faceIdx, mipIdx ) );
		}

		//------------------------------------------------------------------------

		void FillSubres2DCfg( IDirect3DTexture9* tex, const D3DLOCKED_RECT& lockedLevel, UINT32 i, UINT64& ioSize, Texture2DConfig::SubresCfg& oCfg )
		{
			D3DSURFACE_DESC surfaceDesc;
			MD_D3DV( tex->GetLevelDesc( i, &surfaceDesc ) );

			oCfg.pitch	= lockedLevel.Pitch;

			ioSize += surfaceDesc.Height * lockedLevel.Pitch / GetFormatHeightDivider( surfaceDesc.Format );
		}

		void FillSubres3DCfg( IDirect3DVolumeTexture9* tex, const D3DLOCKED_BOX& lockedLevel, UINT32 i, UINT64& ioSize, Texture3DConfig::SubresCfg& oCfg )
		{
			D3DVOLUME_DESC levelDesc;
			MD_D3DV( tex->GetLevelDesc( i, &levelDesc ) );

			oCfg.rowPitch	= lockedLevel.RowPitch;
			oCfg.slicePitch	= lockedLevel.SlicePitch;

			ioSize += levelDesc.Depth * lockedLevel.SlicePitch;
		}

		void FillSubresCubeCfg( IDirect3DCubeTexture9* tex, const D3DLOCKED_RECT& lockedLevel, UINT32 i, UINT64& ioSize, TextureCUBEConfig::SubresCfg& oCfg )
		{
			D3DCUBEMAP_FACES faceIdx;
			UINT32 mipIdx;

			DeriveFaceAndMip( tex, i, faceIdx, mipIdx );

			D3DSURFACE_DESC surfaceDesc;
			MD_D3DV( tex->GetLevelDesc( mipIdx, &surfaceDesc ) );

			oCfg.pitch	= lockedLevel.Pitch;

			ioSize		+= surfaceDesc.Height * lockedLevel.Pitch / GetFormatHeightDivider( surfaceDesc.Format );
		}

		//------------------------------------------------------------------------

		template< typename T>
		struct TextureToSpecifics;

		template <>	struct TextureToSpecifics<IDirect3DTexture9>
		{ 
			typedef D3DLOCKED_RECT LockedLevel;
			static void (* const Lock )( IDirect3DTexture9*, UINT32, D3DLOCKED_RECT& );
			static void (* const Unlock )( IDirect3DTexture9*, UINT32 );
			static void (* const FillSubresCfg)( IDirect3DTexture9* , const D3DLOCKED_RECT& , UINT32, UINT64&, Texture2DConfig::SubresCfg& );
		};

		void (* /*static*/ const TextureToSpecifics<IDirect3DTexture9>::Lock )( IDirect3DTexture9*, UINT32, D3DLOCKED_RECT& )	= Lock2D;
		void (* /*static*/ const TextureToSpecifics<IDirect3DTexture9>::Unlock )( IDirect3DTexture9*, UINT32 )					= Unlock2D;
		void (* /*static*/ const TextureToSpecifics<IDirect3DTexture9>::FillSubresCfg )( 
									IDirect3DTexture9* ,
										const D3DLOCKED_RECT& ,
											UINT32,
												UINT64&,
													Texture2DConfig::SubresCfg& ) = FillSubres2DCfg;

		template <> struct TextureToSpecifics<IDirect3DVolumeTexture9>
		{
			typedef D3DLOCKED_BOX LockedLevel;
			static void (* const Lock )( IDirect3DVolumeTexture9*, UINT32, D3DLOCKED_BOX& );
			static void (* const Unlock )( IDirect3DVolumeTexture9*, UINT32 );
			static void (* const FillSubresCfg )( IDirect3DVolumeTexture9* , const D3DLOCKED_BOX&, UINT32, UINT64&, Texture3DConfig::SubresCfg& );
		};

		void (* /*static*/ const TextureToSpecifics<IDirect3DVolumeTexture9>::Lock )( IDirect3DVolumeTexture9*, UINT32, D3DLOCKED_BOX& )	= Lock3D;
		void (* /*static*/ const TextureToSpecifics<IDirect3DVolumeTexture9>::Unlock )( IDirect3DVolumeTexture9*, UINT32 )					= Unlock3D;
		void (* /*static*/ const TextureToSpecifics<IDirect3DVolumeTexture9>::FillSubresCfg )( 
									IDirect3DVolumeTexture9*,
										const D3DLOCKED_BOX&,
											UINT32,
												UINT64&,
													Texture3DConfig::SubresCfg& ) = FillSubres3DCfg;


		template <> struct TextureToSpecifics<IDirect3DCubeTexture9>
		{
			typedef D3DLOCKED_RECT LockedLevel;
			static void (* const Lock )( IDirect3DCubeTexture9*, UINT32, D3DLOCKED_RECT& );
			static void (* const Unlock )( IDirect3DCubeTexture9*, UINT32 );
			static void (* const FillSubresCfg )( IDirect3DCubeTexture9* , const D3DLOCKED_RECT&, UINT32, UINT64&, TextureCUBEConfig::SubresCfg& );
		};

		void (* /*static*/ const TextureToSpecifics<IDirect3DCubeTexture9>::Lock )( IDirect3DCubeTexture9*, UINT32, D3DLOCKED_RECT& )	= LockCube;
		void (* /*static*/ const TextureToSpecifics<IDirect3DCubeTexture9>::Unlock )( IDirect3DCubeTexture9*, UINT32 )					= UnlockCube;
		void (* /*static*/ const TextureToSpecifics<IDirect3DCubeTexture9>::FillSubresCfg )( 
									IDirect3DCubeTexture9*,
										const D3DLOCKED_RECT&,
											UINT32,
												UINT64&,
													TextureCUBEConfig::SubresCfg& ) = FillSubresCubeCfg;

		template< typename T, typename C >
		void ExtractTextureData( T* texture, C& texCfg, UINT32 numLevels )
		{
			typedef TextureToSpecifics<T> :: LockedLevel LockedLevel;

			typedef TextureToSpecifics<T> Functions;

			typedef Types< LockedLevel > :: Vec LockedLevels;
			LockedLevels lockedLevels( numLevels );

			Types< C::SubresCfg > :: Vec subresCfgs( numLevels );
			Types< UINT64 > :: Vec sizes( numLevels );

			UINT64 totalSize = 0;
			UINT64 prevSize = 0;

			texCfg.subresCfgs.resize( numLevels );

			for( UINT32 i = 0, e = numLevels; i < e; i ++ )
			{
				texCfg.subresCfgs[ i ].dsp = totalSize;

				Functions::Lock( texture, i, lockedLevels[ i ] );
				Functions::FillSubresCfg( texture, lockedLevels[ i ], i, totalSize, texCfg.subresCfgs[ i ] );

				sizes[ i ] = totalSize - prevSize;
				prevSize = totalSize;
			}

			texCfg.data.Resize( totalSize );

			for( UINT32 i = 0, e = numLevels; i < e; i ++ )
			{
				memcpy( &texCfg.data[ texCfg.subresCfgs[ i ].dsp ], lockedLevels[ i ].pBits, size_t( sizes[ i ] ) );
				Functions::Unlock( texture, i );				
			}
		}

		//------------------------------------------------------------------------

		TextureConfigPtr CreateTextureConfig( IDirect3DBaseTexture9* baseTexture, const D3D9FormatMap& fmtMap )
		{
			D3DRESOURCETYPE type = baseTexture->GetType();

			TextureConfigPtr texCfg;

			UINT32 levelCount = baseTexture->GetLevelCount();

			D3DFORMAT d3dFmt ( D3DFMT_UNKNOWN );

			switch( type )
			{
			case D3DRTYPE_TEXTURE:
				{
					IDirect3DTexture9* texture = static_cast< IDirect3DTexture9* >(baseTexture);

					D3DSURFACE_DESC surf;

					MD_D3DV( texture->GetLevelDesc( 0, &surf ) );

					d3dFmt = surf.Format;

					Texture2DConfig* tex2DCfg = new Texture2DConfig;
					tex2DCfg->width		= surf.Width;
					tex2DCfg->height	= surf.Height;

					ExtractTextureData( texture, *tex2DCfg, levelCount );

					texCfg.reset( tex2DCfg );
				}
				break;

			case D3DRTYPE_VOLUMETEXTURE:
				{
					IDirect3DVolumeTexture9* texture = static_cast< IDirect3DVolumeTexture9* >(baseTexture);

					D3DVOLUME_DESC volume;

					MD_D3DV( texture->GetLevelDesc( 0, &volume ) );

					d3dFmt = volume.Format;

					Texture3DConfig* tex3DCfg = new Texture3DConfig;

					tex3DCfg->width		= volume.Width;
					tex3DCfg->height	= volume.Height;
					tex3DCfg->depth		= volume.Depth;

					ExtractTextureData( texture, *tex3DCfg, levelCount );

					texCfg.reset( tex3DCfg );
				}
				break;

			case D3DRTYPE_CUBETEXTURE:
				{
					IDirect3DCubeTexture9* texture = static_cast< IDirect3DCubeTexture9* >(baseTexture);

					D3DSURFACE_DESC surf;

					MD_D3DV( texture->GetLevelDesc( 0, &surf ) );

					d3dFmt = surf.Format;

					TextureCUBEConfig* texCUBECfg = new TextureCUBEConfig;					

					texCUBECfg->width		= surf.Width;
					texCUBECfg->height		= surf.Height;

					ExtractTextureData( texture, *texCUBECfg, 6 * levelCount );

					texCfg.reset( texCUBECfg );
				}
				break;
			default:
				MD_FERROR( L"CreateTextureConfig: Unknown resource format!" );
				break;
			}

			texCfg->numMips = levelCount;

			texCfg->fmt = fmtMap.GetFormat( d3dFmt );

			texCfg->fmt->Conform( texCfg->data );

			return texCfg;
		}
	}

#undef MD_D3DV

}

#endif