#include "Precompiled.h"

#include "WrapSys/Src/WindowConfig.h"
#include "WrapSys/Src/Window.h"

#include "SysWinDrv/Src/WindowExtraData.h"

#include "Wrap3D/Src/Usages.h"
#include "Wrap3D/Src/PrimitiveTopologies.h"
#include "Wrap3D/Src/Formats.h"
#include "Wrap3D/Src/DeviceConfig.h"
#include "Wrap3D/Src/ViewportConfig.h"
#include "Wrap3D/Src/EffectConfig.h"

#include "D3D9Helpers/Src/D3D9FormatMap.h"

#include "D3D9Texture1D.h"
#include "D3D9Texture2D.h"
#include "D3D9Texture3D.h"
#include "D3D9TextureCube.h"

#include "D3D9VertexBuffer.h"
#include "D3D9IndexBuffer.h"

#include "D3D9EffectStateManagerConfig.h"
#include "D3D9EffectStateManager.h"
#include "D3D9EffectPool.h"
#include "D3D9Effect.h"

#include "D3D9InputLayout.h"
#include "D3D9Viewport.h"

#include "D3D9ScissorRect.h"
#include "D3D9ShaderResource.h"

#include "D3D9RenderTarget.h"
#include "D3D9DepthStencil.h"

#include "D3D9Usage.h"

#include "D3D9PrimitiveTopology.h"
#include "D3D9Format.h"
#include "D3D9ExtraFormats.h"

#include "D3D9TextureCoordinatorConfig.h"
#include "D3D9TextureCoordinator.h"

#include "D3D9Instance.h"

#include "D3D9Device.h"

namespace Mod
{
	//------------------------------------------------------------------------

	D3D9DeviceImpl::D3D9DeviceImpl( const DeviceConfig& cfg ) :
	Device ( cfg ),
	mFormatMap( NULL ),
	mDrawPrimitiveType( D3DPT_TRIANGLELIST ),
	mInputLayout( NULL )
	{
		// init usages
#define MD_INIT_USAGE(usg,busg,bpl,tusg,tupl,lflgs)			\
		{													\
			D3D9UsageConfig cfg;							\
			cfg.bufferUsage		= busg;						\
			cfg.bufPool			= bpl;						\
			cfg.textureUsage	= tusg;						\
			cfg.texPool			= tupl;						\
			cfg.lockFlags		= lflgs;					\
			mUsages->usg.reset( new D3D9Usage( cfg ) );		\
		}

		MD_INIT_USAGE( DEFAULT, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT, 0, D3DPOOL_DEFAULT, 0 );	
		MD_INIT_USAGE( DYNAMIC, D3DUSAGE_DYNAMIC|D3DUSAGE_WRITEONLY,D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC, D3DPOOL_DEFAULT, 0 );

		// this one is a special one
		{
			D3D9UsageConfig cfg;
			cfg.bufferUsage		= D3DUSAGE_WRITEONLY;
			cfg.bufPool			= D3DPOOL_DEFAULT;
			cfg.textureUsage	= 0;
			cfg.texPool			= D3DPOOL_DEFAULT;
			cfg.lockFlags		= 0;
			AssignImmutableUsage( new D3D9Usage( cfg ) );
		}

#undef MD_INIT_USAGE


		// init primitive topologies
#define MD_INIT_PRIMITIVE_TOPOLOGY(pty,d3d9pty) mPrimitiveTopologies->pty.reset( new D3D9PrimitiveTopology ( d3d9pty ) )
#define MD_UNSUPPORTED_PR_TOPOLOGY(pty)
		{
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			MD_INIT_PRIMITIVE_TOPOLOGY( UNDEFINED,		D3DPT_POINTLIST		);
			MD_INIT_PRIMITIVE_TOPOLOGY( POINTLIST,		D3DPT_POINTLIST		);
			MD_INIT_PRIMITIVE_TOPOLOGY( LINELIST,		D3DPT_LINELIST		);
			MD_INIT_PRIMITIVE_TOPOLOGY( LINESTRIP,		D3DPT_LINESTRIP		);
			MD_INIT_PRIMITIVE_TOPOLOGY( TRIANGLELIST,	D3DPT_TRIANGLELIST	);
			MD_INIT_PRIMITIVE_TOPOLOGY( TRIANGLESTRIP,	D3DPT_TRIANGLESTRIP	);
			MD_INIT_PRIMITIVE_TOPOLOGY( LINELIST_ADJ,	D3DPT_POINTLIST		);
			MD_UNSUPPORTED_PR_TOPOLOGY( LINESTRIP_ADJ						);
			MD_UNSUPPORTED_PR_TOPOLOGY( TRIANGLELIST_ADJ					);
			MD_UNSUPPORTED_PR_TOPOLOGY( TRIANGLESTRIP_ADJ					);
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == PrimitiveTopologies::NUM_TOPOLOGIES );
		}
#undef MD_UNSUPPORTED_PR_TOPOLOGY
#undef MD_INIT_PRIMITIVE_TOPOLOGY

		IDirect3D9& d3d = *D3D9Instance::Single().Get();

		UINT adapterIdx( 0 );
		{
			UINT adapterCount = d3d.GetAdapterCount();
			for( UINT i = 0, e = adapterCount; i < e; i ++ )
			{
				D3DADAPTER_IDENTIFIER9 identifier;
				MD_D3DV( d3d.GetAdapterIdentifier( i, 0, &identifier ) );

				if( strstr( identifier.Description, "PerfHUD" ) )
				{
					adapterIdx = i;
					break;
				}				
			}
		}

		D3DDEVTYPE devType = adapterIdx ? D3DDEVTYPE_REF : D3DDEVTYPE_HAL;

		// fill caps
		{
			FillD3D9CapsConstants( mCaps, &d3d, adapterIdx, devType );

			ConfigType& cfg = config();
			cfg.MAX_SIMULTANEOUS_RENDERTARGETS	= NUM_SIMULTANEOUS_RENDERTARGETS;
			cfg.MAX_SHADER_RESOURCES			= mCaps.NUM_TEXTURE_SLOTS;
			cfg.MAX_VERTEX_BUFFERS				= mCaps.NUM_VERTEXBUFFER_SLOTS;
			cfg.MAX_VS_INPUT_SLOTS				= mCaps.NUM_VERTEXBUFFER_SLOTS;
			cfg.MAX_SO_BUFFERS					= 0;
			cfg.MAX_VIEWPORTS					= 1;
			cfg.MAX_SCISSORRECTS				= 1;
			cfg.MAX_RENDERTARGET_DIMMENSION		= mCaps.MAX_TEXTURE_DIMMENSION;
			cfg.MAX_TEXTURE_DIMMENSION			= mCaps.MAX_TEXTURE_DIMMENSION;
			cfg.COPIEABLE_DEPTH_STENCIL			= false;

			cfg.PLATFORM_EFFECT_DEFINES.resize( 1 );
			cfg.PLATFORM_EFFECT_DEFINES.back().name = "MD_D3D9";
		}

		// device		
		{
			const WindowConfig& winCfg = cfg.targetWindow->GetConfig();

			const Window::ExtraData& extraData = cfg.targetWindow->GetExtraData();
			MD_CHECK_TYPE(const WindowExtraData,&extraData);
			const WindowExtraData& winExtraData = static_cast<const WindowExtraData&>(extraData);

			D3DPRESENT_PARAMETERS d3dpp = {};

			d3dpp.BackBufferWidth			= winCfg.width;
			d3dpp.BackBufferHeight			= winCfg.height;
			d3dpp.BackBufferFormat			= D3D9_BACKBUFFER_FORMAT;
			d3dpp.BackBufferCount			= 2;

			d3dpp.MultiSampleType			= D3DMULTISAMPLE_TYPE( cfg.MSAA <= 1 ? 0 : cfg.MSAA );
			d3dpp.MultiSampleQuality		= 0;

			d3dpp.SwapEffect				= D3DSWAPEFFECT_DISCARD;
			d3dpp.hDeviceWindow				= winExtraData.hwnd;
			d3dpp.Windowed					= cfg.isFullScreen ? FALSE : TRUE;
			d3dpp.EnableAutoDepthStencil	= FALSE;
			d3dpp.AutoDepthStencilFormat	= D3DFMT_D24S8;
			d3dpp.Flags						= 0;

			/* FullScreen_RefreshRateInHz must be zero for Windowed mode */
			d3dpp.FullScreen_RefreshRateInHz	= 0;
			d3dpp.PresentationInterval			= D3DPRESENT_INTERVAL_IMMEDIATE;

			IDirect3DDevice9* dev( NULL );
			MD_D3DV( d3d.CreateDevice( adapterIdx, devType, NULL, D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_PUREDEVICE | D3DCREATE_MULTITHREADED, &d3dpp, &dev ) );

			mDev.set( dev );
		}

		// formats
		{
			const FormatConfig::CapsBits TEXTURE_BITS = Format::TEXTURE1D | Format::TEXTURE2D | Format::TEXTURE3D | Format::SHADER_SAMPLE;

		// init formats ( after device was created to properly initialize caps )
#define MD_INIT_FORMAT(fmt,d3dfmt,d3dibfmt,declfmt,caps) mFormats->fmt.reset( new D3D9Format ( FormatProxy<D3DFORMAT( d3dfmt ),d3dibfmt>(), &d3d, adapterIdx, devType, declfmt, caps ) );
#define MD_UNSUPPORTED( fmt )
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			MD_INIT_FORMAT( R8_UNORM,				D3DFMT_L8				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS													)
			MD_INIT_FORMAT( R8G8B8A8_UNORM,			D3DFMT_A8R8G8B8			, D3DFMT_UNKNOWN	, D3DDECLTYPE_UBYTE4N	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R8G8B8A8_SNORM,			D3DFMT_A8R8G8B8			, D3DFMT_UNKNOWN	, D3DDECLTYPE_UBYTE4N	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R32G32B32A32_FLOAT,		D3DFMT_A32B32G32R32F	, D3DFMT_UNKNOWN	, D3DDECLTYPE_FLOAT4	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R32G32B32_FLOAT,		MDFMT_R32G32B32_FLOAT	, D3DFMT_UNKNOWN	, D3DDECLTYPE_FLOAT3	, TEXTURE_BITS | Format::VERTEX_BUFFER							)
			MD_INIT_FORMAT( R32G32_FLOAT,			D3DFMT_G32R32F			, D3DFMT_UNKNOWN	, D3DDECLTYPE_FLOAT2	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_UNSUPPORTED( R32G32B32_UINT																																				)
			MD_UNSUPPORTED( R32G32_UINT																																					)
			MD_INIT_FORMAT( R32_FLOAT,				D3DFMT_R32F				, D3DFMT_UNKNOWN	, D3DDECLTYPE_FLOAT1	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R32_UINT,				D3DFMT_INDEX32			, D3DFMT_INDEX32	, D3DDECLTYPE_UNUSED	, Format::INDEX_BUFFER											)
			MD_UNSUPPORTED( R32_SINT																																					)
			MD_INIT_FORMAT( R16G16B16A16_FLOAT,		D3DFMT_A16B16G16R16F	, D3DFMT_UNKNOWN	, D3DDECLTYPE_FLOAT16_4	, TEXTURE_BITS | Format::RENDER_TARGET							)
			MD_INIT_FORMAT( R16G16B16A16_UNORM,		D3DFMT_A16B16G16R16		, D3DFMT_UNKNOWN	, D3DDECLTYPE_USHORT4N	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R16G16B16A16_SNORM,		D3DFMT_A16B16G16R16		, D3DFMT_UNKNOWN	, D3DDECLTYPE_SHORT4N	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R16G16B16A16_UINT,		D3DFMT_A16B16G16R16		, D3DFMT_UNKNOWN	, D3DDECLTYPE_SHORT4	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R16G16_FLOAT,			D3DFMT_G16R16F			, D3DFMT_UNKNOWN	, D3DDECLTYPE_FLOAT16_2	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R16G16_SNORM,			D3DFMT_G16R16			, D3DFMT_UNKNOWN	, D3DDECLTYPE_SHORT2N	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R16G16_UNORM,			D3DFMT_G16R16			, D3DFMT_UNKNOWN	, D3DDECLTYPE_USHORT2N	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R16_TYPELESS,			D3DFMT_G16R16			, D3DFMT_UNKNOWN	, D3DDECLTYPE_SHORT2	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R16_FLOAT,				D3DFMT_R16F				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS | Format::RENDER_TARGET 							)
			MD_INIT_FORMAT( R16_UINT,				D3DFMT_L16				, D3DFMT_INDEX16	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS | Format::INDEX_BUFFER							)
			MD_INIT_FORMAT( R16_UNORM,				D3DFMT_L16				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS													)
			MD_INIT_FORMAT( D16_UNORM,				D3DFMT_D16				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, Format::DEPTH_STENCIL											)
			MD_INIT_FORMAT( R8G8_UNORM,				D3DFMT_A8L8				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS													)
			MD_INIT_FORMAT( R8_UINT,				D3DFMT_L8				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS													)
			MD_INIT_FORMAT( R8G8B8A8_SINT,			D3DFMT_A8R8G8B8			, D3DFMT_UNKNOWN	, D3DDECLTYPE_D3DCOLOR	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R8G8B8A8_UINT,			D3DFMT_A8R8G8B8			, D3DFMT_UNKNOWN	, D3DDECLTYPE_UBYTE4	, TEXTURE_BITS | Format::RENDER_TARGET | Format::VERTEX_BUFFER	)
			MD_INIT_FORMAT( R8_SINT,				D3DFMT_L8				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS													)
			MD_INIT_FORMAT( D24_UNORM_S8_UINT,		D3DFMT_D24S8			, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, Format::DEPTH_STENCIL											)
			MD_INIT_FORMAT( R24G8_TYPELESS,			D3DFMT_D24S8			, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, Format::DEPTH_STENCIL											)
			MD_INIT_FORMAT( X24_TYPELESS_G8_UINT,	D3DFMT_D24S8			, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, Format::DEPTH_STENCIL											)
			MD_INIT_FORMAT( R24_UNORM_X8_TYPELESS,	D3DFMT_D24S8			, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, Format::DEPTH_STENCIL											)
			MD_INIT_FORMAT( BC1_UNORM,				D3DFMT_DXT1				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS													)
			MD_INIT_FORMAT( BC2_UNORM,				D3DFMT_DXT3				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS													)
			MD_INIT_FORMAT( BC3_UNORM,				D3DFMT_DXT5				, D3DFMT_UNKNOWN	, D3DDECLTYPE_UNUSED	, TEXTURE_BITS													)
			MD_UNSUPPORTED( BC5_SNORM																																					)
			MD_UNSUPPORTED( BC5_UNORM																																					)
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == Formats::NUM_FORMATS );
#undef MD_UNSUPPORTED
#undef MD_INIT_FORMAT

			mFormatMap.reset( new D3D9FormatMap( *mFormats ) );
		}

		// texture coordinator
		{
			D3D9TextureCoordinatorConfig ccfg;

			ccfg.device			= mDev;
			ccfg.numTextures	= mCaps.NUM_TEXTURE_SLOTS;

			mTextureCoordinator.reset( new D3D9TextureCoordinator( ccfg ) );
		}


		// state manager
		{
			D3D9EffectStateManagerConfig smcfg;
			smcfg.device			= mDev;
			smcfg.texCoordinator	= mTextureCoordinator;

			mEffectStateManager.set( new D3D9EffectStateManager( smcfg ) );
		}

	}

	//------------------------------------------------------------------------

	D3D9DeviceImpl::~D3D9DeviceImpl()
	{

	}

	//------------------------------------------------------------------------

	const D3D9FormatMap&
	D3D9DeviceImpl::GetD3D9FormatMap() const
	{
		return *mFormatMap;
	}

	//------------------------------------------------------------------------

	D3D9DeviceImpl::DeviceType*
	D3D9DeviceImpl::GetD3D9Device() const
	{
		return &*mDev;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	TexturePtr
	D3D9DeviceImpl::CreateTextureImpl( const Texture1DConfig& cfg ) /*OVERRIDE*/
	{
		return TexturePtr( new D3D9Texture1D( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	TexturePtr
	D3D9DeviceImpl::CreateTextureImpl( const Texture2DConfig& cfg ) /*OVERRIDE*/
	{
		return TexturePtr( new D3D9Texture2D( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	TexturePtr
	D3D9DeviceImpl::CreateTextureImpl( const Texture3DConfig& cfg ) /*OVERRIDE*/
	{
		return TexturePtr( new D3D9Texture3D( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	TexturePtr
	D3D9DeviceImpl::CreateTextureImpl( const TextureCUBEConfig& cfg ) /*OVERRIDE*/
	{
		return TexturePtr( new D3D9TextureCube( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	BufferPtr
	D3D9DeviceImpl::CreateBufferImpl( const VertexBufferConfig& cfg ) /*OVERRIDE*/
	{
		return BufferPtr( new D3D9VertexBuffer( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	BufferPtr
	D3D9DeviceImpl::CreateBufferImpl( const IndexBufferConfig& cfg ) /*OVERRIDE*/
	{
		return BufferPtr( new D3D9IndexBuffer( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	BufferPtr
	D3D9DeviceImpl::CreateBufferImpl( const ShaderBufferConfig& cfg ) /*OVERRIDE*/
	{
		MD_FERROR( L"Buffer not supported!" ); &cfg;
		return BufferPtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	BufferPtr
	D3D9DeviceImpl::CreateBufferImpl( const ShaderIndexBufferConfig& cfg )	/*OVERRIDE*/
	{
		MD_FERROR( L"Buffer not supported!" ); &cfg;
		return BufferPtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	BufferPtr
	D3D9DeviceImpl::CreateBufferImpl( const ShaderVertexBufferConfig& cfg )	/*OVERRIDE*/
	{
		MD_FERROR( L"Buffer not supported!" ); &cfg;
		return BufferPtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	BufferPtr
	D3D9DeviceImpl::CreateBufferImpl( const SOVertexBufferConfig& cfg )	/*OVERRIDE*/
	{
		MD_FERROR( L"Buffer not supported!" ); &cfg;
		return BufferPtr();
	}
	
	//------------------------------------------------------------------------
	/*virtual*/ 
	
	BufferPtr
	D3D9DeviceImpl::CreateBufferImpl( const SOShaderBufferConfig& cfg ) /*OVERRIDE*/
	{
		MD_FERROR( L"Buffer not supported!" ); &cfg;
		return BufferPtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/

	BufferPtr
	D3D9DeviceImpl::CreateBufferImpl( const SOShaderVertexBufferConfig& cfg ) /*OVERRIDE*/
	{
		MD_FERROR( L"Buffer not supported!" ); &cfg;
		return BufferPtr();
	}
	
	//------------------------------------------------------------------------
	/*virtual*/
	
	BufferPtr
	D3D9DeviceImpl::CreateBufferImpl( const ConstantBufferConfig& cfg )	/*OVERRIDE*/
	{
		MD_FERROR( L"Buffer not supported!" ); &cfg;
		return BufferPtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	EffectPoolPtr
	D3D9DeviceImpl::CreateEffectPoolImpl( const EffectPoolConfig& cfg )	/*OVERRIDE*/
	{
		return EffectPoolPtr( new D3D9EffectPool( cfg ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	EffectPtr
	D3D9DeviceImpl::CreateEffectImpl( const EffectConfig& cfg )	/*OVERRIDE*/
	{
		D3D9Effect* d3d9eff;
		EffectPtr result( d3d9eff = new D3D9Effect( cfg, &*mDev ) );

		if( cfg.pool )
		{
			static_cast<D3D9EffectPool&>( *cfg.pool ).UpdateEffect( result );
		}

		d3d9eff;
		d3d9eff->SetStateManager( &*mEffectStateManager );

		return result;
	}


	//------------------------------------------------------------------------
	/*virtual*/

	InputLayoutPtr
	D3D9DeviceImpl::CreateInputLayoutImpl( const InputLayoutConfig& cfg ) /*OVERRIDE*/
	{
		return InputLayoutPtr( new D3D9InputLayout( cfg, mCaps, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	ViewportPtr
	D3D9DeviceImpl::CreateViewportImpl( const ViewportConfig& cfg )	/*OVERRIDE*/
	{
		return ViewportPtr( new D3D9Viewport( cfg ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
		
	ScissorRectPtr
	D3D9DeviceImpl::CreateScissorRectImpl( const ScissorRectConfig& cfg ) /*OVERRIDE*/
	{
		return ScissorRectPtr( new D3D9ScissorRect( cfg ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	ShaderResourcePtr
	D3D9DeviceImpl::CreateShaderResourceImpl( const ShaderResourceConfig& cfg )	/*OVERRIDE*/
	{
		return ShaderResourcePtr( new D3D9ShaderResource( cfg ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	RenderTargetPtr
	D3D9DeviceImpl::CreateRenderTargetImpl( const RenderTargetConfig& cfg )	/*OVERRIDE*/
	{
		return RenderTargetPtr( new D3D9RenderTarget( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	DepthStencilPtr
	D3D9DeviceImpl::CreateDepthStencilImpl( const DepthStencilConfig& cfg ) /*OVERRIDE*/
	{
		return DepthStencilPtr( new D3D9DepthStencil( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	DeviceQueryPtr
	D3D9DeviceImpl::CreateDeviceQueryImpl( const DeviceQueryConfig& cfg ) /*OVERRIDE*/
	{
		&cfg;
		MD_FERROR( L"Unimplemented!" );
		return DeviceQueryPtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	StagedResourcePtr
	D3D9DeviceImpl::CreateStagedBufferImpl( const StagedResourceConfig& cfg ) /*OVERRIDE*/
	{
		&cfg;
		MD_FERROR( L"Unimplemented!" );
		return StagedResourcePtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	StagedResourcePtr
	D3D9DeviceImpl::CreateStagedTexture1DImpl( const StagedResourceConfig& cfg ) /*OVERRIDE*/
	{
		&cfg;
		MD_FERROR( L"Unimplemented!" );
		return StagedResourcePtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/

	StagedResourcePtr
	D3D9DeviceImpl::CreateStagedTexture2DImpl( const StagedResourceConfig& cfg ) /*OVERRIDE*/
	{
		&cfg;
		MD_FERROR( L"Unimplemented!" );
		return StagedResourcePtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	StagedResourcePtr
	D3D9DeviceImpl::CreateStagedTexture3DImpl( const StagedResourceConfig& cfg ) /*OVERRIDE*/
	{
		&cfg;
		MD_FERROR( L"Unimplemented!" );
		return StagedResourcePtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	RenderTargetPtr
	D3D9DeviceImpl::GetDefaultRenderTargetImpl() /*OVERRIDE*/
	{
		IDirect3DSurface9* surf;
		MD_D3DV( mDev->GetBackBuffer( 0, 0, D3DBACKBUFFER_TYPE_MONO, &surf ) );
		// we don't release cause rendertarget does Release on destruction

		return RenderTargetPtr( new D3D9RenderTarget( surf ) );
	}

	//------------------------------------------------------------------------

	/*virtual*/
	TexturePtr
	D3D9DeviceImpl::GetDefaultRenderTargetTextureImpl()	/*OVERRIDE*/
	{
		return TexturePtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	DepthStencilPtr
	D3D9DeviceImpl::GetDefaultDepthStencilImpl() /*OVERRIDE*/
	{
		IDirect3DSurface9* surf;
		MD_D3DV( mDev->GetBackBuffer( 0, 0, D3DBACKBUFFER_TYPE_MONO, &surf ) );

		surf->Release();

		D3DSURFACE_DESC desc;
		surf->GetDesc( &desc );

		const ConfigType& cfg = GetConfig();

		IDirect3DSurface9* dsSurf;
		MD_D3DV( mDev->CreateDepthStencilSurface( desc.Width, desc.Height, D3DFMT_D24S8, D3DMULTISAMPLE_TYPE( cfg.MSAA > 1 ? cfg.MSAA : 0), 0, FALSE, &dsSurf, NULL ) );

		return DepthStencilPtr( new D3D9DepthStencil( dsSurf ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	ViewportPtr
	D3D9DeviceImpl::GetDefaultViewportImpl() /*OVERRIDE*/
	{
		IDirect3DSurface9* surf;
		MD_D3DV( mDev->GetBackBuffer( 0, 0, D3DBACKBUFFER_TYPE_MONO, &surf ) );

		surf->Release();

		D3DSURFACE_DESC desc;
		surf->GetDesc( &desc );

		ViewportConfig vpcfg;

		vpcfg.topLeftX	= 0;
		vpcfg.topLeftY	= 0;
		vpcfg.width		= desc.Width;
		vpcfg.height	= desc.Height;
		vpcfg.minDepth	= 0;
		vpcfg.maxDepth	= 1;

		return ViewportPtr( new D3D9Viewport( vpcfg ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::UpdateRenderTargetsImpl( const RenderTargets& rts, DepthStencilPtr ds ) /*OVERRIDE*/
	{
		MD_ASSERT( rts.size() <= NUM_SIMULTANEOUS_RENDERTARGETS );

		for( size_t i = 0, e = rts.size(); i < e; i ++ )
		{
			if( rts[ i ] )
				static_cast<D3D9RenderTarget&>(*rts[ i ]).BindTo( &*mDev, (UINT32)i );
			else
				D3D9RenderTarget::SetBindToZero( &*mDev, (UINT32)i );
		}

		if( ds )
		{
			static_cast<D3D9DepthStencil&>(*ds).BindTo( &*mDev );
		}
		else
		{
			D3D9DepthStencil::SetBindToZero( &*mDev );
		}
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::ClearRenderTargetImpl( RenderTargetPtr rt, const Math::float4& colr ) /*OVERRIDE*/
	{
		static_cast<D3D9RenderTarget&>( *rt ).Clear( &*mDev, colr );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::ClearDepthStencilImpl( DepthStencilPtr ds, float depth, UINT32 stencil ) /*OVERRIDE*/
	{
		static_cast<D3D9DepthStencil&>(*ds).Clear( &*mDev, depth, stencil );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::UpdateVSShaderResourcesImpl( const ShaderResources& shresses ) /*OVERRIDE*/
	{
		shresses;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void	
	D3D9DeviceImpl::UpdateGSShaderResourcesImpl( const ShaderResources& shresses ) /*OVERRIDE*/
	{
		shresses;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::UpdatePSShaderResourcesImpl( const ShaderResources& shresses ) /*OVERRIDE*/
	{
		MD_ASSERT( shresses.size() <= mCaps.NUM_TEXTURE_SLOTS );

		for( size_t i = 0, e = shresses.size(); i < e; i ++ )
		{
			static_cast<D3D9ShaderResource&>(*shresses[i]).BindTo( &*mDev, UINT32( i ) );
		}

		mTextureCoordinator->Sync();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::UpdateBuffersImpl( const Buffers& buffs ) /*OVERRIDE*/
	{
		MD_ASSERT( buffs.size() <= mCaps.NUM_VERTEXBUFFER_SLOTS );

		for( size_t i = 0, e = buffs.size(); i < e; i ++ )
		{
			if( buffs[ i ] )
			{
				static_cast<D3D9Buffer&>( *buffs[ i ] ).BindAsVB( &*mDev, (UINT32)i );
			}
		}
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::UpdateIndexBufferImpl( const BufferPtr& buff ) /*OVERRIDE*/
	{
		static_cast<D3D9Buffer&>(*buff).BindAsIB( &*mDev );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D9DeviceImpl::UpdateSOBuffersImpl( const Buffers& buffs ) /*OVERRIDE*/
	{
		buffs;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::UpdateViewportsImpl( const Viewports& vpts ) /*OVERRIDE*/
	{
		// no simultaneous viewports in D3D9
		MD_ASSERT( vpts.size() <= 1 );

		if( vpts[0] )
			static_cast<D3D9Viewport&>( *vpts[0] ).BindTo( &*mDev );
		else
			D3D9Viewport::SetBindToZero( &*mDev );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::UpdateScissorRectsImpl( const ScissorRects& scrects ) /*OVERRIDE*/
	{
		// no simultaneous viewports in D3D9
		MD_ASSERT( scrects.size() <= 1 );

		if( !scrects.empty() && scrects[0] )
			static_cast<D3D9ScissorRect&>( *scrects[0] ).BindTo( &*mDev );
		else
			D3D9ScissorRect::SetBindToZero( &*mDev );		
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::SetPrimitiveTopologyImpl( const PrimitiveTopology* topology ) /*OVERRIDE*/
	{
		mDrawPrimitiveType = static_cast< const D3D9PrimitiveTopology *> ( topology )->GetD3D9Value();
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::SetInputLayoutImpl( const InputLayout* inputLayout ) /*OVERRIDE*/
	{
		mInputLayout = static_cast<const D3D9InputLayout*>(inputLayout);
		mInputLayout->Bind( &*mDev );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::StartFrameImpl() /*OVERRIDE*/
	{
		MD_D3DV( mDev->BeginScene() );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::EndFrameImpl() /*OVERRIDE*/
	{
		MD_D3DV( mDev->EndScene() );
	}

	//------------------------------------------------------------------------
	
	void
	D3D9DeviceImpl::PresentImpl() /*OVERRIDE*/
	{
		MD_D3DV( mDev->Present( NULL, NULL, NULL, NULL ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	namespace
	{
		UINT32 vertsToPrims( D3DPRIMITIVETYPE primType, UINT32 numVerts )
		{
			switch( primType )
			{
			case D3DPT_POINTLIST:
				return numVerts;
			case D3DPT_LINELIST:
				MD_FERROR_ON_TRUE( numVerts & 1 );
				return numVerts / 2;
			case D3DPT_LINESTRIP:
				return numVerts - 1;
			case D3DPT_TRIANGLELIST:
				MD_FERROR_ON_TRUE( numVerts % 3 );
				return numVerts / 3;
			case D3DPT_TRIANGLESTRIP:
			case D3DPT_TRIANGLEFAN:
				return numVerts - 2;
			default:
				MD_FERROR( L"Unknown primitive type!" );
			}

			return 0;
		}
	}
	
	/*virtual*/
	void
	D3D9DeviceImpl::DrawImpl( UINT64 vertexCount, UINT64 startVertex ) /*OVERRIDE*/
	{
		mTextureCoordinator->Update();
		MD_D3DV( mDev->DrawPrimitive( mDrawPrimitiveType, (UINT32)startVertex, vertsToPrims( mDrawPrimitiveType, (UINT32)vertexCount ) ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	void
	D3D9DeviceImpl::DrawIndexedImpl( UINT64 indexCount, UINT64 startIndexLocation, INT64 baseVertexLocation ) /*OVERRIDE*/
	{
		mTextureCoordinator->Update();

		MD_D3DV( mDev->DrawIndexedPrimitive( mDrawPrimitiveType, (INT32)baseVertexLocation, 0, 
											(UINT)GetMinBufferVerts() - std::max( (INT32)baseVertexLocation, 0 ),
											(UINT32)startIndexLocation, vertsToPrims( mDrawPrimitiveType, (UINT32)indexCount ) ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	void
	D3D9DeviceImpl::DrawInstancedImpl( UINT64 vertexCountPerInstance, UINT64 instanceCount, UINT64 startVertexLocation, UINT64 startInstanceLocation ) /*OVERRIDE*/
	{
		// NOTE: this isn't supposed to work according to D3D9 docs, but maybe it's a question of hardware
		// and on DX10 cards it will?
		MD_FERROR_ON_TRUE( startInstanceLocation );

		mInputLayout->SetSSFreqs( &*mDev, (UINT32)instanceCount );
		mTextureCoordinator->Update();
		MD_D3DV( mDev->DrawPrimitive( mDrawPrimitiveType, (INT32)startVertexLocation, vertsToPrims( mDrawPrimitiveType, (UINT32)vertexCountPerInstance ) ) );
		mInputLayout->RestoreSSFreqs( &*mDev );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D9DeviceImpl::DrawIndexedInstancedImpl( UINT64 indexCountPerInstance, UINT64 instanceCount, UINT64 startIndexLocation, INT64 baseVertexLocation, UINT64 startInstanceLocation ) /*OVERRIDE*/
	{
		MD_FERROR_ON_TRUE( startInstanceLocation );

		mInputLayout->SetSSFreqs( &*mDev, (UINT32)instanceCount );
		mTextureCoordinator->Update();
		MD_D3DV( mDev->DrawIndexedPrimitive(	mDrawPrimitiveType, (INT32)baseVertexLocation, 0, 
												(UINT)GetMinBufferVerts() - std::max( (INT32)baseVertexLocation, 0 ),
												(UINT32)startIndexLocation, vertsToPrims( mDrawPrimitiveType, (UINT32)indexCountPerInstance ) ) );
		mInputLayout->RestoreSSFreqs( &*mDev );

	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::DrawAutoImpl() /*OVERRIDE*/
	{
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D9DeviceImpl::ResolveTextureMipImpl( const TexturePtr& dest, UINT32 destMip, const TexturePtr& src, UINT32 srcMip, const Format* fmt ) /*OVERRIDE*/
	{
		// do not support format conversion or different mips for now
		MD_FERROR_ON_TRUE( fmt || destMip || srcMip );

		MD_CHECK_TYPE( D3D9Texture2D, &*src );
		MD_CHECK_TYPE( D3D9Texture2D, &*dest );

		D3D9Texture2D* d3d9src = static_cast<D3D9Texture2D*>( &*src );
		D3D9Texture2D* d3d9dest = static_cast<D3D9Texture2D*>( &*dest );

		d3d9src->ResolveTo( &*mDev, *d3d9dest );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::GetSOContentsImpl( Bytes& oBytes, UINT32 slot ) /*OVERRIDE*/
	{
		oBytes, slot;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::GetBufferContentsImpl( Bytes& oBytes, const BufferPtr& buf ) /*OVERRIDE*/
	{
		oBytes, buf;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	void
	D3D9DeviceImpl::GetTexture1DContentsImpl( Bytes& oBytes, const TexturePtr& tex ) /*OVERRIDE*/
	{
		oBytes, tex;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::GetTexture2DContentsImpl( Bytes& oBytes, const TexturePtr& tex ) /*OVERRIDE*/
	{
		oBytes, tex;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::GetTexture3DContentsImpl( Bytes& oBytes, const TexturePtr& tex ) /*OVERRIDE*/
	{
		oBytes, tex;
		MD_FERROR( L"Unsuported!" );		
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::GetTextureCUBEContentsImpl( Bytes& oBytes, const TexturePtr& tex ) /*OVERRIDE*/
	{
		oBytes, tex;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::CopyImpl( TexturePtr dest, TexturePtr src  ) /*OVERRIDE*/
	{
		static_cast<D3D9Texture&>(*src).CopyTo( &*mDev, static_cast<D3D9Texture&>(*dest) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::CopyImpl( BufferPtr dest, BufferPtr src ) /*OVERRIDE*/
	{
		dest, src;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void	
	D3D9DeviceImpl::CopyStagedBufferImpl( const StagedResourcePtr& staged ) /*OVERRIDE*/
	{
		staged;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::CopyStagedTexture1DImpl( const StagedResourcePtr& staged ) /*OVERRIDE*/
	{
		staged;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::CopyStagedTexture2DImpl( const StagedResourcePtr& staged ) /*OVERRIDE*/
	{
		staged;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::CopyStagedTexture3DImpl( const StagedResourcePtr& staged ) /*OVERRIDE*/
	{
		staged;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::PushMessageFilterImpl( const DeviceMessage** messages, UINT32 num ) /*OVERRIDE*/
	{
		messages, num;
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::PopMessageFilterImpl() /*OVERRIDE*/
	{
		MD_FERROR( L"Unsuported!" );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	void
	D3D9DeviceImpl::ResetImpl() /*OVERRIDE*/
	{
	
	}

	//------------------------------------------------------------------------
	/*virtual*/

	INT32
	D3D9DeviceImpl::StartPerfEventImpl( const String& name ) const /*OVERRIDE*/
	{
#if MD_USE_PERF_FUNCTIONS
		return D3DPERF_BeginEvent( 0, name.c_str() );
#else
		name;
		return 0;
#endif
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	INT32
	D3D9DeviceImpl::EndPerfEventImpl() const /*OVERRIDE*/
	{
#if MD_USE_PERF_FUNCTIONS
		return D3DPERF_EndEvent();
#else
		return 0;
#endif
	}

	//------------------------------------------------------------------------
	
	D3D9Device::D3D9Device ( const DeviceConfig& cfg ) :
	Base( cfg )
	{

	}

	//------------------------------------------------------------------------
	/*virtual*/

	D3D9Device::~D3D9Device ()
	{

	}

}