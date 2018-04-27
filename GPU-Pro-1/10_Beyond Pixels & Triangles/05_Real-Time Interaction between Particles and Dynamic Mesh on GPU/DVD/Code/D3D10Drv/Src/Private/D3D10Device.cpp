#include "Precompiled.h"

#include "Wrap3D/Src/Formats.h"
#include "Wrap3D/Src/Usages.h"
#include "Wrap3D/Src/DeviceConfig.h"
#include "Wrap3D/Src/TextureConfig.h"
#include "Wrap3D/Src/StagedResourceConfig.h"
#include "Wrap3D/Src/RenderTargetConfig.h"
#include "Wrap3D/Src/DepthStencilConfig.h"
#include "Wrap3D/Src/PrimitiveTopologies.h"
#include "Wrap3D/Src/DeviceQueryTypes.h"
#include "Wrap3D/Src/DeviceMessages.h"
#include "Wrap3D/Src/ViewportConfig.h"

#include "WrapSys/Src/System.h"
#include "WrapSys/Src/Window.h"
#include "WrapSys/Src/WindowConfig.h"
#include "WrapSys/Src/DynamicLibrary.h"
#include "WrapSys/Src/DynamicLibraryConfig.h"

#include "SysWinDrv/Src/WindowExtraData.h"

#include "FormatHelpers.h"
#include "D3D10Format.h"
#include "D3D10Usage.h"
#include "D3D10DeviceQueryType.h"
#include "D3D10DeviceMessage.h"
#include "D3D10FormatMap.h"
#include "D3D10UsageMap.h"

#include "D3D10Texture1D.h"
#include "D3D10Texture2D.h"
#include "D3D10Texture3D.h"

#include "D3D10StagedBuffer.h"
#include "D3D10StagedTexture1D.h"
#include "D3D10StagedTexture2D.h"
#include "D3D10StagedTexture3D.h"

#include "D3D10ShaderResource.h"
#include "D3D10RenderTarget.h"
#include "D3D10DepthStencil.h"

#include "D3D10DeviceQuery.h"

#include "D3D10Buffers.h"
#include "D3D10EffectPool.h"
#include "D3D10Effect.h"
#include "D3D10InputLayout.h"

#include "D3D10Viewport.h"
#include "D3D10ScissorRect.h"

#include "D3D10ResourceMap.h"

#include "D3D10PrimitiveTopology.h"

#include "D3D10Exception.h"

#include "D3D10ImportedFunctions.h"

#include "D3D10Helpers.h"

#include "DXGIFactory.h"

#include "D3D10Device.h"

namespace Mod
{

	D3D10DeviceImpl::D3D10DeviceImpl ( const DeviceConfig& cfg ):
	Device( cfg ),
	mDev( NULL ),
	mFormatMap( NULL ),
	mUsageMap( NULL )
	{

		{
			ConfigType& cfg = config();

			cfg.MAX_SIMULTANEOUS_RENDERTARGETS	= D3D10Device::NUM_SIMULTANEOUS_RENDERTARGETS;
			cfg.MAX_SHADER_RESOURCES			= D3D10Device::NUM_SHADER_RESOURCE_SLOTS;
			cfg.MAX_VERTEX_BUFFERS				= D3D10Device::NUM_VERTEXBUFFER_SLOTS;
			cfg.MAX_VS_INPUT_SLOTS				= D3D10Device::NUM_VS_INPUT_SLOTS;
			cfg.MAX_VIEWPORTS					= D3D10Device::NUM_VIEWPORTS;
			cfg.MAX_SCISSORRECTS				= D3D10Device::NUM_SCISSORRECTS;
			cfg.MAX_SO_BUFFERS					= D3D10Device::NUM_SO_BUFFER_SLOTS;
			cfg.MAX_RENDERTARGET_DIMMENSION		= D3D10Device::MAX_RENDERTARGET_DIMMENSION;
			cfg.MAX_TEXTURE_DIMMENSION			= D3D10Device::MAX_TEXTURE_DIMMENSION;

			cfg.PLATFORM_EFFECT_DEFINES.resize( 1 );

			cfg.PLATFORM_EFFECT_DEFINES.back().name = "MD_D3D10";
		}

		// init usages
#define INIT_USAGE(usg,flags) mUsages->usg.reset( new D3D10Usage( D3D10_USAGE_##usg,flags) )

		INIT_USAGE(DEFAULT, 0);	
		INIT_USAGE(DYNAMIC, D3D10_CPU_ACCESS_WRITE);

		// this one is a special one
		AssignImmutableUsage( new D3D10Usage( D3D10_USAGE_IMMUTABLE, 0 ) );

#undef INIT_USAGE

		// init primitive topologies
#define INIT_PRIMITIVE_TOPOLOGY(pty) mPrimitiveTopologies->pty.reset( new D3D10PrimitiveTopology ( D3D10_PRIMITIVE_TOPOLOGY_##pty ) )
		{
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			INIT_PRIMITIVE_TOPOLOGY( UNDEFINED );
			INIT_PRIMITIVE_TOPOLOGY( POINTLIST );
			INIT_PRIMITIVE_TOPOLOGY( LINELIST );
			INIT_PRIMITIVE_TOPOLOGY( LINESTRIP );
			INIT_PRIMITIVE_TOPOLOGY( TRIANGLELIST );
			INIT_PRIMITIVE_TOPOLOGY( TRIANGLESTRIP );
			INIT_PRIMITIVE_TOPOLOGY( LINELIST_ADJ );
			INIT_PRIMITIVE_TOPOLOGY( LINESTRIP_ADJ );
			INIT_PRIMITIVE_TOPOLOGY( TRIANGLELIST_ADJ );
			INIT_PRIMITIVE_TOPOLOGY( TRIANGLESTRIP_ADJ );
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == PrimitiveTopologies::NUM_TOPOLOGIES );
		}
#undef INIT_PRIMITIVE_TOPOLOGY

		// init query types
#define INIT_DEVICE_QUERY_TYPE(pty) mQueryTypes->pty.reset( new D3D10DeviceQueryTypeImpl<D3D10_QUERY_##pty>() )
		{
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			INIT_DEVICE_QUERY_TYPE( SO_STATISTICS );
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == DeviceQueryTypes::NUM_QUERY_TYPES );
		}
#undef INIT_DEVICE_QUERY_TYPE

		// init device messages
#define INIT_DEVICE_MESSAGE(pty) mDeviceMessages->pty.reset( new D3D10DeviceMessage( D3D10_MESSAGE_ID_##pty ) )
		{
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			INIT_DEVICE_MESSAGE( DEVICE_DRAW_CONSTANT_BUFFER_TOO_SMALL );
			INIT_DEVICE_MESSAGE( CREATEINPUTLAYOUT_EMPTY_LAYOUT );
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == DeviceMessages::NUM_ITEMS );
		}
#undef INIT_DEVICE_MESSAGE

		mUsageMap.reset( new D3D10UsageMap( *mUsages ) );

		// create device and swap chain
		{

			UINT deviceFlags( /*D3D10_CREATE_DEVICE_SINGLETHREADED*/ 0 );
			D3D10_DRIVER_TYPE driverType = D3D10_DRIVER_TYPE_HARDWARE;
			IDXGIAdapter *adapter( NULL );

			if( cfg.useDebugLayer )
			{
				deviceFlags |= D3D10_CREATE_DEVICE_DEBUG;
			}

			if( cfg.allowDebugDevices )
			{
				const DXGIFactory::Ptr& factory = DXGIFactory::Single().Get();

				UINT i = 0;

				IDXGIAdapter * ad;
				while( factory->EnumAdapters(i, &ad) != DXGI_ERROR_NOT_FOUND )
				{
					DXGI_ADAPTER_DESC desc;
					ad->GetDesc(&desc);
					if (wcsstr(desc.Description,L"PerfHUD"))
					{
						adapter		= ad;
						driverType	= D3D10_DRIVER_TYPE_REFERENCE;
						deviceFlags	&= ~D3D10_CREATE_DEVICE_DEBUG;
						break;
					}
					++i;
				}
			}

#ifdef MD_D3D10_STATIC_LINK
			MD_InitD3D10ImportedFunctions();
#else
			DynamicLibraryConfig dlcfg = { 
#ifdef MD_D3D10_1
				L"d3d10_1.dll" 
#else
				L"d3d10.dll" 
#endif
			};
			mD3D10Library = System::Single().CreateDynamicLibrary( dlcfg );

			MD_InitD3D10ImportedFunctions( mD3D10Library );
#endif

#ifdef MD_D3D10_1
			typedef HRESULT ( WINAPI *CreateFuncPtrType) (		IDXGIAdapter *,			D3D10_DRIVER_TYPE,
																HMODULE,				UINT,
																D3D10_FEATURE_LEVEL1,	UINT,
																DXGI_SWAP_CHAIN_DESC*,	IDXGISwapChain **,
																ID3D10Device1 ** );

#else
			typedef HRESULT ( WINAPI *CreateFuncPtrType)(	IDXGIAdapter *,		D3D10_DRIVER_TYPE,
															HMODULE,			UINT,
															UINT,				DXGI_SWAP_CHAIN_DESC *,
															IDXGISwapChain **,	ID3D10Device ** );
#endif

			CreateFuncPtrType CreateFuncPtr;


#ifdef MD_D3D10_1
#define MD_CREATE_DEVICE_FUNC D3D10CreateDeviceAndSwapChain1
#define MD_CREATE_DEVICE_FUNC_STR "D3D10CreateDeviceAndSwapChain1"
#else
#define MD_CREATE_DEVICE_FUNC D3D10CreateDeviceAndSwapChain
#define MD_CREATE_DEVICE_FUNC_STR "D3D10CreateDeviceAndSwapChain"
#endif

#ifdef MD_D3D10_STATIC_LINK
			CreateFuncPtr = MD_CREATE_DEVICE_FUNC;
#else
			CreateFuncPtr	= (CreateFuncPtrType)mD3D10Library->GetProcAddress( MD_CREATE_DEVICE_FUNC_STR );
#endif			

			// check for consistency... (got an error here - update the CreateFuncPtrType from d3d header)
			sizeof (CreateFuncPtr = MD_CREATE_DEVICE_FUNC);


			IDXGISwapChain* swapChain;
			DeviceType* d3d10Device;

			DXGI_SWAP_CHAIN_DESC swapChainDesc = {};

			MD_FERROR_ON_FALSE( cfg.targetWindow );

			const WindowConfig& winCfg = cfg.targetWindow->GetConfig();

			const Window::ExtraData& extraData = cfg.targetWindow->GetExtraData();
			MD_CHECK_TYPE(const WindowExtraData,&extraData);
			const WindowExtraData& winExtraData = static_cast<const WindowExtraData&>(extraData);


			swapChainDesc.BufferDesc.Width						= winCfg.width;
			swapChainDesc.BufferDesc.Height						= winCfg.height;
			swapChainDesc.BufferDesc.RefreshRate.Numerator		= 0;
			swapChainDesc.BufferDesc.RefreshRate.Denominator	= 0;
			swapChainDesc.BufferDesc.Format						= D3D10_DISPLAY_FORMAT;
			swapChainDesc.BufferDesc.ScanlineOrdering			= DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE;
			swapChainDesc.BufferDesc.Scaling					= DXGI_MODE_SCALING_CENTERED;
			swapChainDesc.SampleDesc.Count						= cfg.MSAA ? cfg.MSAA : 1;
			swapChainDesc.SampleDesc.Quality					= 0;
			swapChainDesc.BufferUsage							= DXGI_USAGE_RENDER_TARGET_OUTPUT;
			swapChainDesc.BufferCount							= 1;
			swapChainDesc.OutputWindow							= winExtraData.hwnd;
			swapChainDesc.Windowed								= !cfg.isFullScreen;
			swapChainDesc.SwapEffect							= DXGI_SWAP_EFFECT_DISCARD;
			swapChainDesc.Flags									= cfg.isFullScreen ? DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH : 0;

#ifdef MD_D3D10_1
			if( CreateFuncPtr( adapter, driverType, NULL, deviceFlags, D3D10_FEATURE_LEVEL_10_1, D3D10_1_SDK_VERSION, &swapChainDesc, &swapChain, &d3d10Device ) != S_OK )
			{
				D3D10_THROW_IF( CreateFuncPtr( adapter, driverType, NULL, deviceFlags, D3D10_FEATURE_LEVEL_10_0, D3D10_1_SDK_VERSION, &swapChainDesc, &swapChain, &d3d10Device ) )
				config().COPIEABLE_DEPTH_STENCIL = false;
			}
			else
			{
				config().COPIEABLE_DEPTH_STENCIL = true;
			}
#else
			D3D10_THROW_IF( CreateFuncPtr( adapter, driverType, NULL, deviceFlags,  D3D10_SDK_VERSION, &swapChainDesc, &swapChain, &d3d10Device ) );
#endif
			
#undef MD_CREATE_DEVICE_FUNC
#undef MD_CREATE_DEVICE_FUNC_STR

			mDev.set( d3d10Device );
			mSwapChain.set( swapChain );

			// was debug flag on?

			if( deviceFlags & D3D10_CREATE_DEVICE_DEBUG )
			{
				ID3D10InfoQueue* queue ( NULL );
				D3D10_THROW_IF( mDev->QueryInterface( IID_ID3D10InfoQueue, reinterpret_cast<void**>( &queue ) ) );
				mInfoQueue.set( queue );
			}
		}

		// init formats ( after device was created to properly initialize caps )
#define INIT_FORMAT(fmt) mFormats->fmt.reset( new D3D10Format ( FormatProxy<DXGI_FORMAT_##fmt>(), &*mDev ) );
		{
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			INIT_FORMAT( R8_UNORM )
			INIT_FORMAT( R8G8B8A8_UNORM )
			INIT_FORMAT( R8G8B8A8_SNORM )
			INIT_FORMAT( R32G32B32A32_FLOAT )			
			INIT_FORMAT( R32G32B32_FLOAT )
			INIT_FORMAT( R32G32_FLOAT )
			INIT_FORMAT( R32G32B32_UINT )
			INIT_FORMAT( R32G32_UINT )
			INIT_FORMAT( R32_FLOAT )
			INIT_FORMAT( R32_UINT )
			INIT_FORMAT( R32_SINT )
			INIT_FORMAT( R16G16B16A16_FLOAT )
			INIT_FORMAT( R16G16B16A16_UNORM )
			INIT_FORMAT( R16G16B16A16_SNORM )
			INIT_FORMAT( R16G16B16A16_UINT )
			INIT_FORMAT( R16G16_FLOAT )
			INIT_FORMAT( R16G16_SNORM )
			INIT_FORMAT( R16G16_UNORM )
			INIT_FORMAT( R16_TYPELESS )
			INIT_FORMAT( R16_FLOAT )
			INIT_FORMAT( R16_UINT )
			INIT_FORMAT( R16_UNORM )
			INIT_FORMAT( D16_UNORM )
			INIT_FORMAT( R8G8_UNORM )
			INIT_FORMAT( R8_UINT )
			INIT_FORMAT( R8G8B8A8_SINT )
			INIT_FORMAT( R8G8B8A8_UINT )
			INIT_FORMAT( R8_SINT )
			INIT_FORMAT( D24_UNORM_S8_UINT )
			INIT_FORMAT( R24G8_TYPELESS )
			INIT_FORMAT( X24_TYPELESS_G8_UINT )
			INIT_FORMAT( R24_UNORM_X8_TYPELESS )
			INIT_FORMAT( BC1_UNORM )
			INIT_FORMAT( BC2_UNORM )
			INIT_FORMAT( BC3_UNORM )
			INIT_FORMAT( BC5_SNORM )
			INIT_FORMAT( BC5_UNORM )
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == Formats::NUM_FORMATS );
		}
#undef INIT_FORMAT

		mFormatMap.reset( new D3D10FormatMap( *mFormats ) );

		SetRenderTarget( GetDefaultRenderTarget(), 0 );

	}


	//------------------------------------------------------------------------

	D3D10DeviceImpl::~D3D10DeviceImpl ()
	{
		FreeDeviceDependentResources();
	}

	//------------------------------------------------------------------------

	D3D10DeviceImpl::DeviceType*
	D3D10DeviceImpl::GetD3D10Device() const
	{
		return &*mDev;
	}

	//------------------------------------------------------------------------

	TexturePtr
	D3D10DeviceImpl::CreateTextureImpl( const Texture1DConfig& cfg )
	{		
		return TexturePtr( new D3D10Texture1D( cfg, &*mDev) );
	}

	//------------------------------------------------------------------------

	TexturePtr
	D3D10DeviceImpl::CreateTextureImpl( const Texture2DConfig& cfg )
	{
		Texture2DConfigEx ecfg ( cfg );
		return TexturePtr( new D3D10Texture2D( ecfg, &*mDev) );
	}

	//------------------------------------------------------------------------

	TexturePtr
	D3D10DeviceImpl::CreateTextureImpl( const Texture3DConfig& cfg )
	{
		return TexturePtr( new D3D10Texture3D( cfg, &*mDev) );
	}

	//------------------------------------------------------------------------

	TexturePtr
	D3D10DeviceImpl::CreateTextureImpl( const TextureCUBEConfig& cfg )
	{
		Texture2DConfigEx ecfg( cfg );
		return TexturePtr( new D3D10Texture2D( ecfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	BufferPtr
	D3D10DeviceImpl::CreateBufferImpl( const VertexBufferConfig& cfg )
	{
		return BufferPtr( new D3D10Buffer_Vertex( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	BufferPtr
	D3D10DeviceImpl::CreateBufferImpl( const IndexBufferConfig& cfg  )
	{
		return BufferPtr( new D3D10Buffer_Index(cfg, &*mDev ) );
	}
	//------------------------------------------------------------------------

	BufferPtr		
	D3D10DeviceImpl::CreateBufferImpl( const ShaderBufferConfig& cfg  )
	{
		return BufferPtr( new D3D10Buffer_Shader(cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	BufferPtr
	D3D10DeviceImpl::CreateBufferImpl( const ShaderIndexBufferConfig& cfg )
	{
		return BufferPtr( new D3D10Buffer_ShaderIndex( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	BufferPtr
	D3D10DeviceImpl::CreateBufferImpl( const ShaderVertexBufferConfig& cfg  )
	{
		return BufferPtr( new D3D10Buffer_ShaderVertex(cfg, &*mDev ) );
	}
	//------------------------------------------------------------------------

	BufferPtr
	D3D10DeviceImpl::CreateBufferImpl( const SOVertexBufferConfig& cfg  )
	{
		return BufferPtr( new D3D10Buffer_SOVertex(cfg, &*mDev ) );
	}
	//------------------------------------------------------------------------

	BufferPtr
	D3D10DeviceImpl::CreateBufferImpl( const SOShaderBufferConfig& cfg  )
	{
		return BufferPtr( new D3D10Buffer_SOShader(cfg, &*mDev ) );
	}
	//------------------------------------------------------------------------

	BufferPtr
	D3D10DeviceImpl::CreateBufferImpl( const SOShaderVertexBufferConfig& cfg  )
	{
		return BufferPtr( new D3D10Buffer_SOShaderVertex(cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	BufferPtr
	D3D10DeviceImpl::CreateBufferImpl( const ConstantBufferConfig& cfg )
	{
		return BufferPtr( new D3D10Buffer_Constant( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	EffectPoolPtr
	D3D10DeviceImpl::CreateEffectPoolImpl( const EffectPoolConfig& cfg )
	{
		return EffectPoolPtr( new D3D10EffectPool( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	EffectPtr
	D3D10DeviceImpl::CreateEffectImpl( const EffectConfig& cfg )
	{
		return EffectPtr( new D3D10Effect(cfg, &*mDev) );
	}

	//------------------------------------------------------------------------

	InputLayoutPtr
	D3D10DeviceImpl::CreateInputLayoutImpl( const InputLayoutConfig& cfg )
	{
		return InputLayoutPtr( new D3D10InputLayout( cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::ClearRenderTargetImpl( RenderTargetPtr rt, const Math::float4& colr )
	{
		static_cast< D3D10RenderTarget* >(rt.get())->Clear( &*mDev, colr );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::ClearDepthStencilImpl( DepthStencilPtr ds, float depth, UINT32 stencil )
	{
		static_cast< D3D10DepthStencil* > ( ds.get() )->Clear( &*mDev, depth, stencil );
	}

	//------------------------------------------------------------------------

	namespace
	{
		template < void ( MD_D3D_CALLING_CONV ID3D10Device::* func) ( UINT , UINT , ID3D10ShaderResourceView* const * ) >
		void UpdateShaderResources( ID3D10Device * dev, const Device::ShaderResources& shresses )
		{
			ID3D10ShaderResourceView* resses[ D3D10Device::NUM_SHADER_RESOURCE_SLOTS ] = {};

			int num;
			FillSlots( resses, shresses, num );

			if( num <= 0)
				return;

			(dev->*func)( 0, num, resses );
		}
	}

	void
	D3D10DeviceImpl::UpdateVSShaderResourcesImpl( const ShaderResources& shresses )
	{
		UpdateShaderResources<&ID3D10Device::VSSetShaderResources>( &*mDev, shresses );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::UpdateGSShaderResourcesImpl( const ShaderResources& shresses )
	{
		UpdateShaderResources<&ID3D10Device::GSSetShaderResources>( &*mDev, shresses );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::UpdatePSShaderResourcesImpl( const ShaderResources& shresses )
	{
		UpdateShaderResources<&ID3D10Device::PSSetShaderResources>( &*mDev, shresses );
	}

	//------------------------------------------------------------------------

	ViewportPtr
	D3D10DeviceImpl::CreateViewportImpl( const ViewportConfig& cfg )
	{
		return ViewportPtr ( new D3D10Viewport( cfg ) );
	}

	//------------------------------------------------------------------------

	ScissorRectPtr
	D3D10DeviceImpl::CreateScissorRectImpl( const ScissorRectConfig& cfg )
	{
		return ScissorRectPtr( new D3D10ScissorRect( cfg ) );
	}

	//------------------------------------------------------------------------

	ShaderResourcePtr
	D3D10DeviceImpl::CreateShaderResourceImpl( const ShaderResourceConfig& cfg )
	{
		return ShaderResourcePtr( new D3D10ShaderResource(	cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	RenderTargetPtr
	D3D10DeviceImpl::CreateRenderTargetImpl( const RenderTargetConfig& cfg )
	{
		return RenderTargetPtr( new D3D10RenderTarget(	cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	DepthStencilPtr
	D3D10DeviceImpl::CreateDepthStencilImpl( const DepthStencilConfig& cfg )
	{
		return DepthStencilPtr( new D3D10DepthStencil(	cfg, &*mDev ));
	}

	//------------------------------------------------------------------------

	DeviceQueryPtr
	D3D10DeviceImpl::CreateDeviceQueryImpl( const DeviceQueryConfig& cfg )
	{
		return DeviceQueryPtr( new D3D10DeviceQuery(cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	StagedResourcePtr
	D3D10DeviceImpl::CreateStagedBufferImpl( const StagedResourceConfig& cfg ) /*OVERRIDE*/
	{
		return StagedResourcePtr( new D3D10StagedBuffer( cfg, &*mDev, GetD3D10ResourceSize( *mFormatMap, static_cast<ID3D10Buffer*>( &*static_cast<D3D10Buffer&>(*cfg.buf).GetResource()))));
	}

	//------------------------------------------------------------------------
	/*virtual*/

	StagedResourcePtr
	D3D10DeviceImpl::CreateStagedTexture1DImpl( const StagedResourceConfig& cfg ) /*OVERRIDE*/
	{
		return StagedResourcePtr( new D3D10StagedTexture1D( cfg, &*mDev, GetD3D10ResourceSize( *mFormatMap, static_cast<ID3D10Texture1D*>( &*static_cast<D3D10Texture1D&>(*cfg.tex).GetResource()))));
	}

	//------------------------------------------------------------------------
	/*virtual*/

	StagedResourcePtr
	D3D10DeviceImpl::CreateStagedTexture2DImpl( const StagedResourceConfig& cfg ) /*OVERRIDE*/
	{
		return StagedResourcePtr( new D3D10StagedTexture2D( cfg, &*mDev, GetD3D10ResourceSize( *mFormatMap, static_cast<ID3D10Texture2D*>( &*static_cast<D3D10Texture2D&>(*cfg.tex).GetResource()))));
	}

	//------------------------------------------------------------------------
	/*virtual*/

	StagedResourcePtr
	D3D10DeviceImpl::CreateStagedTexture3DImpl( const StagedResourceConfig& cfg ) /*OVERRIDE*/
	{
		return StagedResourcePtr( new D3D10StagedTexture3D( cfg, &*mDev, GetD3D10ResourceSize( *mFormatMap, static_cast<ID3D10Texture3D*>( &*static_cast<D3D10Texture3D&>(*cfg.tex).GetResource()))));
	}

	//------------------------------------------------------------------------

	RenderTargetPtr
	D3D10DeviceImpl::GetDefaultRenderTargetImpl()
	{		

		RenderTargetConfig cfg;
		cfg.tex = GetDefaultRenderTargetTexture();

		return RenderTargetPtr( new D3D10RenderTarget(cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	/*virtual*/
	TexturePtr
	D3D10DeviceImpl::GetDefaultRenderTargetTextureImpl() /*OVERRIDE*/
	{
		// extract main render target
		ID3D10Texture2D *backBufferTexture;
		D3D10_THROW_IF( ( mSwapChain->GetBuffer( 0, __uuidof( ID3D10Texture2D ), (LPVOID*)&backBufferTexture ) ) );

		backBufferTexture->AddRef();

		return CreateTextureFromResource( ComPtr<ID3D10Resource>(backBufferTexture), *mFormatMap, *mUsageMap );
	}

	//------------------------------------------------------------------------

	DepthStencilPtr
	D3D10DeviceImpl::GetDefaultDepthStencilImpl()
	{
		// extract main render target
		ID3D10Texture2D *backBufferTexture;
		D3D10_THROW_IF( ( mSwapChain->GetBuffer( 0, __uuidof( ID3D10Texture2D ), (LPVOID*)&backBufferTexture ) ) );

		D3D10_TEXTURE2D_DESC desc;
		backBufferTexture->GetDesc( &desc );

		TexturePtr tex;
		{
			Texture2DConfig cfg;
			cfg.shaderResource	= GetConfig().defaultDepthStencilAllowsSR;
			cfg.depthStencil	= true;
			cfg.usage			= mUsages->DEFAULT;
			cfg.fmt				= cfg.shaderResource ? mFormats->R24G8_TYPELESS : mFormats->D24_UNORM_S8_UINT;
			cfg.width			= desc.Width;
			cfg.height			= desc.Height;
			cfg.numMips			= 1;
			cfg.sampleCount		= GetConfig().MSAA;

			tex = CreateTexture( cfg );
		}


		DepthStencilConfig cfg;
		cfg.tex = tex;
		cfg.fmt = mFormats->D24_UNORM_S8_UINT;
		return DepthStencilPtr( new D3D10DepthStencil(cfg, &*mDev ) );
	}

	//------------------------------------------------------------------------

	ViewportPtr
	D3D10DeviceImpl::GetDefaultViewportImpl()
	{
		// extract main render target
		ID3D10Texture2D *backBufferTexture;
		D3D10_THROW_IF( ( mSwapChain->GetBuffer( 0, __uuidof( ID3D10Texture2D ), (LPVOID*)&backBufferTexture ) ) );

		D3D10_TEXTURE2D_DESC desc;
		backBufferTexture->GetDesc( &desc );

		ViewportConfig cfg;

		cfg.topLeftX	= 0;
		cfg.topLeftY	= 0;
		cfg.width		= desc.Width;
		cfg.height		= desc.Height;
		cfg.minDepth	= 0;
		cfg.maxDepth	= 1;

		return ViewportPtr( new D3D10Viewport(cfg) );
	}

	//------------------------------------------------------------------------

	namespace
	{

		template <typename T>
		struct TypeToD3D10Type;

#define MD_CLEANUP_FUNC										\
		template <typename T>								\
		static void Cleanup( T* slots, int& last )			\
		{													\
			while( last > 0 && !slots[last-1] ) last--;		\
		}

#define MD_NOCLEANUP_FUNC									\
		template <typename T>								\
		static void Cleanup( T* slots, int& last )			\
		{													\
			slots, last;									\
		}

		template <>	struct TypeToD3D10Type<RenderTarget>	{ typedef D3D10RenderTarget		Result;	MD_NOCLEANUP_FUNC };
		template <>	struct TypeToD3D10Type<ShaderResource>	{ typedef D3D10ShaderResource	Result;	MD_NOCLEANUP_FUNC };
		template <>	struct TypeToD3D10Type<Buffer>			{ typedef D3D10Buffer			Result;	MD_NOCLEANUP_FUNC };
		template <>	struct TypeToD3D10Type<Viewport>		{ typedef D3D10Viewport			Result;	MD_CLEANUP_FUNC };
		template <>	struct TypeToD3D10Type<ScissorRect>		{ typedef D3D10ScissorRect		Result;	MD_CLEANUP_FUNC };

#undef MD_NOCLEANUP_FUNC
#undef MD_CLEANUP_FUNC

		template < typename T, typename U, int N >
		void FillSlots( T (& slots)[N], const U& source, int& last )
		{
			typedef TypeToD3D10Type< typename U::value_type::value_type > Props;
			typedef Props::Result D3D10Type;

			last = 0;
			for( U::const_iterator i = source.begin(), e = source.end();
				i != e; ++i )
			{
				MD_ASSERT( last <= N );
				if( U::value_type::value_type * val = (*i).get() )
				{
					static_cast< D3D10Type* > (val)->BindTo( slots[last++] );				
				}
				else
					D3D10Type::SetBindToZero( slots[last++] );
				
			}

			Props::Cleanup( slots, last );
		}
	}

	void
	D3D10DeviceImpl::UpdateRenderTargetsImpl( const RenderTargets& rts, DepthStencilPtr ds )
	{

		ID3D10RenderTargetView* views[ NUM_SIMULTANEOUS_RENDERTARGETS ];
	
		int num;
		FillSlots( views, rts, num );

		ID3D10DepthStencilView* dsv(NULL);

		if( ds )
		{
			static_cast<D3D10DepthStencil*>(ds.get())->BindTo( dsv );
		}

		if( num <= 0 && !dsv )
			return;

		mDev->OMSetRenderTargets(	static_cast<UINT>( num ),
									views,
									dsv );

	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::UpdateBuffersImpl( const Buffers& buffs )
	{
		ID3D10Buffer* buffers[ NUM_VERTEXBUFFER_SLOTS ] = {};
		UINT strides[ NUM_VERTEXBUFFER_SLOTS ];
		UINT offsets[ NUM_VERTEXBUFFER_SLOTS ];

		D3D10Buffer::IABindSlot slots[ NUM_VERTEXBUFFER_SLOTS ];
		for( int i = 0; i < NUM_VERTEXBUFFER_SLOTS; i ++ )
		{
			D3D10Buffer::IABindSlot& s = slots[i];
			s.buffer = buffers + i;
			s.offset = offsets + i;
			s.stride = strides + i;
		}

		int num;
		FillSlots( slots, buffs, num );

		if( num <= 0)
			return;

		mDev->IASetVertexBuffers( 0, num, buffers, strides, offsets );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::UpdateIndexBufferImpl( const BufferPtr& buff )
	{
		if( buff )
		{
			MD_CHECK_TYPE( D3D10Buffer_Index, buff.get() );
			static_cast<D3D10Buffer_Index*>( buff.get() )->BindTo( &*mDev );
		}
		else
			D3D10Buffer_Index::SetBindToZero( &*mDev );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::UpdateSOBuffersImpl( const Buffers& buffs )
	{
		ID3D10Buffer* buffers[ NUM_SO_BUFFER_SLOTS ] = {};
		UINT offsets[ NUM_SO_BUFFER_SLOTS ];

		D3D10Buffer::SOBindSlot slots[ NUM_SO_BUFFER_SLOTS ];
		for( int i = 0; i < NUM_SO_BUFFER_SLOTS; i ++ )
		{
			D3D10Buffer::SOBindSlot& s = slots[i];
			s.buffer = buffers + i;
			s.offset = offsets + i;
		}

		int num;
		FillSlots( slots, buffs, num );

		if( num <= 0)
			return;

		mDev->SOSetTargets( num, buffers, offsets );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::UpdateViewportsImpl( const Viewports& vpts )
	{
		D3D10_VIEWPORT viewports[ NUM_VIEWPORTS ];

		int num;
		FillSlots( viewports, vpts, num );

		if( num <= 0 )
			return;

		mDev->RSSetViewports( num, viewports );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::UpdateScissorRectsImpl( const ScissorRects& scrects )
	{
		D3D10_RECT rects[ NUM_SCISSORRECTS ];

		int num;
		FillSlots( rects, scrects, num );

		if( num <= 0 )
			return;

		mDev->RSSetScissorRects( num, rects );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::SetPrimitiveTopologyImpl( const PrimitiveTopology* topology )
	{
		mDev->IASetPrimitiveTopology( static_cast< const D3D10PrimitiveTopology* >(topology)->GetD3D10Value() );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::SetInputLayoutImpl( const InputLayout* inputLayout )
	{
		static_cast< const D3D10InputLayout* >( inputLayout )->Bind( &*mDev );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::StartFrameImpl()
	{

	}
	//------------------------------------------------------------------------

	void D3D10DeviceImpl::EndFrameImpl()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::PresentImpl()
	{
		mSwapChain->Present( 0, 0 );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::DrawImpl( UINT64 vertexCount, UINT64 startVertex )
	{
		mDev->Draw(		static_cast<UINT32>(vertexCount), 
						static_cast<UINT32>(startVertex) );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::DrawIndexedImpl( UINT64 indexCount, UINT64 startIndexLocation, INT64 baseVertexLocation )
	{
		mDev->DrawIndexed(	static_cast<UINT>(indexCount), 
							static_cast<UINT>(startIndexLocation), 
							static_cast<INT>(baseVertexLocation) );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::DrawInstancedImpl( UINT64 vertexCountPerInstance, UINT64 instanceCount, UINT64 startVertexLocation, UINT64 startInstanceLocation )
	{
		mDev->DrawInstanced(	static_cast<UINT>(vertexCountPerInstance),
								static_cast<UINT>(instanceCount ),
								static_cast<UINT>(startVertexLocation), 
								static_cast<UINT>(startInstanceLocation) );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::DrawIndexedInstancedImpl( UINT64 indexCountPerInstance, UINT64 instanceCount, UINT64 startIndexLocation, INT64 baseVertexLocation, UINT64 startInstanceLocation )
	{
		mDev->DrawIndexedInstanced(	static_cast<UINT>(indexCountPerInstance),
									static_cast<UINT>(instanceCount ),
									static_cast<UINT>(startIndexLocation), 
									static_cast<INT>(baseVertexLocation),
									static_cast<UINT>(startInstanceLocation) );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::DrawAutoImpl()
	{
		mDev->DrawAuto();
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::ResolveTextureMipImpl( const TexturePtr& dest, UINT32 destMip, const TexturePtr& src, UINT32 srcMip, const Format* fmt )
	{
		MD_CHECK_TYPE( D3D10Texture2D, &*src );
		MD_CHECK_TYPE( D3D10Texture2D, &*dest );

		D3D10Texture2D* d3d10src = static_cast<D3D10Texture2D*>( &*src );
		D3D10Texture2D* d3d10dest = static_cast<D3D10Texture2D*>( &*dest );

		const D3D10Texture2D::ResourcePtr& srcRes	= d3d10src->GetResource();
		const D3D10Texture2D::ResourcePtr& destRes	= d3d10dest->GetResource();

		mDev->ResolveSubresource( &*destRes, destMip, &*srcRes, srcMip, static_cast<const D3D10Format*>(fmt)->GetValue() );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::GetSOContentsImpl( Bytes& oBytes, UINT32 slot )
	{
		ID3D10Buffer* buffs[ NUM_SO_BUFFER_SLOTS ];
		UINT offsets[ NUM_SO_BUFFER_SLOTS ];
		mDev->SOGetTargets( slot+1, buffs, offsets );

		if( ID3D10Buffer* src = buffs[slot] )
		{
			D3D10_BUFFER_DESC bdesc;
			src->GetDesc( &bdesc );

			bdesc.BindFlags			= 0;
			bdesc.CPUAccessFlags	= D3D10_CPU_ACCESS_READ;			
			bdesc.Usage				= D3D10_USAGE_STAGING;

			ID3D10Buffer *dest;
			D3D10_THROW_IF( mDev->CreateBuffer( &bdesc, NULL, &dest ) );

			mDev->CopyResource( dest, src );

			void * data;
			dest->Map( D3D10_MAP_READ, 0, &data );

			oBytes.Resize( bdesc.ByteWidth );
			memcpy( oBytes.GetRawPtr(), data, bdesc.ByteWidth );

			dest->Release();
		}
	}

	//------------------------------------------------------------------------

	namespace
	{

		template <typename T>
		void GetResourceContentImpl( const D3D10FormatMap& fmap, ID3D10Device* dev, Bytes& oBytes, const T& res )
		{
			ComPtr< T::ResType > stagingRes;
			D3D10::CreateStagedResource( stagingRes, res, dev );

			const T::ResourcePtr&	resPtr = res.GetResource();
			dev->CopyResource( &*stagingRes, &*resPtr );

			// TODO : do all mips for textures

			D3D10ResourceMap< UINT8, T::ResType* > resMap( fmap, &*stagingRes, D3D10_MAP_READ );

			oBytes.Resize( resMap.GetMappedSize() );
			memcpy( oBytes.GetRawPtr(), resMap.GetRawMappedPtr(), size_t( oBytes.GetSize() ) );			
		}
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::GetBufferContentsImpl( Bytes& oBytes, const BufferPtr& buf )
	{
		GetResourceContentImpl( *mFormatMap, &*mDev, oBytes, static_cast<D3D10Buffer&>( *buf ) );
	}

	void
	D3D10DeviceImpl::GetTexture1DContentsImpl( Bytes& oBytes, const TexturePtr& tex )
	{
		GetResourceContentImpl( *mFormatMap, &*mDev, oBytes, static_cast<D3D10Texture1D&>( *tex ) );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::GetTexture2DContentsImpl( Bytes& oBytes, const TexturePtr& tex )
	{
		GetResourceContentImpl( *mFormatMap, &*mDev, oBytes, static_cast<D3D10Texture2D&>( *tex ) );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::GetTexture3DContentsImpl( Bytes& oBytes, const TexturePtr& tex )
	{
		GetResourceContentImpl( *mFormatMap, &*mDev, oBytes, static_cast<D3D10Texture3D&>( *tex ) );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::GetTextureCUBEContentsImpl( Bytes& oBytes, const TexturePtr& tex )
	{
		MD_THROW( L"Unimplemented.." );
		oBytes, tex;
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::CopyImpl( TexturePtr dest, TexturePtr src )
	{
		const D3D10Texture::ResourcePtr& rsrc	= static_cast<D3D10Texture*> ( src.get() )->GetResource();
		const D3D10Texture::ResourcePtr& rdest	= static_cast<D3D10Texture*> ( dest.get() )->GetResource();

		mDev->CopyResource( &*rdest, &*rsrc );

	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::CopyImpl( BufferPtr dest, BufferPtr src )
	{
		const D3D10Buffer::ResourcePtr& rsrc	= static_cast<D3D10Buffer*> ( src.get() )->GetResource();
		const D3D10Buffer::ResourcePtr& rdest	= static_cast<D3D10Buffer*> ( dest.get() )->GetResource();

		mDev->CopyResource( &*rdest, &*rsrc );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D10DeviceImpl::CopyStagedBufferImpl( const StagedResourcePtr& staged ) /*OVERRIDE*/
	{
		static_cast<D3D10StagedBuffer&>( *staged ).Sync( &*mDev );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D10DeviceImpl::CopyStagedTexture1DImpl( const StagedResourcePtr& staged ) /*OVERRIDE*/
	{
		static_cast<D3D10StagedTexture1D&>( *staged ).Sync( &*mDev );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D10DeviceImpl::CopyStagedTexture2DImpl( const StagedResourcePtr& staged ) /*OVERRIDE*/
	{
		static_cast<D3D10StagedTexture2D&>( *staged ).Sync( &*mDev );
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D10DeviceImpl::CopyStagedTexture3DImpl( const StagedResourcePtr& staged ) /*OVERRIDE*/
	{
		static_cast<D3D10StagedTexture3D&>( *staged ).Sync( &*mDev );
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::PushMessageFilterImpl( const DeviceMessage** messages, UINT32 num )
	{
		if( !mInfoQueue.null() )
		{
			D3D10_MESSAGE_ID *denyList = (D3D10_MESSAGE_ID*)_alloca( sizeof( D3D10_MESSAGE_ID ) * num );

			MD_FERROR_ON_FALSE( denyList );

			for( UINT32 i = 0; i < num; i ++ )
			{
				denyList[ i ] = static_cast< const D3D10DeviceMessage* > ( messages[ i ] )->GetValue();
			}

			D3D10_INFO_QUEUE_FILTER flt = {};
			flt.DenyList.NumIDs = num;
			flt.DenyList.pIDList = denyList;	

			mInfoQueue->PushStorageFilter( &flt );
		}
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::PopMessageFilterImpl()
	{
		if( !mInfoQueue.null() )
		{
			mInfoQueue->PopStorageFilter();
		}
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceImpl::ResetImpl()
	{
		mDev->Flush();
		mDev->ClearState();
		mDev->Flush();
	}

	//------------------------------------------------------------------------
	/*virtual*/

	INT32
	D3D10DeviceImpl::StartPerfEventImpl( const String& name ) const /*OVERRIDE*/
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
	D3D10DeviceImpl::EndPerfEventImpl() const /*OVERRIDE*/
	{
#if MD_USE_PERF_FUNCTIONS
		return D3DPERF_EndEvent();
#else
		return 0;
#endif
	}


	//------------------------------------------------------------------------

	const D3D10FormatMap&
	D3D10DeviceImpl::GetD3D10FormatMap() const
	{
		return *mFormatMap;
	}

	//------------------------------------------------------------------------

	const D3D10UsageMap&
	D3D10DeviceImpl::GetD3D10UsageMap() const
	{
		return *mUsageMap;
	}

	//------------------------------------------------------------------------

	D3D10Device::D3D10Device ( const DeviceConfig& cfg ) : 
	D3D10DeviceImpl( cfg )
	{
	}

	//------------------------------------------------------------------------

	D3D10Device::~D3D10Device ()
	{
	}

}

