#ifndef D3D10DEVICE_DEVICE_H_INCLUDED
#define D3D10DEVICE_DEVICE_H_INCLUDED

#include "Wrap3D\Src\Device.h"
#include "WrapSys\Src\Forw.h"

#include "Forw.h"

namespace Mod
{

	// impl is to hide real private stuff from friends

	class D3D10DeviceImpl : public Device
	{
		// types & consts
	public:
		static const UINT32 NUM_SHADER_RESOURCE_SLOTS		= D3D10_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
		static const UINT32 NUM_VERTEXBUFFER_SLOTS			= D3D10_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT;
		static const UINT32 NUM_VS_INPUT_SLOTS				= D3D10_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT;
		static const UINT32 NUM_SO_BUFFER_SLOTS				= D3D10_SO_BUFFER_SLOT_COUNT;
		static const UINT32 NUM_SIMULTANEOUS_RENDERTARGETS	= D3D10_SIMULTANEOUS_RENDER_TARGET_COUNT;
		static const UINT32 NUM_VIEWPORTS					= D3D10_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
		static const UINT32 NUM_SCISSORRECTS				= D3D10_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
		static const UINT32 MAX_RENDERTARGET_DIMMENSION		= D3D10_REQ_TEXTURE2D_U_OR_V_DIMENSION;
		static const UINT32 MAX_TEXTURE_DIMMENSION			= D3D10_REQ_TEXTURE2D_U_OR_V_DIMENSION;

#ifdef MD_D3D10_1
		typedef ID3D10Device1 DeviceType;
#else
		typedef ID3D10Device DeviceType;
#endif


		// construction/ destruction
	public:
		D3D10DeviceImpl ( const DeviceConfig& cfg );
		virtual ~D3D10DeviceImpl ();

		// manipulation/ access
	public:
		const D3D10FormatMap& GetD3D10FormatMap() const;
		const D3D10UsageMap& GetD3D10UsageMap() const;

		// stuff for befrienders
	protected:
		DeviceType* GetD3D10Device() const;

		// polymorphism
	private:
		virtual TexturePtr	CreateTextureImpl( const Texture1DConfig& cfg ) OVERRIDE;
		virtual TexturePtr	CreateTextureImpl( const Texture2DConfig& cfg ) OVERRIDE;
		virtual TexturePtr	CreateTextureImpl( const Texture3DConfig& cfg ) OVERRIDE;
		virtual TexturePtr	CreateTextureImpl( const TextureCUBEConfig& cfg ) OVERRIDE;

		virtual BufferPtr CreateBufferImpl( const VertexBufferConfig& cfg )			OVERRIDE;
		virtual BufferPtr CreateBufferImpl( const IndexBufferConfig& cfg )			OVERRIDE;
		virtual BufferPtr CreateBufferImpl( const ShaderBufferConfig& cfg )			OVERRIDE;
		virtual BufferPtr CreateBufferImpl( const ShaderIndexBufferConfig& cfg )	OVERRIDE;
		virtual BufferPtr CreateBufferImpl( const ShaderVertexBufferConfig& cfg )	OVERRIDE;
		virtual BufferPtr CreateBufferImpl( const SOVertexBufferConfig& cfg )		OVERRIDE;
		virtual BufferPtr CreateBufferImpl( const SOShaderBufferConfig& cfg )		OVERRIDE;
		virtual BufferPtr CreateBufferImpl( const SOShaderVertexBufferConfig& cfg )	OVERRIDE;
		virtual BufferPtr CreateBufferImpl( const ConstantBufferConfig& cfg )		OVERRIDE;

		virtual EffectPoolPtr	CreateEffectPoolImpl( const EffectPoolConfig& cfg )		OVERRIDE;
		virtual EffectPtr		CreateEffectImpl( const EffectConfig& cfg )				OVERRIDE;		
		virtual InputLayoutPtr	CreateInputLayoutImpl( const InputLayoutConfig& cfg )	OVERRIDE;
		virtual ViewportPtr		CreateViewportImpl( const ViewportConfig& cfg )			OVERRIDE;
		virtual ScissorRectPtr	CreateScissorRectImpl( const ScissorRectConfig& cfg )	OVERRIDE;

		virtual ShaderResourcePtr	CreateShaderResourceImpl( const ShaderResourceConfig& cfg )	OVERRIDE;
		virtual RenderTargetPtr		CreateRenderTargetImpl( const RenderTargetConfig& cfg )	OVERRIDE;
		virtual DepthStencilPtr		CreateDepthStencilImpl( const DepthStencilConfig& cfg ) OVERRIDE;
		virtual DeviceQueryPtr		CreateDeviceQueryImpl( const DeviceQueryConfig& cfg ) OVERRIDE;

		virtual StagedResourcePtr	CreateStagedBufferImpl( const StagedResourceConfig& cfg ) OVERRIDE;
		virtual StagedResourcePtr	CreateStagedTexture1DImpl( const StagedResourceConfig& cfg ) OVERRIDE;
		virtual StagedResourcePtr	CreateStagedTexture2DImpl( const StagedResourceConfig& cfg ) OVERRIDE;
		virtual StagedResourcePtr	CreateStagedTexture3DImpl( const StagedResourceConfig& cfg ) OVERRIDE;

		virtual RenderTargetPtr	GetDefaultRenderTargetImpl()			OVERRIDE;
		virtual TexturePtr		GetDefaultRenderTargetTextureImpl()		OVERRIDE;
		virtual DepthStencilPtr	GetDefaultDepthStencilImpl() 			OVERRIDE;
		virtual ViewportPtr		GetDefaultViewportImpl()				OVERRIDE;

		virtual void UpdateRenderTargetsImpl( const RenderTargets& rts, DepthStencilPtr ds ) OVERRIDE;
		virtual void ClearRenderTargetImpl( RenderTargetPtr rt, const Math::float4& colr ) OVERRIDE;
		virtual void ClearDepthStencilImpl( DepthStencilPtr ds, float depth, UINT32 stencil ) OVERRIDE;

		virtual void UpdateVSShaderResourcesImpl( const ShaderResources& shresses ) OVERRIDE;

		virtual void UpdateGSShaderResourcesImpl( const ShaderResources& shresses ) OVERRIDE;

		virtual void UpdatePSShaderResourcesImpl( const ShaderResources& shresses ) OVERRIDE;

		virtual void UpdateBuffersImpl( const Buffers& buffs ) OVERRIDE;

		virtual void UpdateIndexBufferImpl( const BufferPtr& buff ) OVERRIDE;

		virtual void UpdateSOBuffersImpl( const Buffers& buffs ) OVERRIDE;		

		virtual void UpdateViewportsImpl( const Viewports& vpts ) OVERRIDE;

		virtual void UpdateScissorRectsImpl( const ScissorRects& scrects ) OVERRIDE;
		
		virtual void SetPrimitiveTopologyImpl( const PrimitiveTopology* topology ) OVERRIDE;

		virtual void SetInputLayoutImpl( const InputLayout* inputLayout ) OVERRIDE;

		virtual void StartFrameImpl() OVERRIDE;
		virtual void EndFrameImpl() OVERRIDE;
		virtual void PresentImpl() OVERRIDE;

		virtual void DrawImpl( UINT64 vertexCount, UINT64 startVertex ) OVERRIDE;
		virtual void DrawIndexedImpl( UINT64 indexCount, UINT64 startIndexLocation, INT64 baseVertexLocation ) OVERRIDE;
		virtual void DrawInstancedImpl( UINT64 vertexCountPerInstance, UINT64 instanceCount, UINT64 startVertexLocation, UINT64 startInstanceLocation ) OVERRIDE;
		virtual void DrawIndexedInstancedImpl( UINT64 indexCountPerInstance, UINT64 instanceCount, UINT64 startIndexLocation, INT64 baseVertexLocation, UINT64 startInstanceLocation ) OVERRIDE;
		virtual void DrawAutoImpl() OVERRIDE;

		virtual void ResolveTextureMipImpl( const TexturePtr& dest, UINT32 destMip, const TexturePtr& src, UINT32 srcMip, const Format* fmt ) OVERRIDE;

		virtual void GetSOContentsImpl( Bytes& oBytes, UINT32 slot ) OVERRIDE;

		virtual void GetBufferContentsImpl( Bytes& oBytes, const BufferPtr& tex ) OVERRIDE;
		virtual void GetTexture1DContentsImpl( Bytes& oBytes, const TexturePtr& tex ) OVERRIDE;
		virtual void GetTexture2DContentsImpl( Bytes& oBytes, const TexturePtr& tex ) OVERRIDE;
		virtual void GetTexture3DContentsImpl( Bytes& oBytes, const TexturePtr& tex ) OVERRIDE;
		virtual void GetTextureCUBEContentsImpl( Bytes& oBytes, const TexturePtr& tex ) OVERRIDE;

		virtual void CopyImpl( TexturePtr dest, TexturePtr src  ) OVERRIDE;
		virtual void CopyImpl( BufferPtr dest, BufferPtr src ) OVERRIDE;

		virtual void CopyStagedBufferImpl( const StagedResourcePtr& staged ) OVERRIDE;
		virtual void CopyStagedTexture1DImpl( const StagedResourcePtr& staged ) OVERRIDE;
		virtual void CopyStagedTexture2DImpl( const StagedResourcePtr& staged ) OVERRIDE;
		virtual void CopyStagedTexture3DImpl( const StagedResourcePtr& staged ) OVERRIDE;


		virtual void PushMessageFilterImpl( const DeviceMessage** messages, UINT32 num ) OVERRIDE;
		virtual void PopMessageFilterImpl() OVERRIDE;

		virtual void ResetImpl() OVERRIDE;

		virtual INT32 StartPerfEventImpl( const String& name ) const OVERRIDE;
		virtual INT32 EndPerfEventImpl() const OVERRIDE; 

		// data
	private:
		ComPtr< DeviceType >			mDev;

		ComPtr< IDXGISwapChain >		mSwapChain;
		Types< D3D10FormatMap >::MemPtr	mFormatMap;
		Types< D3D10UsageMap >::MemPtr	mUsageMap;

#ifndef MD_D3D10_STATIC_LINK
		DynamicLibraryPtr				mD3D10Library;
#endif

		ComPtr< ID3D10InfoQueue >		mInfoQueue;
	};

	//------------------------------------------------------------------------

	class D3D10Device : public D3D10DeviceImpl
	{
		friend class D3D10TextureLoader;

		// construction/ destruction
	public:
		D3D10Device ( const DeviceConfig& cfg );
		virtual ~D3D10Device ();

		// stuff for friends ;]
	private:
		using D3D10DeviceImpl::GetD3D10Device;

	};

}


#endif