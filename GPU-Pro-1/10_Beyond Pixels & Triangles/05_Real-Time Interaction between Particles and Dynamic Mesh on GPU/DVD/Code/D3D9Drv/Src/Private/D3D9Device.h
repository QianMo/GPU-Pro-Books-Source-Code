#ifndef D3D9DEVICE_DEVICE_H_INCLUDED
#define D3D9DEVICE_DEVICE_H_INCLUDED

#include "Wrap3D\Src\Device.h"
#include "WrapSys\Src\Forw.h"

#include "Forw.h"
#include "D3D9CapsConstants.h"

namespace Mod
{


	// impl is to hide real private stuff from friends

	class D3D9DeviceImpl : public Device
	{
		// types & consts
	public:
		static const UINT32 NUM_SO_BUFFER_SLOTS				= D3D_MAX_SIMULTANEOUS_RENDERTARGETS;
		static const UINT32 NUM_SIMULTANEOUS_RENDERTARGETS	= D3D_MAX_SIMULTANEOUS_RENDERTARGETS;
		static const UINT32 NUM_VIEWPORTS					= 1;
		static const UINT32 NUM_SCISSORRECTS				= 1;

		typedef IDirect3DDevice9 DeviceType;

		// construction/ destruction
	public:
		D3D9DeviceImpl ( const DeviceConfig& cfg );
		virtual ~D3D9DeviceImpl ();

		// manipulation/ access
	public:
		const D3D9FormatMap& GetD3D9FormatMap() const;
#if 0
		const D3D9UsageMap& GetD3D9UsageMap() const;
#endif

		// stuff for befrienders
	protected:
		DeviceType* GetD3D9Device() const;

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

		virtual RenderTargetPtr	GetDefaultRenderTargetImpl()		OVERRIDE;
		virtual TexturePtr		GetDefaultRenderTargetTextureImpl()	OVERRIDE;
		virtual DepthStencilPtr	GetDefaultDepthStencilImpl() 		OVERRIDE;
		virtual ViewportPtr		GetDefaultViewportImpl()			OVERRIDE;

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

		virtual void GetBufferContentsImpl( Bytes& oBytes, const BufferPtr& buf ) OVERRIDE;
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
		D3D9CapsConstants				mCaps;

		D3DPRIMITIVETYPE				mDrawPrimitiveType;
		const D3D9InputLayout*			mInputLayout;
	
		ComPtr< DeviceType >			mDev;
		ComPtr<D3D9EffectStateManager>	mEffectStateManager;

		Types< D3D9FormatMap >::MemPtr	mFormatMap;

		D3D9TextureCoordinatorPtr		mTextureCoordinator;
	};

	//------------------------------------------------------------------------

	class D3D9Device : public D3D9DeviceImpl
	{
		friend class D3D9TextureLoader;
		friend class D3D9EffectStateManager;
		// types
	public:
		typedef D3D9DeviceImpl Base;

		// construction/ destruction
	public:
		D3D9Device ( const DeviceConfig& cfg );
		virtual ~D3D9Device ();

		// stuff for friends ;]
	private:
		using D3D9DeviceImpl::GetD3D9Device;

	};

}

#endif