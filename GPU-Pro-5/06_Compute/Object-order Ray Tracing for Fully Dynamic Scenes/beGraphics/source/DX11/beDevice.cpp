/*******************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*******************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beDevice.h"
#include "beGraphics/DX11/beFormat.h"
#include <beGraphics/DX/beError.h>
#include <lean/logging/log.h>

namespace beGraphics
{

namespace DX11
{

// Gets the device from the given device child.
lean::com_ptr<ID3D11Device> GetDevice(ID3D11DeviceChild &deviceChild)
{
	lean::com_ptr<ID3D11Device> pDevice;
	deviceChild.GetDevice(pDevice.rebind());
	return pDevice;
}

namespace
{

/// Converts the given DirectX feature level into a shader model version number.
lean::uint4 GetShaderModel(D3D_FEATURE_LEVEL featureLevel)
{
	switch (featureLevel)
	{
	case ::D3D_FEATURE_LEVEL_9_1:
		return 1;
	case ::D3D_FEATURE_LEVEL_9_2:
		return 2;
	case ::D3D_FEATURE_LEVEL_9_3:
		return 3;
	case ::D3D_FEATURE_LEVEL_10_0:
	case ::D3D_FEATURE_LEVEL_10_1:
		return 4;
	case ::D3D_FEATURE_LEVEL_11_0:
		return 5;
	}

	LEAN_ASSERT_UNREACHABLE();
}

/// Converts the given DirectX feature level into a shader model version number.
const lean::utf8_t* GetShaderModelString(D3D_FEATURE_LEVEL featureLevel)
{
	switch (featureLevel)
	{
	case ::D3D_FEATURE_LEVEL_9_1:
		return "<= Shader Model 2.0";
	case ::D3D_FEATURE_LEVEL_9_2:
		return "Shader Model 2.0";
	case ::D3D_FEATURE_LEVEL_9_3:
		return "Shader Model 3.0";
	case ::D3D_FEATURE_LEVEL_10_0:
	case ::D3D_FEATURE_LEVEL_10_1:
		return "Shader Model 4.0";
	case ::D3D_FEATURE_LEVEL_11_0:
		return "Shader Model 5.0";
	}

	LEAN_ASSERT_UNREACHABLE();
}

/// Prints information on the given device & adapter.
void LogDeviceAndAdapterInfo(IDXGIAdapter1 *pAdapter, const DeviceDesc &deviceDesc, D3D_FEATURE_LEVEL featureLevel)
{
	LEAN_LOG_BREAK();
	LEAN_LOG("DirectX 11 Device:");

	::DXGI_ADAPTER_DESC1 adapterDesc;

	if (BE_LOG_DX_ERROR_MSG(pAdapter->GetDesc1(&adapterDesc), "IDXGIAdapter1::GetDesc1()"))
		LEAN_LOG("  Adapter: " << adapterDesc.Description);

	LEAN_LOG("  " << GetShaderModelString(featureLevel));

	switch (deviceDesc.Type)
	{
	case DeviceType::Software:
		LEAN_LOG("  Device Type: Software");
		break;
	default:
		LEAN_LOG("  Device Type: Hardware");
	}

	LEAN_LOG("  Windowed: " << deviceDesc.Windowed);
	LEAN_LOG("  Multihead: " << deviceDesc.MultiHead);

	LEAN_LOG_BREAK();
}

/// Creates a DirectX 11 device.
lean::com_ptr<ID3D11Device, true> CreateD3DDevice(IDXGIAdapter1 *pAdapter, const DeviceDesc &desc, lean::uint4 outputID)
{
	lean::com_ptr<ID3D11Device> pDevice;
	lean::com_ptr<ID3D11DeviceContext> pImmediateContext;
	D3D_FEATURE_LEVEL featureLevel;

	BE_THROW_DX_ERROR_MSG(
		::D3D11CreateDevice(
			(desc.Type != DeviceType::Software) ? pAdapter : nullptr,
			(desc.Type != DeviceType::Software) ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_REFERENCE,
			NULL,
			(desc.Debug) ? D3D11_CREATE_DEVICE_DEBUG : 0,
			NULL,
			0,
			D3D11_SDK_VERSION,
			pDevice.rebind(),
			&featureLevel,
			pImmediateContext.rebind() ),
		"D3D11CreateDevice()");

	LEAN_ASSERT(pDevice != nullptr);
	LEAN_ASSERT(pImmediateContext != nullptr);

	LogDeviceAndAdapterInfo(pAdapter, desc, featureLevel);

	if (GetShaderModel(featureLevel) < desc.FeatureLevel)
		LEAN_THROW_ERROR_MSG("The device does not match the given minimum feature level.");

	return pDevice.transfer();
}

/// Gets the output speicified by the given number.
inline lean::com_ptr<IDXGIOutput, true> CheckedGetOutput(IDXGIAdapter1 *pAdapter, lean::uint4 outputID)
{
	lean::com_ptr<IDXGIOutput> pOutput;
	
	BE_THROW_DX_ERROR_MSG(
		pAdapter->EnumOutputs(outputID, pOutput.rebind()),
		"IDXGIAdapter1::EnumOutputs()" );

	LEAN_ASSERT(pOutput != nullptr);

	return pOutput.transfer();
}

/// Creates a device head swap chain.
lean::resource_ptr<SwapChain, true> CreateDeviceHeadSwapChain(ID3D11Device *pDevice,
	const DeviceDesc &desc, const SwapChainDesc &swapChainDesc,
	IDXGIAdapter1 *pAdapter, lean::uint4 outputID)
{
	return lean::bind_resource(
		new SwapChain(
			pDevice,
			ToSwapChainDesc(desc, swapChainDesc),
			CheckedGetOutput(pAdapter, outputID).get() )
		);
}

/// Gets the virtual head viewport for the given output.
Viewport GetHeadRect(IDXGIAdapter1 *pAdapter, lean::uint4 outputID)
{
	DXGI_OUTPUT_DESC desc;

	BE_THROW_DX_ERROR_MSG(
		CheckedGetOutput(pAdapter, outputID)->GetDesc(&desc),
		"IDXGIOutput::CheckedGetOutput()");
	
	return Viewport(static_cast<float>(desc.DesktopCoordinates.left),
		static_cast<float>(desc.DesktopCoordinates.top),
		static_cast<float>(desc.DesktopCoordinates.right),
		static_cast<float>(desc.DesktopCoordinates.bottom));
}

} // namespace

// Constructor.
Device::Device(IDXGIAdapter1 *pAdapter, const DeviceDesc &desc,
	const SwapChainDesc *swapChains, lean::uint4 outputID)
		: m_pDevice( CreateD3DDevice(pAdapter, desc, outputID) )
{
	if (swapChains)
	{
		if (desc.MultiHead && !desc.Windowed)
		{
			Viewport virtualScreenRect(2.0e16f, 2.0e16f, -2.0e16f, -2.0e16f);
			lean::com_ptr<IDXGIOutput> pOutput;

			// Create full-screen swap chain for each output
			for (int i = 0; pAdapter->EnumOutputs(i, pOutput.rebind()) != DXGI_ERROR_NOT_FOUND; ++i)
			{
				Viewport headRect = GetHeadRect(pAdapter, i);

				m_heads.push_back(
					Head(
						CreateDeviceHeadSwapChain(m_pDevice, desc, swapChains[i], pAdapter, i),
						headRect )
				);

				// Width and height misused as right and bottom
				virtualScreenRect.X = min(virtualScreenRect.X, headRect.X);
				virtualScreenRect.Y = min(virtualScreenRect.Y, headRect.Y);
				virtualScreenRect.Width = max(virtualScreenRect.Width, headRect.Width);
				virtualScreenRect.Height = max(virtualScreenRect.Height, headRect.Height);
			}

			if (m_heads.empty())
				LEAN_THROW_ERROR_MSG("No outputs connected to the device.");

			// Convert to actual width and height
			virtualScreenRect.Width -= virtualScreenRect.X;
			virtualScreenRect.Height -= virtualScreenRect.Y;

			for (head_vector::iterator it = m_heads.begin(); it != m_heads.end(); ++it)
			{
				Head &head = *it;

				// Transform to unit space [0; 1] & convert to actual width and height
				head.virtualViewport.X = (head.virtualViewport.X - virtualScreenRect.X) / virtualScreenRect.Width;
				head.virtualViewport.Y = (head.virtualViewport.Y - virtualScreenRect.Y) / virtualScreenRect.Height;
				head.virtualViewport.Width = (head.virtualViewport.Width - head.virtualViewport.X) / virtualScreenRect.Width;
				head.virtualViewport.Height = (head.virtualViewport.Height - head.virtualViewport.Y) / virtualScreenRect.Height;
			}
		}
		else
			// Create single swap chain
			m_heads.push_back(
				Head(
					CreateDeviceHeadSwapChain(m_pDevice, desc, *swapChains, pAdapter, outputID),
					Viewport(0.0f, 0.0f, 1.0f, 1.0f) )
				);
	}
}

// Constructor.
Device::Device(ID3D11Device *pDevice)
	: m_pDevice(pDevice)
{
	LEAN_ASSERT(m_pDevice != nullptr);
}

// Destructor.
Device::~Device()
{
}

// Gets the device feature level.
lean::uint4 Device::GetFeatureLevel() const
{
	return GetShaderModel(m_pDevice->GetFeatureLevel());
}

// Gets the number of device heads.
lean::uint4 Device::GetHeadCount() const
{
	return static_cast<uint4>(m_heads.size());
}

// Gets the swap chain of the device head identified by the given number.
SwapChain* Device::GetHeadSwapChain(lean::uint4 headID) const
{
	return (headID < m_heads.size())
		? m_heads[headID].pSwapChain
		: nullptr;
}

//  Gets a viewport corresponding to the requested device head's virtual desktop position.
Viewport Device::GetVirtualHeadViewport(lean::uint4 headID) const
{
	return (headID < m_heads.size())
		? m_heads[headID].virtualViewport
		: Viewport();
}

// Presents the rendered image.
void Device::Present(bool bVSync)
{
	for (head_vector::const_iterator it = m_heads.begin(); it != m_heads.end(); ++it)
		it->pSwapChain->Present(bVSync);
}

// Gets the back buffer.
lean::com_ptr<ID3D11Texture2D, true> GetBackBuffer(IDXGISwapChain *pSwapChain, uint4 index)
{
	lean::com_ptr<ID3D11Texture2D> pTexture;

	BE_LOG_DX_ERROR_MSG(
		pSwapChain->GetBuffer(index, IID_ID3D11Texture2D, reinterpret_cast<void**>(pTexture.rebind())),
		"IDXGISwapChain::GetBuffer()");

	return pTexture.transfer();
}

namespace
{

/// Constructs a DirectX 11 swap chain description from the given description.
DXGI_SWAP_CHAIN_DESC ToAPI(const SwapChainDesc &desc)
{
	DXGI_SWAP_CHAIN_DESC descDX;
	descDX.BufferDesc = DX11::ToAPI(desc.Display);
	if (descDX.BufferDesc.RefreshRate.Numerator == 0)
		descDX.BufferDesc.RefreshRate.Denominator = 0;
	descDX.BufferCount = desc.BufferCount;
	descDX.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	descDX.OutputWindow = reinterpret_cast<HWND>(desc.Window);
	descDX.Windowed = desc.Windowed;
	descDX.SampleDesc.Count = desc.Samples.Count;
	descDX.SampleDesc.Quality = desc.Samples.Quality;
	descDX.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	descDX.Flags = 0; // DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
	return descDX;
}

/// Constructs a swap chain description from the given DirectX 11 description.
SwapChainDesc FromAPI(const DXGI_SWAP_CHAIN_DESC &descDX)
{
	return SwapChainDesc(
		DX11::FromAPI(descDX.BufferDesc),
		descDX.BufferCount,
		descDX.OutputWindow,
		descDX.Windowed != FALSE,
		SampleDesc(
			descDX.SampleDesc.Count,
			descDX.SampleDesc.Quality ) );
}

/// Auto-adapts sample count & multisampling quality for the given display format.
void AdaptMultisampling(ID3D11Device *pDevice, DXGI_FORMAT format, DXGI_SAMPLE_DESC &sampleDesc)
{
	UINT sampleQualityLevelCount = 0;

	while (sampleQualityLevelCount == 0 && sampleDesc.Count != 0)
	{
		BE_LOG_DX_ERROR_MSG(
			pDevice->CheckMultisampleQualityLevels(format, sampleDesc.Count, &sampleQualityLevelCount),
			"ID3D11Device::CheckMultisampleQualityLevels()");

		if (sampleQualityLevelCount == 0)
			// Adapt sample count, if unavailable
			--sampleDesc.Count;
	}

	// Clamp sample count to valid range
	sampleDesc.Count = max<UINT>(sampleDesc.Count, 1);

	// Clamp multisampling quality to valid range
	UINT maxSampleQuality = (sampleQualityLevelCount != 0) ? sampleQualityLevelCount - 1 : 0;
	sampleDesc.Quality = (sampleDesc.Quality > 0) ? min(sampleDesc.Quality, maxSampleQuality) : maxSampleQuality;
}

/// Gets a swap chain description from the given swap chain.
SwapChainDesc GetSwapChainDesc(IDXGISwapChain *pSwapChain)
{
	DXGI_SWAP_CHAIN_DESC descDX;
	BE_THROW_DX_ERROR_MSG(
		pSwapChain->GetDesc(&descDX),
		"IDXGISwapChain::GetDesc()");
	return FromAPI(descDX);
}

/// Prints information on the given swap chain.
void LogSwapChainInfo(IDXGISwapChain *pSwapChain, const SwapChainDesc &desc)
{
	LEAN_LOG_BREAK();
	LEAN_LOG("DirectX 11 Swap Chain:");

	DXGI_SWAP_CHAIN_DESC actualDesc;
	BE_LOG_DX_ERROR_MSG(
		pSwapChain->GetDesc(&actualDesc),
		"IDXGISwapChain::GetDesc()");

	LEAN_LOG("  Resolution: " << actualDesc.BufferDesc.Width << " x " << actualDesc.BufferDesc.Height
		<< " (" << desc.Display.Width << " x " << desc.Display.Height << ")");
	LEAN_LOG("  Multisampling: " << actualDesc.SampleDesc.Count << " Q" << actualDesc.SampleDesc.Quality
		<< " (" << desc.Samples.Count << " Q" << desc.Samples.Quality << ")");
	LEAN_LOG("  Buffers: " << actualDesc.BufferCount << " (" << desc.BufferCount << ")");
	LEAN_LOG("  Windowed: " << desc.Windowed);
	LEAN_LOG("  Refresh: " << static_cast<float>(actualDesc.BufferDesc.RefreshRate.Numerator) / max(static_cast<float>(actualDesc.BufferDesc.RefreshRate.Denominator), 1.0f)
		<< " (" << desc.Display.Refresh.ToFloat()  << ")");
	LEAN_LOG("  Format: " << DX11::FromAPI(actualDesc.BufferDesc.Format) << " (" << desc.Display.Format << ")");

	LEAN_LOG_BREAK();
}

/// Creates a DirectX 11 swap chain.
lean::com_ptr<IDXGISwapChain, true> CreateD3DSwapChain(ID3D11Device *pDevice, const SwapChainDesc &desc, bool suppressFullScreen = true)
{
	lean::com_ptr<IDXGIDevice1> pGIDevice;
	
	BE_THROW_DX_ERROR_MSG(
		pDevice->QueryInterface(IID_IDXGIDevice1, reinterpret_cast<void**>(pGIDevice.rebind())),
		"ID3D11Device::QueryInterface()" );

	lean::com_ptr<IDXGIAdapter1> pAdapter;

	BE_THROW_DX_ERROR_MSG(
		pGIDevice->GetParent(IID_IDXGIAdapter1, reinterpret_cast<void**>(pAdapter.rebind())),
		"IDXGIDevice1::GetParent()" );

	lean::com_ptr<IDXGIFactory1> pFactory;

	BE_THROW_DX_ERROR_MSG(
		pAdapter->GetParent(IID_IDXGIFactory1, reinterpret_cast<void**>(pFactory.rebind())),
		"IDXGIAdapter1::GetParent()" );

	lean::com_ptr<IDXGISwapChain> pSwapChain;

	DXGI_SWAP_CHAIN_DESC descDX = ToAPI(desc);
	AdaptMultisampling(pDevice, descDX.BufferDesc.Format, descDX.SampleDesc);

	if (suppressFullScreen)
		descDX.Windowed = true;

	BE_THROW_DX_ERROR_MSG(
		pFactory->CreateSwapChain(pDevice, &descDX, pSwapChain.rebind()),
		"IDXGIFactory1::CreateSwapChain()" );

	LEAN_ASSERT(pSwapChain != nullptr);

	LogSwapChainInfo(pSwapChain, desc);

	return pSwapChain.transfer();
}

/// Asserts that the given pointer is non-null.
template <class T>
LEAN_INLINE T* AssertNonNullptr(T *pointer)
{
	LEAN_ASSERT(pointer);
	return pointer;
}

} // namespace

// Constructor.
SwapChain::SwapChain(ID3D11Device *pDevice, const SwapChainDesc &desc, IDXGIOutput *pOutput)
	: m_pSwapChain( CreateD3DSwapChain(pDevice, desc) ),
	m_desc( GetSwapChainDesc(m_pSwapChain) )
{
	if (!desc.Windowed)
		// Output may be nullptr
		BE_THROW_DX_ERROR_MSG(
			m_pSwapChain->SetFullscreenState(true, pOutput),
			"IDXGISwapChain::SetFullscreenState()" );
}

// Constructor.
SwapChain::SwapChain(IDXGISwapChain *pSwapChain)
	: m_pSwapChain( AssertNonNullptr(pSwapChain) ),
	m_desc( GetSwapChainDesc(pSwapChain) )
{
}

// Destructor.
SwapChain::~SwapChain()
{
	BOOL isFullscreen;

	if (BE_LOG_DX_ERROR_MSG(m_pSwapChain->GetFullscreenState(&isFullscreen, nullptr), "IDXGIOutput::GetFullscreenState") && isFullscreen)
		BE_LOG_DX_ERROR_MSG(
			m_pSwapChain->SetFullscreenState(FALSE, nullptr),
			"IDXGIOutput::SetFullscreenState");
}

// Presents the rendered image.
void SwapChain::Present(bool bVSync)
{
	m_pSwapChain->Present(bVSync ? 1 : 0, 0);
}

// Resizes the swap chain buffers.
void SwapChain::Resize(uint4 width, uint4 height)
{
	DXGI_SWAP_CHAIN_DESC descDX;
	BE_THROW_DX_ERROR_MSG(
		m_pSwapChain->GetDesc(&descDX),
		"IDXGISwapChain::GetDesc()");
	BE_THROW_DX_ERROR_MSG(
		m_pSwapChain->ResizeBuffers(
				descDX.BufferCount,
				width,
				height,
				descDX.BufferDesc.Format,
				descDX.Flags
			),
		"IDXGISwapChain::ResizeBuffers()");
}

// Gets the swap chain description.
SwapChainDesc SwapChain::GetDesc() const
{
	return m_desc;
}

} // namespace

// Creates a swap chain for the given device.
lean::resource_ptr<SwapChain, true> CreateSwapChain(const Device &device, const SwapChainDesc &desc)
{
	return lean::bind_resource<beGraphics::SwapChain>(
		new DX11::SwapChain(ToImpl(device), desc, nullptr)  );
}

} // namespace