/*******************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*******************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beAdapters.h"
#include "beGraphics/DX11/beDevice.h"
#include "beGraphics/DX11/beFormat.h"
#include <beGraphics/DX/beError.h>
#include <lean/strings/conversions.h>

namespace beGraphics
{

namespace DX11
{

namespace
{

/// Creates a DirectX 11 graphics interface factory.
lean::com_ptr<IDXGIFactory1, true> CreateDXGIFactory()
{
	lean::com_ptr<IDXGIFactory1> pFactory;
	
	BE_THROW_DX_ERROR_MSG(
		::CreateDXGIFactory1(IID_IDXGIFactory1, reinterpret_cast<void**>(pFactory.rebind())),
		"CreateDXGIFactory1()");

	LEAN_ASSERT(pFactory != nullptr);

	return pFactory.transfer();
}

// Fills the given vector with the list of adapters.
template <class AdapterPtr, class Allocator>
void EnumAdapters(IDXGIFactory1 *pFactory, std::vector<AdapterPtr, Allocator> &adapters)
{
	adapters.reserve(16);

	lean::com_ptr<IDXGIAdapter1> pAdapter;

	for (int i = 0; pFactory->EnumAdapters1(i, pAdapter.rebind()) != DXGI_ERROR_NOT_FOUND; ++i)
	{
		LEAN_ASSERT(pAdapter != nullptr);
		adapters.push_back(pAdapter);
	}
}

} // namespace

// Constructor.
Graphics::Graphics()
	: m_pFactory(CreateDXGIFactory())
{
	EnumAdapters(m_pFactory, m_adapters);
}

// Constructor.
Graphics::Graphics(IDXGIFactory1 *pFactory)
	: m_pFactory(pFactory)
{
	LEAN_ASSERT(m_pFactory != nullptr);

	EnumAdapters(m_pFactory, m_adapters);
}

// Destructor.
Graphics::~Graphics()
{
}

// Creates a device for the given adapter.
lean::resource_ptr<beGraphics::Device, true> Graphics::CreateDevice(
	const DeviceDesc &desc, const SwapChainDesc *swapChains,
	lean::uint4 adapterID, lean::uint4 outputID) const
{
	return lean::bind_resource<beGraphics::Device>(
		(adapterID < m_adapters.size())
			? new Device(m_adapters[adapterID], desc, swapChains, outputID)
			: nullptr );
}

// Gets the number of adapters available.
lean::uint4 Graphics::GetAdapterCount() const
{
	return static_cast<uint4>(m_adapters.size());
}

// Gets the adapter identified by the given number.
lean::resource_ptr<beGraphics::Adapter, 1> Graphics::GetAdapter(lean::uint4 adapterID) const
{
	return lean::bind_resource(
		(adapterID < m_adapters.size())
			? new Adapter(m_adapters[adapterID])
			: nullptr );
}

namespace
{

// Fills the given vector with the list of outputs.
template <class OutputPtr, class Allocator>
void EnumOutputs(IDXGIAdapter1 *pAdapter, std::vector<OutputPtr, Allocator> &outputs)
{
	outputs.reserve(4);

	lean::com_ptr<IDXGIOutput> pOutput;

	for (int i = 0; pAdapter->EnumOutputs(i, pOutput.rebind()) != DXGI_ERROR_NOT_FOUND; ++i)
	{
		LEAN_ASSERT(pOutput != nullptr);
		outputs.push_back(pOutput);
	}
}

} // namespace

// Constructor.
Adapter::Adapter(IDXGIAdapter1 *pAdapter)
	: m_pAdapter(pAdapter)
{
	LEAN_ASSERT(m_pAdapter != nullptr);

	EnumOutputs(m_pAdapter, m_outputs);
}

// Destructor.
Adapter::~Adapter()
{
}

// Gets the adapter's name.
Exchange::utf8_string Adapter::GetName() const
{
	DXGI_ADAPTER_DESC1 desc;
	
	BE_THROW_DX_ERROR_MSG(
		m_pAdapter->GetDesc1(&desc),
		"IDXGIAdapter1::GetDesc1()" );

	return lean::utf_to_utf8<Exchange::utf8_string>(desc.Description);
}

// Gets the number of outputs.
lean::uint4 Adapter::GetOutputCount() const
{
	return static_cast<uint4>(m_outputs.size());
}

// Gets whether the given format is supported for the given output.
bool Adapter::IsFormatSupported(lean::uint4 outputID, Format::T format)
{
	UINT displayModeCount = 0;

	BE_THROW_DX_ERROR_MSG(
			m_outputs[outputID]->GetDisplayModeList(ToAPI(format), 0, &displayModeCount, nullptr),
			"IDXGIOutput::GetDisplayModeList(), count" );

	return (displayModeCount != 0);
}

// Gets the display modes available for the given output.
Adapter::display_mode_vector Adapter::GetDisplayModes(lean::uint4 outputID, Format::T format, bool ignoreRefresh) const
{
	display_mode_vector displayModes;

	if (outputID < m_outputs.size())
	{
		DXGI_FORMAT formatDX = ToAPI(format);
		UINT displayModeCount = 0;

		BE_THROW_DX_ERROR_MSG(
			m_outputs[outputID]->GetDisplayModeList(formatDX, 0, &displayModeCount, nullptr),
			"IDXGIOutput::GetDisplayModeList(), count" );

		typedef std::vector<DXGI_MODE_DESC> display_mode_vector_dx;
		display_mode_vector_dx displayModesDX(displayModeCount);

		BE_THROW_DX_ERROR_MSG(
			m_outputs[outputID]->GetDisplayModeList(formatDX, 0, &displayModeCount, &displayModesDX[0]),
			"IDXGIOutput::GetDisplayModeList()" );

		displayModes.reserve(displayModeCount);

		const DXGI_MODE_DESC *pPrevModeDX = nullptr;

		for (UINT i = 0; i < displayModeCount; ++i)
		{
			DXGI_MODE_DESC &modeDX = displayModesDX[i];

			if (ignoreRefresh)
			{
				modeDX.RefreshRate.Numerator = 0;
				modeDX.RefreshRate.Denominator = 1;
			}

			if (!pPrevModeDX ||
				modeDX.Width != pPrevModeDX->Width ||
				modeDX.Height != pPrevModeDX->Height ||
				modeDX.RefreshRate.Numerator != pPrevModeDX->RefreshRate.Numerator ||
				modeDX.RefreshRate.Denominator != pPrevModeDX->RefreshRate.Denominator )
			{
				displayModes.push_back( FromAPI(displayModesDX[i]) );
				pPrevModeDX = &modeDX;
			}
		}
	}

	return displayModes;
}

} // namespace

} // namespace