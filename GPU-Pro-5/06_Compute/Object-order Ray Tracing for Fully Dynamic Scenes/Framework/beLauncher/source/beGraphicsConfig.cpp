/*****************************************************/
/* breeze Framework Launch Lib  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beLauncherInternal/stdafx.h"
#include "beLauncher/beGraphicsConfig.h"

#include <lean/tags/noncopyable.h>
#include <lean/smart/resource_ptr.h>

#include "../resource/beLauncherResource.h"
#include <lean/logging/win_errors.h>

#include <lean/io/numeric.h>
#include <lean/strings/conversions.h>

namespace beLauncher
{

namespace
{

/// Graphics configuration dialog helper class.
class GraphicsConfigDlg : public lean::noncopyable
{
private:
	GraphicsConfig m_config;

	beGraphics::Adapter::display_mode_vector m_displayModes;

public:
	/// Constructor.
	GraphicsConfigDlg(const GraphicsConfig &config);

	/// Updates the given graphics configuration dialog.
	void Update(HWND hWnd);
	/// Updates the configuration from the given dialog.
	void UpdateFrom(HWND hWnd);

	/// Updates the adapters in the given configuration dialog.
	void UpdateAdapters(HWND hWnd);
	/// Updates the outputs in the given configuration dialog.
	void UpdateOutputs(HWND hWnd, beGraphics::Adapter *pAdapter);
	/// Updates the display modes in the given configuration dialog.
	void UpdateModes(HWND hWnd, beGraphics::Adapter *pAdapter, lean::uint4 outputID);
	/// Updates the anti-aliasing modes in the given configuration dialog.
	void UpdateAntiAliasing(HWND hWnd);

	/// Updates the display options in the given configuration dialog.
	void UpdateDisplayOptions(HWND hWnd);
	/// Updates the advanced options in the given configuration dialog.
	void UpdateAdvancedOptions(HWND hWnd);
	
	/// Updates the configuration after the adapter has been changed.
	void AdapterChanged(HWND hWnd);
	/// Updates the configuration after the output has been changed.
	void OutputChanged(HWND hWnd);
	/// Updates the configuration after the display mode has been changed.
	void ResolutionChanged(HWND hWnd);
	/// Updates the configuration after the anti-aliasing has been changed.
	void AntiAliasingChanged(HWND hWnd);
	
	/// Updates the configuration after the display options have been changed.
	void DisplayOptionsChanged(HWND hWnd);
	/// Updates the configuration after the advanced options have been changed.
	void AdvancedOptionsChanged(HWND hWnd);

	/// Gets the device description.
	const GraphicsConfig& GetConfig() { return m_config; }
};

} // namespace
} // namespace

// Constructor.
beLauncher::GraphicsConfigDlg::GraphicsConfigDlg(const GraphicsConfig &config)
	: m_config(config)
{
	if (!m_config.Graphics)
		m_config.Graphics = beGraphics::GetGraphics();
}

// Updates the given graphics configuration dialog.
void beLauncher::GraphicsConfigDlg::Update(HWND hWnd)
{
	// Updating adapters includes outputs & modes
	UpdateAdapters(hWnd);
	UpdateAntiAliasing(hWnd);

	UpdateDisplayOptions(hWnd);
	UpdateAdvancedOptions(hWnd);
}

// Updates the graphics configuration from the given dialog.
void beLauncher::GraphicsConfigDlg::UpdateFrom(HWND hWnd)
{
	// Reverse dependency order
	AdvancedOptionsChanged(hWnd);
	DisplayOptionsChanged(hWnd);
	
	AntiAliasingChanged(hWnd);
	OutputChanged(hWnd);
	ResolutionChanged(hWnd);
	AdapterChanged(hWnd);
}

// Updates the adapters in the given configuration dialog.
void beLauncher::GraphicsConfigDlg::UpdateAdapters(HWND hWnd)
{
	::SendMessageW(::GetDlgItem(hWnd, IDC_ADAPTER), CB_RESETCONTENT, NULL, NULL);

	const unsigned int adapterCount = m_config.Graphics->GetAdapterCount();

	for (unsigned int adapterID = 0; adapterID != adapterCount; ++adapterID)
	{
		lean::resource_ptr<beGraphics::Adapter> pAdapter = m_config.Graphics->GetAdapter(adapterID);

		::SendMessageW( ::GetDlgItem(hWnd, IDC_ADAPTER), CB_INSERTSTRING,
			adapterID, reinterpret_cast<LPARAM>(lean::utf_to_utf16(pAdapter->GetName()).c_str()) );

		if (adapterID == m_config.AdapterID)
		{
			::SendMessageW(::GetDlgItem(hWnd, IDC_ADAPTER), CB_SETCURSEL, adapterID, NULL);
			UpdateOutputs(hWnd, pAdapter);
		}
	}
}

// Updates the outputs in the given configuration dialog.
void beLauncher::GraphicsConfigDlg::UpdateOutputs(HWND hWnd, beGraphics::Adapter *pAdapter)
{
	::SendMessageW(::GetDlgItem(hWnd, IDC_OUTPUT), CB_RESETCONTENT, NULL, NULL);

	m_config.OutputCount = pAdapter->GetOutputCount();

	for (unsigned int outputID = 0; outputID != m_config.OutputCount; ++outputID)
	{
		std::basic_stringstream<lean::utf16_t> outputName;
		outputName << L"Display #" << (outputID + 1);

		::SendMessageW( ::GetDlgItem(hWnd, IDC_OUTPUT), CB_INSERTSTRING,
			outputID, reinterpret_cast<LPARAM>(outputName.str().c_str()) );

		if (outputID == m_config.OutputID)
		{
			::SendMessageW(::GetDlgItem(hWnd, IDC_OUTPUT), CB_SETCURSEL, outputID, NULL);
			UpdateModes(hWnd, pAdapter, outputID);
		}
	}
}

// Updates the display modes in the given configuration dialog.
void beLauncher::GraphicsConfigDlg::UpdateModes(HWND hWnd, beGraphics::Adapter *pAdapter, lean::uint4 outputID)
{
	::SendMessageW(::GetDlgItem(hWnd, IDC_RESOLUTION), CB_RESETCONTENT, NULL, NULL);

	m_displayModes = pAdapter->GetDisplayModes(outputID, m_config.SwapChain.Display.Format);
	bool modeSelected = false;

	for (size_t modeID = 0; modeID != m_displayModes.size(); ++modeID)
	{
		const beGraphics::DisplayMode &mode = m_displayModes[modeID];

		std::basic_stringstream<lean::utf16_t> modeName;
		modeName << mode.Width << "x" << mode.Height;
//			<< " @ " << mode.Refresh.ToFloat();

		::SendMessageW( ::GetDlgItem(hWnd, IDC_RESOLUTION), CB_INSERTSTRING,
			modeID, reinterpret_cast<LPARAM>(modeName.str().c_str()) );

		if (!modeSelected && mode.Width >= m_config.SwapChain.Display.Width && mode.Height >= m_config.SwapChain.Display.Height)
		{
			::SendMessageW(::GetDlgItem(hWnd, IDC_RESOLUTION), CB_SETCURSEL, modeID, NULL);
			modeSelected = true;
		}

		if (!modeSelected)
			::SendMessageW(::GetDlgItem(hWnd, IDC_RESOLUTION), CB_SETCURSEL, m_displayModes.size() - 1, NULL);
	}
}

// Updates the anti-aliasing modes in the given configuration dialog.
void beLauncher::GraphicsConfigDlg::UpdateAntiAliasing(HWND hWnd)
{
	::SendMessageW(::GetDlgItem(hWnd, IDC_ANTIALIASING), CB_RESETCONTENT, NULL, NULL);

	for (unsigned int i = 0; i <= 16; ++i)
	{
		std::basic_stringstream<lean::utf16_t> aaName;
		aaName << i << "x";

		::SendMessageW( ::GetDlgItem(hWnd, IDC_ANTIALIASING), CB_INSERTSTRING,
			i, reinterpret_cast<LPARAM>(aaName.str().c_str()) );

		if (i == m_config.SwapChain.Samples.Count)
			::SendMessageW(::GetDlgItem(hWnd, IDC_ANTIALIASING), CB_SETCURSEL, i, NULL);
	}
}

// Updates the display options in the given configuraton dialog.
void beLauncher::GraphicsConfigDlg::UpdateDisplayOptions(HWND hWnd)
{
	::CheckDlgButton(hWnd, IDC_FULLSCREEN, (m_config.DeviceDesc.Windowed) ? BST_UNCHECKED : BST_CHECKED);
	::CheckDlgButton(hWnd, IDC_MULTIHEAD, (m_config.DeviceDesc.MultiHead) ? BST_CHECKED : BST_UNCHECKED);
	::CheckDlgButton(hWnd, IDC_VSYNC, (m_config.VSync) ? BST_CHECKED : BST_UNCHECKED);
}

// Updates the advanced options in the given configuration dialog.
void beLauncher::GraphicsConfigDlg::UpdateAdvancedOptions(HWND hWnd)
{
	::SetWindowTextA(
		::GetDlgItem(hWnd, IDC_SAMPLINGQUALITY),
		static_cast<std::stringstream&>(std::stringstream() << m_config.SwapChain.Samples.Quality).str().c_str() );

	::CheckDlgButton(hWnd, IDC_REFERENCEDEVICE, (m_config.DeviceDesc.Type == beGraphics::DeviceType::Software) ? BST_CHECKED : BST_UNCHECKED);
}

// Updates the configuration after the adapter has been changed.
void beLauncher::GraphicsConfigDlg::AdapterChanged(HWND hWnd)
{
	int adapterID = static_cast<int>( ::SendMessageW(::GetDlgItem(hWnd, IDC_ADAPTER), CB_GETCURSEL, NULL, NULL) );

	if (adapterID != CB_ERR && adapterID != m_config.AdapterID)
	{
		m_config.AdapterID = adapterID;
	
		UpdateOutputs(hWnd, m_config.Graphics->GetAdapter(m_config.AdapterID).get());
	}
}

// Updates the configuration after the output has been changed.
void beLauncher::GraphicsConfigDlg::OutputChanged(HWND hWnd)
{
	int outputID = static_cast<int>( ::SendMessageW(::GetDlgItem(hWnd, IDC_OUTPUT), CB_GETCURSEL, NULL, NULL) );

	if (outputID != CB_ERR && m_config.OutputID != outputID)
	{
		m_config.OutputID = outputID;
		UpdateModes(hWnd, m_config.Graphics->GetAdapter(m_config.AdapterID).get(), m_config.OutputID);
	}
}

// Updates the configuration after the display mode has been changed.
void beLauncher::GraphicsConfigDlg::ResolutionChanged(HWND hWnd)
{
	int resolutionID = static_cast<int>( ::SendMessageW(::GetDlgItem(hWnd, IDC_RESOLUTION), CB_GETCURSEL, NULL, NULL) );

	if (resolutionID != CB_ERR)
		m_config.SwapChain.Display = m_displayModes[resolutionID];
}

// Updates the configuration after the anti-aliasing has been changed.
void beLauncher::GraphicsConfigDlg::AntiAliasingChanged(HWND hWnd)
{
	int sampleCount = static_cast<int>( ::SendMessageW(::GetDlgItem(hWnd, IDC_ANTIALIASING), CB_GETCURSEL, NULL, NULL) );

	if (sampleCount != CB_ERR)
		m_config.SwapChain.Samples.Count = sampleCount;
}

// Updates the configuration after the display options have been changed.
void beLauncher::GraphicsConfigDlg::DisplayOptionsChanged(HWND hWnd)
{
	m_config.DeviceDesc.Windowed = (::IsDlgButtonChecked(hWnd, IDC_FULLSCREEN) == BST_UNCHECKED);
	m_config.DeviceDesc.MultiHead = (::IsDlgButtonChecked(hWnd, IDC_MULTIHEAD) != BST_UNCHECKED);
	m_config.VSync = (::IsDlgButtonChecked(hWnd, IDC_VSYNC) != BST_UNCHECKED);
}

// Updates the configuration after the advanced options have been changed.
void beLauncher::GraphicsConfigDlg::AdvancedOptionsChanged(HWND hWnd)
{
	std::string samplingQuality;
	samplingQuality.resize(::GetWindowTextLengthA(::GetDlgItem(hWnd, IDC_SAMPLINGQUALITY)));
	samplingQuality.resize(::GetWindowTextA(::GetDlgItem(hWnd, IDC_SAMPLINGQUALITY), &samplingQuality[0], static_cast<int>(samplingQuality.size())));
	std::stringstream(samplingQuality) >> m_config.SwapChain.Samples.Quality;

	if (::IsDlgButtonChecked(hWnd, IDC_REFERENCEDEVICE) != BST_UNCHECKED)
		m_config.DeviceDesc.Type = beGraphics::DeviceType::Software;
	else if (m_config.DeviceDesc.Type == beGraphics::DeviceType::Software)
		m_config.DeviceDesc.Type = beGraphics::DeviceType::Hardware;
}

namespace beLauncher
{

namespace
{

struct GraphicsConfigDlgParams
{
	GraphicsConfigDlg *pDlg;
	HICON hIcon;

	GraphicsConfigDlgParams(GraphicsConfigDlg *pDlg, HICON hIcon)
		: pDlg(pDlg),
		hIcon(hIcon) { }
};

#ifdef _M_X64
	typedef INT_PTR DLG_PROC_RESULT;
#else
	typedef BOOL DLG_PROC_RESULT;
#endif

/// Graphics configuration dialog callback.
DLG_PROC_RESULT CALLBACK GraphicsConfigDlgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	GraphicsConfigDlg *pDlg = reinterpret_cast<GraphicsConfigDlg*>(::GetWindowLongPtrW(hWnd, GWLP_USERDATA));

	switch(uMsg)
	{
	case WM_INITDIALOG:
		{
			GraphicsConfigDlgParams *pParams = reinterpret_cast<GraphicsConfigDlgParams*>(lParam);
			
			::SendMessageW(hWnd, WM_SETICON, static_cast<WPARAM>(ICON_SMALL), reinterpret_cast<LPARAM>(pParams->hIcon));
			::SendMessageW(hWnd, WM_SETICON, static_cast<WPARAM>(ICON_BIG), reinterpret_cast<LPARAM>(pParams->hIcon));

			pDlg = pParams->pDlg;
			::SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pDlg));

			LEAN_ASSERT(pDlg);

			pDlg->Update(hWnd);
		}
		return TRUE;
	
	case WM_COMMAND:
		switch(LOWORD(wParam))
		{
		case IDOK:
			LEAN_ASSERT(pDlg);

			pDlg->UpdateFrom(hWnd);

			EndDialog(hWnd, 1);
			break;
		
		case IDC_ADAPTER:
			LEAN_ASSERT(pDlg);

			if (HIWORD(wParam) == CBN_SELCHANGE)
				pDlg->AdapterChanged(hWnd);
			break;

		case IDC_OUTPUT:
			LEAN_ASSERT(pDlg);

			if (HIWORD(wParam) == CBN_SELCHANGE)
				pDlg->OutputChanged(hWnd);
			break;

		case IDC_RESOLUTION:
			LEAN_ASSERT(pDlg);

			if (HIWORD(wParam) == CBN_SELCHANGE)
				pDlg->ResolutionChanged(hWnd);
			break;

		case IDCANCEL:
			EndDialog(hWnd, 0);
			break;
		}
		break;
	}

	return FALSE;
}

}

} // namespace

// Opens a graphics configuration dialog.
bool beLauncher::OpenGraphicsConfiguration(GraphicsConfig &config, HINSTANCE hInstance, HICON hIcon, HWND hParent)
{
	GraphicsConfigDlg dlg(config);

	if (hIcon == NULL)
		hIcon = ::LoadIconW(NULL, IDI_APPLICATION);

	INT_PTR result = ::DialogBoxParamW(hInstance, MAKEINTRESOURCE(IDD_BE_GRAPHICSCONFIG),
		hParent, &GraphicsConfigDlgProc, reinterpret_cast<LPARAM>(&GraphicsConfigDlgParams(&dlg, hIcon)));

	if (result == -1)
		LEAN_THROW_WIN_ERROR_MSG("DialogBoxParam()");

	if (result != 0)
		config = dlg.GetConfig();

	return (result != 0);
}
