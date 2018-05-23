/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "AppWindow.h"
#include "App.h"
#include <vector>

namespace app
{

namespace
{

/// Creates the main window.
HWND CreateMainWindow(HINSTANCE hInstance, beLauncher::ApplicationWindow *pAppWindow,
	const lean::utf16_ntri &title, const beLauncher::GraphicsConfig &graphicsConfig)
{
	HWND hWnd = beLauncher::CreateAppWindow(title.c_str(), hInstance, pAppWindow);
	
	beLauncher::ScaleWindow(hWnd, graphicsConfig.SwapChain.Display.Width, graphicsConfig.SwapChain.Display.Height);
	beLauncher::CenterWindow(hWnd);

	return hWnd;
}

/// Creates the graphics device.
lean::resource_ptr<beg::Device, true> CreateGraphicsDevice(HINSTANCE hInstance, HWND hMainWnd,
	const beLauncher::GraphicsConfig &graphicsConfig)
{
	beg::DeviceDesc deviceDesc(graphicsConfig.DeviceDesc);
	beg::SwapChainDesc swapChainDesc(graphicsConfig.SwapChain);

	// DEBUG: OVERRIDE
//	swapChainDesc.Display.Width = 256;
//	swapChainDesc.Display.Height = 256;

	std::vector<beg::SwapChainDesc> swapChains(graphicsConfig.OutputCount, swapChainDesc);

	deviceDesc.Window = hMainWnd;
#ifdef LEAN_DEBUG_BUILD
//	deviceDesc.Debug = true;
#endif
	swapChains[0].Window = hMainWnd;

	// Create auxiliary windows for multi-head swap chains
	if (deviceDesc.MultiHead)
		for (unsigned int i = 1; i < graphicsConfig.OutputCount; ++i)
			swapChains[i].Window = beLauncher::CreateAuxWindow(hInstance, hMainWnd);

	return graphicsConfig.Graphics->CreateDevice(deviceDesc, &swapChains[0],
		graphicsConfig.AdapterID, graphicsConfig.OutputID);
}

} // namespace

#define ML_LITERAL_1 L"."
#define ML_LITERAL_2 L"@"

// Constructor.
AppWindow::AppWindow(HINSTANCE hInstance, const beLauncher::GraphicsConfig &graphicsConfig)
	: m_bInitialized(false),
	m_hWnd( CreateMainWindow(hInstance, this, L"Object-order Ray Tracing for Fully Dynamic Scenes :: GPU Pro 5 Demo :: Contact: tobias" ML_LITERAL_1 L"zirr" ML_LITERAL_2 L"alphanew" ML_LITERAL_1 L"net | " ML_LITERAL_2 L"alphanew", graphicsConfig) ),
	m_graphicsDevice( CreateGraphicsDevice(hInstance, m_hWnd, graphicsConfig) ),
	m_inputDevice( new_resource beLauncher::InputDevice(m_hWnd) ),
	m_keyboard( new_resource beLauncher::Keyboard(m_inputDevice) ),
	m_mouse( new_resource beLauncher::Mouse(m_inputDevice) ),
	m_input(&m_keyState, &m_mouseState),
	m_app( new App(m_graphicsDevice) )
{
	m_bInitialized = true;
}

// Destructor.
AppWindow::~AppWindow()
{
	m_bInitialized = false;
}

// Runs the application window.
void AppWindow::Run()
{
	beLauncher::RunAppWindow(this, m_hWnd);
}

// Called when the window has been created.
void AppWindow::WindowCreated(HWND hWnd)
{
}

// Called when the window has been moved.
void AppWindow::WindowMoved(HWND hWnd, const RECT &rect)
{
	if (m_bInitialized)
	{
		bem::ivec2 pos = bem::vec<int>(rect.left, rect.top);
		bem::ivec2 ext = bem::vec<int>(rect.right - rect.left, rect.bottom - rect.top);
		m_app->UpdateScreen(pos, ext);
	}
}

// Called when the window has been resized.
void AppWindow::WindowResized(HWND hWnd, const RECT &rect)
{
	if (m_bInitialized)
	{
		bem::ivec2 pos = bem::vec<int>(rect.left, rect.top);
		bem::ivec2 ext = bem::vec<int>(rect.right - rect.left, rect.bottom - rect.top);
		m_app->UpdateScreen(pos, ext);
	}
}

// Called when the window is ready for frame updates.
void AppWindow::WindowRun(HWND hWnd)
{
	if (m_bInitialized)
		m_app->Step(m_input);
	SetHandled(m_keyState);
	SetHandled(m_mouseState);
}

// Called when the window has been destroyed.
void AppWindow::WindowDestroyed(HWND hWnd)
{
	// Destroy & release early
	if (m_bInitialized)
	{
		m_bInitialized = false;

		m_hWnd.detach();
		m_app = nullptr;
		m_graphicsDevice = nullptr;
	}
}

} // namespace

#include <AntTweakBar.h>

// Called for every other message in the queue.
bool app::AppWindow::HandleMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam, LRESULT &result)
{
	if (const RAWINPUT *rawInput = m_inputDevice->Process(msg, wParam, lParam))
	{
		m_keyboard->Process(m_keyState, *rawInput);
		m_mouse->Process(m_mouseState, *rawInput);
	}
	return TwEventWin(hWnd, msg, wParam, lParam) != 0;
}
