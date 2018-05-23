#pragma once

#include "Tracing.h"
#include <beLauncher/beAppWindow.h>
#include <beLauncher/beGraphicsConfig.h>
#include <beLauncher/beInput.h>
#include <lean/smart/resource_ptr.h>
#include <lean/smart/handle_guard.h>
#include <lean/smart/scoped_ptr.h>

namespace app
{

class App;

class AppWindow : private beLauncher::ApplicationWindow
{
private:
	bool m_bInitialized;

	lean::handle_guard< HWND, lean::smart::destroy_window_policy<HWND> > m_hWnd;

	lean::resource_ptr<beg::Device> m_graphicsDevice;
	lean::resource_ptr<beLauncher::InputDevice> m_inputDevice;

	lean::resource_ptr<beLauncher::Keyboard> m_keyboard;
	beLauncher::KeyboardState m_keyState;
	lean::resource_ptr<beLauncher::Mouse> m_mouse;
	beLauncher::MouseState m_mouseState;
	beLauncher::InputState m_input;

	lean::scoped_ptr<App> m_app;

protected:
	/// Called when the window has been created.
	void WindowCreated(HWND hWnd);
	/// Called when the window has been moved.
	void WindowMoved(HWND hWnd, const RECT &rect);
	/// Called when the window has been resized.
	void WindowResized(HWND hWnd, const RECT &rect);
	/// Called when the window is ready for frame updates.
	void WindowRun(HWND hWnd);
	/// Called when the window has been destroyed.
	void WindowDestroyed(HWND hWnd);

	/// Called for every other message in the queue.
	bool HandleMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam, LRESULT &result);

public:
	/// Constructor.
	AppWindow(HINSTANCE hInstance, const beLauncher::GraphicsConfig &graphicsConfig);
	/// Destructor.
	~AppWindow();

	/// Runs the application window.
	void Run();

	/// Gets the main window handle.
	HWND GetWindow() const { return m_hWnd; }
	/// Gets the graphics device.
	beg::Device* GetGraphicsDevice() const { return m_graphicsDevice; }
};

} // namespace