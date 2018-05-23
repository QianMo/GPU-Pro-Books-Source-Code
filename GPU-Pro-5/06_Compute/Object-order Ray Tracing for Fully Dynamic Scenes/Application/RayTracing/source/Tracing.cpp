/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

// Tracing.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "Tracing.h"

#include <beLauncher/beInitEngine.h>
#include <beCore/beFileSystem.h>
#include <beLauncher/beGraphicsConfig.h>

#include "AppWindow.h"

#include <cmath>

#include <lean/strings/conversions.h>
#include <lean/io/numeric.h>

#include <iostream>
#include <lean/logging/log.h>
#include <lean/logging/log_stream.h>

using namespace breeze;

namespace
{
	bool g_bExitSilently = false;
	void breakOnExit();
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
#ifdef _DEBUG
	// Catch nasty exit() calls by 3rd party libs
	atexit(&breakOnExit);

	// Mirror stream output
	struct Console
	{
		Console()
		{
			AllocConsole();
			freopen("CONIN$", "rb", stdin);
			freopen("CONOUT$", "wb", stdout);
			freopen("CONOUT$", "wb", stderr);
		}
		~Console()
		{
			FreeConsole();
		}
	} console;
#endif

	beLauncher::InitializeEngine("Logs/tracing.log", "tracing.filesystem.xml");

	// Logging
	struct ConsoleLog
	{
		lean::log_stream stream;

		ConsoleLog()
			: stream(&std::cout)
		{
			lean::error_log().add_target(&stream);
			lean::debug_log().add_target(&stream);
//			lean::info_log().add_target(&stream);
		}
		~ConsoleLog()
		{
			lean::error_log().remove_target(&stream);
			lean::debug_log().remove_target(&stream);
//			lean::info_log().remove_target(&stream);
		}
	} consoleLog;

	// Path environment
	if (!bec::FileSystem::Get().HasLocation("RayTracingEffects"))
	{
		bec::FileSystem::Get().AddPath("RayTracingEffects", "Data/Effects/RayTracing");
		bec::FileSystem::Get().AddPath("RayTracingEffects", "Data/Effects/2.0");
		bec::FileSystem::Get().AddPath("RayTracingEffects", "Data/Effects");
	}
	if (!bec::FileSystem::Get().HasLocation("RayTracingEffectCache"))
		bec::FileSystem::Get().AddPath("RayTracingEffectCache", "Data/Effects/Cache/RayTracing");
	if (!bec::FileSystem::Get().HasLocation("RayTracingEffectCacheDebug"))
		bec::FileSystem::Get().AddPath("RayTracingEffectCacheDebug", "Data/Effects/Cache/Debug/RayTracing");
	
	if (!bec::FileSystem::Get().HasLocation("Textures"))
		bec::FileSystem::Get().AddPath("Textures", "Data/Textures");
	if (!bec::FileSystem::Get().HasLocation("Meshes"))
		bec::FileSystem::Get().AddPath("Meshes", "Data/Meshes");

	// Default graphics options
	beLauncher::GraphicsConfig graphicsConfig;
	graphicsConfig.SwapChain.Display = beg::DisplayMode(1280, 720, beg::Format::R8G8B8X8U_SRGB);
	graphicsConfig.SwapChain.Samples = beg::SampleDesc(1);
	graphicsConfig.VSync = true;

	// Graphics configuration
	if (!beLauncher::OpenGraphicsConfiguration(graphicsConfig, hInstance))
	{
		g_bExitSilently = true;
		return 0;
	}

	try
	{
		app::AppWindow appWindow(hInstance, graphicsConfig);
		appWindow.Run();
	}
	catch (const std::exception &error)
	{
		::MessageBoxW(NULL, lean::utf_to_utf16(error.what()).c_str(), L"Unhandled exception", MB_OK | MB_SERVICE_NOTIFICATION);
#ifdef _DEBUG
		return -1;
#endif
	}
	catch (...)
	{
		::MessageBoxW(NULL, L"Unknown error.", L"Unhandled exception", MB_OK | MB_SERVICE_NOTIFICATION);
#ifdef _DEBUG
		return -1;
#endif
	}

	g_bExitSilently = true;
	return 0;
}

namespace
{

void breakOnExit()
{
	fflush(stdout);
	fflush(stderr);

	if (!g_bExitSilently)
		__debugbreak();
}

}

namespace app
{
	std::string identityString(void const* id)
	{
		return lean::int_to_string((uintptr_t) id);
	}
}
