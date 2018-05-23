
#include <Windows.h>

#include <Apps/GPULiquids/GPU3DParticleLiquids.hpp>
#include <Apps/GPULiquids/GPUShallowWaterSimulation.hpp>
#include <Apps/GPULiquids/GPU2DParticleLiquids.hpp>

#include <Common/Utils/Timer/Timer.hpp>

#include <Input\Keyboard.hpp>


HWND g_hWnd=0;
int32 g_iCurrent=0;
int32 g_width= 1000+16;//+16;;//1024+16;
int32 g_height= 1000+38;//+38;;// 768+38;

Vector<Dx11Renderer*, 3> g_Applications;

///<
bool Init(HINSTANCE _hInstance, HWND _hWnd)
{
	g_hWnd=_hWnd;

	///<
	g_Applications[0] = new Simple3DLiquids();	
	g_Applications[1] = new Simple2DLiquids();
	g_Applications[2] = new ParticleShallowWater();

	bool bApp = g_Applications[g_iCurrent]->Create(g_hWnd);	

	Keyboard::Get().Create(_hInstance,g_hWnd, g_Applications[g_iCurrent]->ScreenDims());

	g_Applications[g_iCurrent]->CreateMenu();

	return bApp;
}

///<
void Update()
{
	Keyboard::Get().Update();

	if (Keyboard::Get().Key(DIK_C))
	{
		ASSERT(g_Applications[g_iCurrent]!=NULL, "Large Index!");

		M::Release(g_Applications[g_iCurrent]);
		TwTerminate();

		g_iCurrent=((g_iCurrent+1)%g_Applications.Size());

		g_Applications[g_iCurrent]->Create(g_hWnd);
		g_Applications[g_iCurrent]->CreateMenu();
	}

	if(g_Applications[g_iCurrent])
		g_Applications[g_iCurrent]->Update();
	
}

///<
void Cleanup()
{
	for (uint32 i=0; i<g_Applications.Size(); ++i)
	{		
		M::Release(g_Applications[i]);
		M::Delete(&g_Applications[i]);			
	}

	Keyboard::Get().Release();
}


///< The window's message handler
LRESULT WINAPI MsgProc(HWND _hWnd, UINT _msg, WPARAM _wParam, LPARAM _lParam)
{
	if (TwEventWin(_hWnd,_msg,_wParam,_lParam))
		return 0;

	switch (_msg)
	{
		case WM_PAINT:
			Update();
			break;
		case WM_DESTROY:			
			PostQuitMessage(0);
			break;
		default:
			return DefWindowProc(_hWnd, _msg, _wParam, _lParam);
			break;
	}

	return 0;
}

///< Main
INT WINAPI wWinMain(HINSTANCE _hInst, HINSTANCE, LPWSTR, INT)
{
	const char* ClassName = "ClassName";

	///< Register the window class.
	WNDCLASSEX wc =
	{
		sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
		GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
		ClassName, NULL
	};
	RegisterClassEx(&wc);


	///< Create the application's window.
	HWND hWnd = CreateWindowEx(0,ClassName, "Prototypes",	WS_OVERLAPPEDWINDOW, 10, 10, g_width, g_height, NULL, NULL, wc.hInstance, NULL);

	
	///< Show the window.
	ShowWindow(hWnd, SW_SHOWDEFAULT);
	UpdateWindow(hWnd);

	bool bInitApp = Init(wc.hInstance, hWnd);
	ASSERT(bInitApp, "App is broken !");

	Timer mTimer;
	MSG msg;
	memset(&msg, 0, sizeof(msg));

	while (GetMessage(&msg, NULL, 0, 0))
	{	
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	Cleanup();

	return (int32) msg.wParam;

}

