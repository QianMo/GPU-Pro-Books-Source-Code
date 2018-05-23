/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "App.h"
#include "Stage.h"

#include <AntTweakBar.h>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#endif

#include <beGraphics/Any/beDevice.h>

#define ML_LITERAL_1 "."
#define ML_LITERAL_2 "@"
#define HELPTEXT "ENTER to capture the mouse, ESC to release.\n" \
		"Rotate using mouse,\nMove using W, A, S, D, SPACE, Q.\n\n" \
		"Press P to pause, \n0-9 to choose viewpoint\nHold R + press 0-9 to record viewpoint.\n\n" \
		"Contact: tobias" ML_LITERAL_1 "zirr" ML_LITERAL_2 "alphanew" ML_LITERAL_1 "net\nTwitter: " ML_LITERAL_2 "alphanew"

namespace app
{

TweakBarRuntime::TweakBarRuntime(beg::Device *device)
{
	TwInit(TW_DIRECT3D11, ToImpl(*device));
	const beg::SwapChainDesc &swapChainDesc = device->GetHeadSwapChain(0)->GetDesc();
	TwWindowSize(swapChainDesc.Display.Width, swapChainDesc.Display.Height);

	TwBar* dummyFrameBar = TwNewBar("Metrics");
	TwSetParam(dummyFrameBar, nullptr, "visible", TW_PARAM_CSTRING, 1, "false");
	TwSetParam(dummyFrameBar, nullptr, "size", TW_PARAM_INT32, 2, &swapChainDesc.Display.Width);

	TwDefine(" GLOBAL help='" HELPTEXT "' ");
}

TweakBarRuntime::~TweakBarRuntime()
{
	TwTerminate();
}

CUDARuntime::CUDARuntime(beg::Device *device)
{
#ifdef CUDA_ENABLED
//	cudaD3D11SetDirect3DDevice(ToImpl(*device));
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif
}

CUDARuntime::~CUDARuntime()
{
#ifdef CUDA_ENABLED
//	cudaDeviceReset();
#endif
}

// Constructor.
App::App(beg::Device *pGraphicsDevice)
	: m_graphicsDevice( LEAN_ASSERT_NOT_NULL(pGraphicsDevice) ),
	tweakBarRT(pGraphicsDevice),
	cudaRT(pGraphicsDevice),
	m_pStage( new Stage(m_graphicsDevice) )
{
}

// Destructor.
App::~App()
{
}

// Steps the application.
void App::Step(const beLauncher::InputState &input)
{
	m_pStage = m_pStage->GetNextScene();

	m_pStage->Step(input);
}

// Updates the screen rectangle.
void App::UpdateScreen(const bem::ivec2 &pos, const bem::ivec2 &ext)
{
	m_pStage->UpdateScreen(pos, ext);
}

} // namespace