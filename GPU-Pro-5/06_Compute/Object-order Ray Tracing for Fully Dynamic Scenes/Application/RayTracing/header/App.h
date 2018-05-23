#pragma once

#include "Tracing.h"
#include <beGraphics/beDevice.h>
#include <beLauncher/beInput.h>
#include <lean/smart/resource_ptr.h>
#include <lean/smart/scoped_ptr.h>

#include <beMath/beVector.h>

namespace app
{

struct TweakBarRuntime
{
	TweakBarRuntime(beg::Device *device);
	~TweakBarRuntime();
};

struct CUDARuntime
{
	CUDARuntime(beg::Device *device);
	~CUDARuntime();
};

class Stage;

/// Application.
class App
{
private:
	lean::resource_ptr<beg::Device> m_graphicsDevice;
	
	TweakBarRuntime tweakBarRT;
	CUDARuntime cudaRT;

	lean::scoped_ptr<Stage> m_pStage;

public:
	/// Constructor.
	App(beg::Device *graphicsDevice);
	/// Destructor.
	~App();

	/// Steps the application.
	void Step(const beLauncher::InputState &input);

	/// Updates the screen rectangle.
	void UpdateScreen(const bem::ivec2 &pos, const bem::ivec2 &ext);

	/// Gets the graphics device.
	beg::Device* GetGraphicsDevice() const { return m_graphicsDevice; }
};

} // namespace