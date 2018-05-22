#ifndef DX11_RASTERIZER_STATE_H
#define DX11_RASTERIZER_STATE_H

#include <render_states.h>

// descriptor for setting up DX11_RASTERIZER_STATE
struct RASTERIZER_DESC
{
	RASTERIZER_DESC()
	{
		fillMode = SOLID_FILL;
		cullMode = NONE_CULL;
		scissorTest = false;
		multisampleEnable = false;
	}

	bool operator== (const RASTERIZER_DESC &desc) const
	{
		if(fillMode!=desc.fillMode)
			return false;
		if(cullMode!=desc.cullMode)
			return false;
		if(scissorTest!=desc.scissorTest)
			return false;
		if(multisampleEnable!=desc.multisampleEnable)
			return false;
		return true;
	}

	fillModes fillMode;
	cullModes cullMode;
	bool scissorTest;
	bool multisampleEnable;
};

// DX11_RASTERIZER_STATE
//   Wrapper for ID3D11RasterizerState.
class DX11_RASTERIZER_STATE
{
public:
  DX11_RASTERIZER_STATE()
	{
		rasterizerState = NULL;
	}

  ~DX11_RASTERIZER_STATE()
	{
		Release();
	}

	void Release();

	bool Create(const RASTERIZER_DESC &desc);

	void Set() const;

	RASTERIZER_DESC GetDesc() const
	{
		return desc;
	}

private:
	RASTERIZER_DESC desc;
  ID3D11RasterizerState *rasterizerState;

};

#endif