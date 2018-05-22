#ifndef DX11_DEPTH_STENCIL_STATE_H
#define DX11_DEPTH_STENCIL_STATE_H

#include <render_states.h>

// descriptor for setting up DX11_DEPTH_STENCIL_STATE
struct DEPTH_STENCIL_DESC
{
	DEPTH_STENCIL_DESC()
	{
		depthTest = true;
		depthMask = true;
		depthFunc = LEQUAL_COMP_FUNC;
		stencilTest = false;
		stencilRef = 0;
		stencilMask = ~0;
		stencilFailOp = KEEP_STENCIL_OP;
		stencilDepthFailOp = INCR_SAT_STENCIL_OP;
		stencilPassOp = INCR_SAT_STENCIL_OP;
		stencilFunc = ALWAYS_COMP_FUNC;
	}

	bool operator== (const DEPTH_STENCIL_DESC &desc) const
	{
		if(depthTest!=desc.depthTest)
			return false;
		if(depthMask!=desc.depthMask)
			return false;
		if(depthFunc!=desc.depthFunc)
			return false;
		if(stencilTest!=desc.stencilTest)
			return false;
		if(stencilRef!=desc.stencilRef)
			return false;
		if(stencilMask!=desc.stencilMask)
			return false;
		if(stencilFailOp!=desc.stencilFailOp)
			return false;
		if(stencilDepthFailOp!=desc.stencilDepthFailOp)
			return false;
		if(stencilPassOp!=desc.stencilPassOp)
			return false;
		if(stencilFunc!=desc.stencilFunc)
			return false;
		return true;
	}

	bool depthTest;
	bool depthMask;
	comparisonFuncs depthFunc;
	bool stencilTest;
	unsigned int stencilRef;
	unsigned int stencilMask;
	stencilOps stencilFailOp;
	stencilOps stencilDepthFailOp;
	stencilOps stencilPassOp;
	comparisonFuncs stencilFunc;
};

// DX11_DEPTH_STENCIL_STATE
//   Wrapper for ID3D11DepthStencilState.
class DX11_DEPTH_STENCIL_STATE
{
public:
  DX11_DEPTH_STENCIL_STATE()
	{
    depthStencilState = NULL;
	}

	~DX11_DEPTH_STENCIL_STATE()
	{
		Release();
	}

	void Release();

	bool Create(const DEPTH_STENCIL_DESC &desc);

	void Set() const;

	DEPTH_STENCIL_DESC GetDesc() const
	{
		return desc;
	}

private:
	DEPTH_STENCIL_DESC desc;
	ID3D11DepthStencilState *depthStencilState;

};

#endif