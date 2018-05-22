#ifndef DX11_BLEND_STATE_H
#define DX11_BLEND_STATE_H

#include <render_states.h>

// descriptor for setting up DX11_BLEND_STATE
struct BLEND_DESC
{
	BLEND_DESC()
	{
		blend = false;
		srcColorBlend = ONE_BLEND;
		dstColorBlend = ONE_BLEND;
		blendColorOp = ADD_BLEND_OP;
		srcAlphaBlend = ONE_BLEND;
		dstAlphaBlend = ONE_BLEND;
		blendAlphaOp = ADD_BLEND_OP;
		constBlendColor.Set(0.0f,0.0f,0.0f,0.0f);
		colorMask = ALL_COLOR_MASK;
	}

	bool operator== (const BLEND_DESC &desc) const
	{
		if(blend!=desc.blend)
			return false;
		if(srcColorBlend!=desc.srcColorBlend)
			return false;
		if(dstColorBlend!=desc.dstColorBlend)
			return false;
		if(blendColorOp!=desc.blendColorOp)
			return false;
		if(srcAlphaBlend!=desc.srcAlphaBlend)
			return false;
		if(dstAlphaBlend!=desc.dstAlphaBlend)
			return false;
		if(blendAlphaOp!=desc.blendAlphaOp)
			return false;	
		if(constBlendColor!=desc.constBlendColor)
			return false;
		if(colorMask!=desc.colorMask)
			return false;
		return true;
	}

	bool blend;
	blendOptions srcColorBlend;
	blendOptions dstColorBlend;
	blendOps blendColorOp;
	blendOptions srcAlphaBlend;
	blendOptions dstAlphaBlend;
	blendOps blendAlphaOp;	
	COLOR constBlendColor;
	unsigned char colorMask;
};

// DX11_BLEND_STATE
//   Wrapper for ID3D11BlendState.
class DX11_BLEND_STATE
{
public:
  DX11_BLEND_STATE()
	{
    blendState = NULL;
	}

	~DX11_BLEND_STATE()
	{
		Release();
	}

	void Release();

	bool Create(const BLEND_DESC &desc);

	void Set() const;

	BLEND_DESC GetDesc() const
	{
		return desc;
	}

private:
	BLEND_DESC desc;
	ID3D11BlendState *blendState;

};

#endif