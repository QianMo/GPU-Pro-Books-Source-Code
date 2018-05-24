#pragma once


#include <essentials/stl.h>

#include <d3d11_1.h>


namespace NGPU
{
	struct Texture
	{
		enum class Type { _2D, RenderTarget, DepthStencilTarget };

		Type type;
		int width, height;
		int mipmapsCount;

		ID3D11Texture2D* texture;
		ID3D11ShaderResourceView* srv;
		ID3D11RenderTargetView* rtv;
		ID3D11DepthStencilView* dsv;
	};

	enum class SamplerFilter { Point, Linear, Anisotropic };
	enum class SamplerAddressing { Wrap, Clamp };
	enum class SamplerComparisonFunction { None, Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always };

	typedef Texture RenderTarget;
	typedef Texture DepthStencilTarget;
}
