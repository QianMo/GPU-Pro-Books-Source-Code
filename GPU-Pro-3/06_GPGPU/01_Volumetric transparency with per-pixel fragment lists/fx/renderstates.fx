BlendState defaultBlender{};

BlendState addBlender{
	BlendEnable[0] = true;
	BlendEnable[1] = true;
	SrcBlend = one;
	DestBlend = one;
	BlendOp = add;
	SrcBlendAlpha = one;
	DestBlendAlpha = one;
	BlendOpAlpha = add;
};

RasterizerState defaultRasterizer{};

RasterizerState noCullRasterizer{	CullMode = none;	};

RasterizerState wireframeRasterizer{	CullMode = front; FillMode=wireframe;	};

DepthStencilState defaultCompositor{ };

DepthStencilState noDepthWriteCompositor{	DepthEnable = true;		DepthWriteMask = zero;	};

DepthStencilState noDepthTestCompositor{	DepthEnable = false;	DepthWriteMask = zero;	};