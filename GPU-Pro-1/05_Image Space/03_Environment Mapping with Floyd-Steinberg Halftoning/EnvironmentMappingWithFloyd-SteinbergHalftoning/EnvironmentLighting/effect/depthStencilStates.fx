shared DepthStencilState defaultCompositor
{
};

shared DepthStencilState noDepthTestCompositor
{
	DepthEnable = false;
	DepthWriteMask = zero;
};
