shared BlendState defaultBlender
{
};

shared BlendState additiveBlender
{
	BlendEnable[0] = true;
	SrcBlend = src_alpha;
	DestBlend = one;
	BlendOp = add;
	SrcBlendAlpha = one;
	DestBlendAlpha = one;
	BlendOpAlpha = add;
};

shared BlendState transparencyBlender
{
	BlendEnable[0] = true;
	SrcBlend = src_alpha;
	DestBlend = inv_src_alpha;
	BlendOp = add;
	SrcBlendAlpha = one;
	DestBlendAlpha = one;
	BlendOpAlpha = add;
};

shared BlendState underlayBlender
{
	BlendEnable[0] = true;
	SrcBlend = inv_dest_alpha;
	DestBlend = dest_alpha;
	BlendOp = add;
	SrcBlendAlpha = one;
	DestBlendAlpha = one;
};
