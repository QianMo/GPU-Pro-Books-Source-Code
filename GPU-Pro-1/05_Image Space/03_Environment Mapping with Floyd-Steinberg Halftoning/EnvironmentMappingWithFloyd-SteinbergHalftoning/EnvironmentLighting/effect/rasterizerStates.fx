shared RasterizerState defaultRasterizer
{
};

shared RasterizerState noCullRasterizer
{
	CullMode = none;
	FillMode = solid;
};

shared RasterizerState backfaceRasterizer
{
	CullMode = back;
	FillMode = solid;
};

shared RasterizerState wireframeRasterizer
{
	FillMode = wireFrame;
};