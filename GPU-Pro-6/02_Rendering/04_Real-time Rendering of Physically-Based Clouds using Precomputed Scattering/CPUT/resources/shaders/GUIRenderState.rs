[DepthStencilStateDX11]
DepthEnable = false

[RasterizerStateDX11]
FillMode = D3D11_FILL_SOLID
CullMode = D3D11_CULL_BACK
FrontCounterClockwise = false
DepthBias = 0.0
DepthBiasClamp = 0.0
SlopeScaledDepthBias = 0.0
DepthClipEnable = true
ScissorEnable = false
MultisampleEnable = false
AntialiasedLineEnable = false

[RenderTargetBlendStateDX11_1]
blendenable = true
srcblend = D3D11_BLEND_SRC_ALPHA
destblend = D3D11_BLEND_INV_SRC_ALPHA
srcblendalpha = D3D11_BLEND_ONE
destblendalpha = D3D11_BLEND_ONE
blendop = D3D11_BLEND_OP_ADD
blendopalpha = D3D11_BLEND_OP_ADD
rendertargetwritemask = 15 // D3D11_COLOR_WRITE_ENABLE_ALL

[BlendStateDX11]
alphatocoverageenable = false
independentblendenable = false;

[SamplerDX11_1]
filter = D3D11_FILTER_MIN_MAG_MIP_POINT
AddressU = D3D11_TEXTURE_ADDRESS_CLAMP
AddressV = D3D11_TEXTURE_ADDRESS_CLAMP
AddressW = D3D11_TEXTURE_ADDRESS_CLAMP
MipLODBias = 0
MaxAnisotropy = 0
ComparisonFunc = D3D11_COMPARISON_NEVER
BorderColor0 = 0
BorderColor1 = 0
BorderColor2 = 0
BorderColor3 = 0
MinLOD = 0
MaxLOD = 3.402823466e+38f // D3D11_FLOAT32_MAX
