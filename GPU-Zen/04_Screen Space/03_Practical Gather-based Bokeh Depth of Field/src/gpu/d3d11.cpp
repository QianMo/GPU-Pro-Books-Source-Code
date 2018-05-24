#include <gpu/d3d11.h>
#include <system/file.h>
#include <image/main.h>


using namespace NSystem;
using namespace NImage;


D3D_DRIVER_TYPE NGPU::driverType = D3D_DRIVER_TYPE_NULL;
D3D_FEATURE_LEVEL NGPU::featureLevel = D3D_FEATURE_LEVEL_11_1;
ID3D11Device* NGPU::device = nullptr;
ID3D11Device1* NGPU::device1 = nullptr;
ID3D11DeviceContext* NGPU::deviceContext = nullptr;
ID3D11DeviceContext1* NGPU::deviceContext1 = nullptr;
IDXGISwapChain* NGPU::swapChain = nullptr;
IDXGISwapChain1* NGPU::swapChain1 = nullptr;
ID3D11RenderTargetView* NGPU::backBufferRTV = nullptr;


void NGPU::CreateD3D11(int width, int height)
{
	HRESULT hr = S_OK;

	uint createDeviceFlags = 0;
	#ifdef MAXEST_FRAMEWORK_DEBUG
		createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
	#endif

	HWND hwnd = GetActiveWindow();

	D3D_DRIVER_TYPE driverTypes[] =
	{
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_WARP,
		D3D_DRIVER_TYPE_REFERENCE,
	};

	D3D_FEATURE_LEVEL featureLevels[] =
	{
		D3D_FEATURE_LEVEL_11_1,
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0,
	};

	uint featureLevelsCount = ARRAY_SIZE(featureLevels);

	// device
	for (uint i = 0; i < ARRAY_SIZE(driverTypes); i++)
	{
		driverType = driverTypes[i];

		hr = D3D11CreateDevice(nullptr, driverType, nullptr, createDeviceFlags, featureLevels, featureLevelsCount, D3D11_SDK_VERSION, &device, &featureLevel, &deviceContext);

		// try D3D_FEATURE_LEVEL_11_0
		if (hr == E_INVALIDARG)
			hr = D3D11CreateDevice(nullptr, driverType, nullptr, createDeviceFlags, &featureLevels[1], featureLevelsCount - 1, D3D11_SDK_VERSION, &device, &featureLevel, &deviceContext);

		if (SUCCEEDED(hr))
			break;
	}

	ASSERT(hr == S_OK);

	// factory
	IDXGIFactory1* dxgiFactory = nullptr;
	{
		IDXGIDevice* dxgiDevice = nullptr;
		hr = device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));

		if (SUCCEEDED(hr))
		{
			IDXGIAdapter* dxgiAdapter = nullptr;
			hr = dxgiDevice->GetAdapter(&dxgiAdapter);

			if (SUCCEEDED(hr))
			{
				hr = dxgiAdapter->GetParent(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&dxgiFactory));
				dxgiAdapter->Release();
			}

			dxgiDevice->Release();
		}
	}

	ASSERT(hr == S_OK);

	// swap chain
	IDXGIFactory2* dxgiFactory2 = nullptr;
	hr = dxgiFactory->QueryInterface(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(&dxgiFactory2));
	if (dxgiFactory2)
	{
		hr = device->QueryInterface(__uuidof(ID3D11Device1), reinterpret_cast<void**>(&device1));
		if (SUCCEEDED(hr))
			deviceContext->QueryInterface( __uuidof(ID3D11DeviceContext1), reinterpret_cast<void**>(&deviceContext1) );

		DXGI_SWAP_CHAIN_DESC1 scd1;
		ZeroMemory(&scd1, sizeof(scd1));
		scd1.Width = width;
		scd1.Height = height;
		scd1.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		scd1.SampleDesc.Count = 1;
		scd1.SampleDesc.Quality = 0;
		scd1.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		scd1.BufferCount = 1;

		hr = dxgiFactory2->CreateSwapChainForHwnd( device, hwnd, &scd1, nullptr, nullptr, &swapChain1 );
		if (SUCCEEDED(hr))
			hr = swapChain1->QueryInterface(__uuidof(IDXGISwapChain), reinterpret_cast<void**>(&swapChain));

		dxgiFactory2->Release();
	}
	else
	{
		DXGI_SWAP_CHAIN_DESC scd;
		ZeroMemory(&scd, sizeof(scd));
		scd.BufferCount = 1;
		scd.BufferDesc.Width = width;
		scd.BufferDesc.Height = height;
		scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		scd.BufferDesc.RefreshRate.Numerator = 60;
		scd.BufferDesc.RefreshRate.Denominator = 1;
		scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		scd.OutputWindow = hwnd;
		scd.SampleDesc.Count = 1;
		scd.SampleDesc.Quality = 0;
		scd.Windowed = true;

		hr = dxgiFactory->CreateSwapChain(device, &scd, &swapChain);
	}

	ASSERT(hr == S_OK);

	dxgiFactory->Release();

	// back buffer RTV

	ID3D11Texture2D* backBufferTexture = nullptr;
	hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBufferTexture));
	ASSERT(hr == S_OK);

	hr = device->CreateRenderTargetView(backBufferTexture, nullptr, &backBufferRTV);
	backBufferTexture->Release();
	ASSERT(hr == S_OK);
}


void NGPU::DestroyD3D11()
{
	SAFE_RELEASE(device);
	SAFE_RELEASE(device1);
	SAFE_RELEASE(deviceContext);
	SAFE_RELEASE(deviceContext1);
	SAFE_RELEASE(swapChain);
	SAFE_RELEASE(swapChain1);
	SAFE_RELEASE(backBufferRTV);
}


bool NGPU::CompileShaderFromFile(const string& path, const string& entryPointName, const string& shaderModelName, const string& shaderMacros, ID3DBlob*& blob)
{
	struct Utils
	{
		struct ShaderMacro
		{
			string name;
			string definition;
		};

		vector<ShaderMacro> ShaderMacrosFromString(const string& shaderMacros)
		{
			vector<string> shaderMacrosArray_string = Split(shaderMacros, '|');

			vector<ShaderMacro> shaderMacrosArray;
			for (uint i = 0; i < shaderMacrosArray_string.size(); i++)
			{
				vector<string> macro_string = Split(shaderMacrosArray_string[i], '=');

				ShaderMacro macro;
				macro.name = macro_string[0];
				macro.definition = (macro_string.size() == 2 ? macro_string[1] : "");
				shaderMacrosArray.push_back(macro);
			}

			return shaderMacrosArray;
		}
	} utils;

	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
	#ifdef MAXEST_FRAMEWORK_DEBUG
		dwShaderFlags |= D3DCOMPILE_DEBUG;
		dwShaderFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
	#endif

	ID3DBlob* unstrippedBlob = nullptr;
	ID3DBlob* errorsBlob = nullptr;

	vector<Utils::ShaderMacro> shaderMacrosArray = utils.ShaderMacrosFromString(shaderMacros);
	D3D_SHADER_MACRO* shaderMacros_D3D11 = new D3D_SHADER_MACRO[shaderMacrosArray.size() + 1];
	for (uint i = 0; i < shaderMacrosArray.size(); i++)
	{
		shaderMacros_D3D11[i].Name = shaderMacrosArray[i].name.c_str();
		shaderMacros_D3D11[i].Definition = shaderMacrosArray[i].definition.c_str();
	}
	shaderMacros_D3D11[shaderMacrosArray.size()].Name = nullptr;
	shaderMacros_D3D11[shaderMacrosArray.size()].Definition = nullptr;

	if (FAILED(D3DCompileFromFile(StringToWString(path).c_str(), shaderMacros_D3D11, D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPointName.c_str(), shaderModelName.c_str(), dwShaderFlags, 0, &unstrippedBlob, &errorsBlob)))
	{
		if (errorsBlob)
		{
			OutputDebugStringA((char*)errorsBlob->GetBufferPointer());
			errorsBlob->Release();
		}

		return false;
	}

	D3DStripShader(unstrippedBlob->GetBufferPointer(), unstrippedBlob->GetBufferSize(), D3DCOMPILER_STRIP_REFLECTION_DATA | D3DCOMPILER_STRIP_DEBUG_INFO, &blob);
	unstrippedBlob->Release();

	delete[] shaderMacros_D3D11;

	// disassembly
	{
		ID3DBlob* disassemblyBlob = nullptr;
		D3DDisassemble(blob->GetBufferPointer(), blob->GetBufferSize(), 0, nullptr, &disassemblyBlob);

		// dump to file
		{
			string asmPath(path.begin(), path.end());
			for (uint i = 0; i < shaderMacrosArray.size(); i++)
			{
				asmPath += "--" + shaderMacrosArray[i].name;

				if (shaderMacrosArray[i].definition.length() > 0)
					asmPath += "=" + shaderMacrosArray[i].definition;
			}
			asmPath += "--asm.txt";

			File file;
			if (file.Open(asmPath, File::OpenMode::WriteText))
			{
				file.WriteBin((char*)disassemblyBlob->GetBufferPointer(), disassemblyBlob->GetBufferSize() - 1);
				file.Close();
			}
		}

		SAFE_RELEASE(disassemblyBlob);
	}

	SAFE_RELEASE(errorsBlob);

	return true;
}


void NGPU::CreateRenderTarget(int width, int height, DXGI_FORMAT format, RenderTarget& renderTarget)
{
	HRESULT hr;

	renderTarget.width = width;
	renderTarget.height = height;
	renderTarget.mipmapsCount = 1;
	renderTarget.type = Texture::Type::RenderTarget;

	//

	D3D11_TEXTURE2D_DESC td;
	ZeroMemory(&td, sizeof(td));
	td.Width = width;
	td.Height = height;
	td.MipLevels = 1;
	td.ArraySize = 1;
	td.Format = format;
	td.SampleDesc.Count = 1;
	td.SampleDesc.Quality = 0;
	td.Usage = D3D11_USAGE_DEFAULT;
	td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	td.CPUAccessFlags = 0;

	hr = device->CreateTexture2D(&td, NULL, &renderTarget.texture);
	ASSERT(hr == S_OK);

	//

	D3D11_RENDER_TARGET_VIEW_DESC rtvd;
	ZeroMemory(&rtvd, sizeof(rtvd));
	rtvd.Format = format;
	rtvd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	rtvd.Texture2D.MipSlice = 0;

	hr = device->CreateRenderTargetView(renderTarget.texture, &rtvd, &renderTarget.rtv);
	ASSERT(hr == S_OK);

	//

	D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
	ZeroMemory(&srvd, sizeof(srvd));
	srvd.Format = format;
	srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvd.Texture2D.MipLevels = 1;
	srvd.Texture2D.MostDetailedMip = 0;

	hr = device->CreateShaderResourceView(renderTarget.texture, &srvd, &renderTarget.srv);
	ASSERT(hr == S_OK);
}


void NGPU::CreateDepthStencilTarget(int width, int height, DepthStencilTarget& depthStencilTarget)
{
	HRESULT hr;

	depthStencilTarget.width = width;
	depthStencilTarget.height = height;
	depthStencilTarget.mipmapsCount = 1;
	depthStencilTarget.type = Texture::Type::DepthStencilTarget;

	//

	D3D11_TEXTURE2D_DESC td;
	ZeroMemory(&td, sizeof(td));
	td.Width = width;
	td.Height = height;
	td.MipLevels = 1;
	td.ArraySize = 1;
	td.Format = DXGI_FORMAT_R24G8_TYPELESS;
	td.SampleDesc.Count = 1;
	td.SampleDesc.Quality = 0;
	td.Usage = D3D11_USAGE_DEFAULT;
	td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	td.CPUAccessFlags = 0;
	td.MiscFlags = 0;

	hr = device->CreateTexture2D(&td, nullptr, &depthStencilTarget.texture);
	ASSERT(hr == S_OK);

	//

	D3D11_DEPTH_STENCIL_VIEW_DESC dsvd;
	ZeroMemory(&dsvd, sizeof(dsvd));
	dsvd.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	dsvd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	dsvd.Texture2D.MipSlice = 0;

	hr = device->CreateDepthStencilView(depthStencilTarget.texture, &dsvd, &depthStencilTarget.dsv);
	ASSERT(hr == S_OK);

	//

	D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
	ZeroMemory(&srvd, sizeof(srvd));
	srvd.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
	srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvd.Texture2D.MipLevels = 1;
	srvd.Texture2D.MostDetailedMip = 0;

	hr = device->CreateShaderResourceView(depthStencilTarget.texture, &srvd, &depthStencilTarget.srv);
	ASSERT(hr == S_OK);
}


void NGPU::CreateTexture(int width, int height, Texture& texture)
{
	HRESULT hr;

	texture.width = width;
	texture.height = height;
	texture.mipmapsCount = MipmapsCount(width, height);
	texture.type = Texture::Type::_2D;

	//

	D3D11_TEXTURE2D_DESC td;
	ZeroMemory(&td, sizeof(td));
	td.Width = width;
	td.Height = height;
	td.MipLevels = texture.mipmapsCount;
	td.ArraySize = 1;
	td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	td.SampleDesc.Count = 1;
	td.SampleDesc.Quality = 0;
	td.Usage = D3D11_USAGE_DEFAULT;
	td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	td.CPUAccessFlags = 0;

	hr = device->CreateTexture2D(&td, NULL, &texture.texture);
	ASSERT(hr == S_OK);

	//

	D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
	ZeroMemory(&srvd, sizeof(srvd));
	srvd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvd.Texture2D.MipLevels = texture.mipmapsCount;
	srvd.Texture2D.MostDetailedMip = 0;

	hr = device->CreateShaderResourceView(texture.texture, &srvd, &texture.srv);
	ASSERT(hr == S_OK);
}


bool NGPU::CreateVertexShader(const string& path, const string& shaderMacros, ID3D11VertexShader*& vertexShader)
{
	ID3DBlob* blob = nullptr;

	if (FAILED(CompileShaderFromFile(path, "VSMain", "vs_5_0", shaderMacros, blob)))
		return false;

	if (!blob)
		return false;

	if (FAILED(device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &vertexShader)))
	{	
		blob->Release();
		return false;
	}

	blob->Release();

	return true;
}


bool NGPU::CreatePixelShader(const string& path, const string& shaderMacros, ID3D11PixelShader*& pixelShader)
{
	ID3DBlob* blob = nullptr;

	if (FAILED(CompileShaderFromFile(path, "PSMain", "ps_5_0", shaderMacros, blob)))
		return false;

	if (!blob)
		return false;

	if (FAILED(device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &pixelShader)))
	{	
		blob->Release();
		return false;
	}

	blob->Release();

	return true;
}


bool NGPU::CreateVertexShader(const string& path, ID3D11VertexShader*& vertexShader)
{
	return CreateVertexShader(path, "", vertexShader);
}


bool NGPU::CreatePixelShader(const string& path, ID3D11PixelShader*& pixelShader)
{
	return CreatePixelShader(path, "", pixelShader);
}


bool NGPU::CreateInputLayout(const string& dummyVertexShaderPath, D3D11_INPUT_ELEMENT_DESC inputLayoutElements[], int inputLayoutElementsCount, ID3D11InputLayout*& inputLayout)
{
	ID3D11VertexShader* vs = nullptr;
	ID3DBlob* vsBlob = nullptr;

	if (FAILED(CompileShaderFromFile(dummyVertexShaderPath, "VSMain", "vs_5_0", "", vsBlob)))
		return false;

	if (!vsBlob)
		return false;

	if (FAILED(device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs)))
	{	
		vsBlob->Release();
		return false;
	}

	if (FAILED(device->CreateInputLayout(inputLayoutElements, inputLayoutElementsCount, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &inputLayout)))
	{
		vs->Release();
		vsBlob->Release();
		return false;
	}

	vs->Release();
	vsBlob->Release();

	return true;
}


void NGPU::CreateVertexBuffer(uint8* data, int dataSize, ID3D11Buffer*& vertexBuffer)
{
	HRESULT hr;

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = dataSize;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA subresourceData;
	ZeroMemory(&subresourceData, sizeof(subresourceData));
	subresourceData.pSysMem = data;

	hr = device->CreateBuffer(&bd, &subresourceData, &vertexBuffer);
	ASSERT(hr == S_OK);
}


void NGPU::CreateIndexBuffer(uint8* data, int dataSize, ID3D11Buffer*& indexBuffer)
{
	HRESULT hr;

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = dataSize;
	bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA subresourceData;
	ZeroMemory(&subresourceData, sizeof(subresourceData));
	subresourceData.pSysMem = data;

	hr = device->CreateBuffer(&bd, &subresourceData, &indexBuffer);
	ASSERT(hr == S_OK);
}


void NGPU::CreateConstantBuffer(int dataSize, ID3D11Buffer*& constantBuffer)
{
	HRESULT hr;

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = dataSize;
	bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bd.CPUAccessFlags = 0;

	hr = device->CreateBuffer(&bd, nullptr, &constantBuffer);
	ASSERT(hr == S_OK);
}


void NGPU::CreateSamplerState(ID3D11SamplerState*& samplerState, SamplerFilter filter, SamplerAddressing addressing, SamplerComparisonFunction comparisonFunction)
{
	HRESULT hr;

	D3D11_SAMPLER_DESC sd;
	ZeroMemory(&sd, sizeof(sd));
	if (filter == SamplerFilter::Point)
	{
		if (comparisonFunction == SamplerComparisonFunction::None)
			sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
		else
			sd.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
	}
	else if (filter == SamplerFilter::Linear)
	{
		if (comparisonFunction == SamplerComparisonFunction::None)
			sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		else
			sd.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR;
	}
	else if (filter == SamplerFilter::Anisotropic)
	{
		if (comparisonFunction == SamplerComparisonFunction::None)
			sd.Filter = D3D11_FILTER_ANISOTROPIC;
		else
			sd.Filter = D3D11_FILTER_COMPARISON_ANISOTROPIC;
	}
	sd.MaxAnisotropy = 16;
	if (addressing == SamplerAddressing::Clamp)
	{
		sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
		sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
		sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	}
	else if (addressing == SamplerAddressing::Wrap)
	{
		sd.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
		sd.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
		sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	}
	sd.ComparisonFunc = (D3D11_COMPARISON_FUNC)comparisonFunction;
	sd.MinLOD = 0;
	sd.MaxLOD = D3D11_FLOAT32_MAX;

	hr = device->CreateSamplerState(&sd, &samplerState);
	ASSERT(hr == S_OK);
}


void NGPU::CreateRasterizerState(ID3D11RasterizerState*& rasterizerState)
{
	HRESULT hr;

	D3D11_RASTERIZER_DESC rd;
	ZeroMemory(&rd, sizeof(rd));
	rd.FillMode = D3D11_FILL_SOLID;
	rd.CullMode = D3D11_CULL_BACK;
	rd.FrontCounterClockwise = true;
	rd.DepthBias = 0;
	rd.DepthBiasClamp = 0.0f;
	rd.SlopeScaledDepthBias = 0.0f;
	rd.DepthClipEnable = true;
	rd.ScissorEnable = false;
	rd.MultisampleEnable = false;
	rd.AntialiasedLineEnable = false;

	hr = device->CreateRasterizerState(&rd, &rasterizerState);
	ASSERT(hr == S_OK);
}


void NGPU::DestroyRenderTarget(RenderTarget& renderTarget)
{
	ASSERT(renderTarget.type == Texture::Type::RenderTarget);

	SAFE_RELEASE(renderTarget.texture);
	SAFE_RELEASE(renderTarget.rtv);
	SAFE_RELEASE(renderTarget.srv);
}


void NGPU::DestroyDepthStencilTarget(DepthStencilTarget& depthStencilTarget)
{
	ASSERT(depthStencilTarget.type == Texture::Type::DepthStencilTarget);

	SAFE_RELEASE(depthStencilTarget.texture);
	SAFE_RELEASE(depthStencilTarget.dsv);
	SAFE_RELEASE(depthStencilTarget.srv);
}


void NGPU::DestroyTexture(Texture& texture)
{
	ASSERT(texture.type == Texture::Type::_2D);

	SAFE_RELEASE(texture.texture);
	SAFE_RELEASE(texture.srv);
}


void NGPU::DestroyVertexShader(ID3D11VertexShader*& vertexShader)
{
	SAFE_RELEASE(vertexShader);
}


void NGPU::DestroyPixelShader(ID3D11PixelShader*& pixelShader)
{
	SAFE_RELEASE(pixelShader);
}


void NGPU::DestroyInputLayout(ID3D11InputLayout*& inputLayout)
{
	SAFE_RELEASE(inputLayout);
}


void NGPU::DestroyBuffer(ID3D11Buffer*& buffer)
{
	SAFE_RELEASE(buffer);
}


void NGPU::DestroySamplerState(ID3D11SamplerState*& samplerState)
{
	SAFE_RELEASE(samplerState);
}


void NGPU::DestroyRasterizerState(ID3D11RasterizerState*& rasterizerState)
{
	SAFE_RELEASE(rasterizerState);
}


void NGPU::UpdateTexture(Texture& texture, int mipmapIndex, uint8* data, int rowPitch)
{
	deviceContext->UpdateSubresource(
		texture.texture,
		D3D11CalcSubresource(mipmapIndex, 0, texture.mipmapsCount),
		NULL,
		data,
		rowPitch,
		0);
}


void NGPU::SetViewport(int width, int height)
{
	D3D11_VIEWPORT viewport;

	viewport.Width = (float)width;
	viewport.Height = (float)height;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.TopLeftX = 0.0f;
	viewport.TopLeftY = 0.0f;

	deviceContext->RSSetViewports(1, &viewport);
}
