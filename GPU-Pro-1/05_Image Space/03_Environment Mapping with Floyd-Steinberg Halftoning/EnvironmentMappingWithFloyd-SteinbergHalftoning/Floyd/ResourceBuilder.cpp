#include "DXUT.h"
#include "ResourceBuilder.h"
#include "ResourceOwner.h"
#include "ResourceSet.h"
#include "xmlParser.h"
#include "ScriptRenderTargetViewVariable.h"
#include "ScriptResourceVariable.h"
#include "ScriptVariableClass.h"

ResourceBuilder::ResourceBuilder(XMLNode& resourcesNode, bool swapChainBound)
{
	this->swapChainBound = swapChainBound;

	loadBuffers(resourcesNode);
	loadTexture1Ds(resourcesNode);
	loadTexture2Ds(resourcesNode);
	loadRenderTargetViews(resourcesNode);
	loadShaderResourceViews(resourcesNode);
	XMLNode variablesNode = resourcesNode.getChildNode(L"variables");
	if(!variablesNode.isEmpty())
		loadVariables(variablesNode);
}

void ResourceBuilder::loadBuffers(XMLNode& resourcesNode)
{
	int iBuffer = 0;
	XMLNode bufferNode;
	while( !(bufferNode = resourcesNode.getChildNode(L"Buffer", iBuffer)).isEmpty() )
	{
		const wchar_t* name = bufferNode|L"name";
		if(name)
		{
			D3D10_BUFFER_DESC desc;
			desc.ByteWidth= bufferNode.readLong(L"byteWidth", 0);
			desc.Usage = bufferNode.readUsage(L"usage");

			desc.BindFlags = 0;
			int iFlag = 0;
			XMLNode flagNode;
			while( !(flagNode = bufferNode.getChildNode(L"binding", iFlag)).isEmpty() )
			{
				desc.BindFlags |= flagNode.readBindFlag(L"flag");
				iFlag++;
			}
			desc.CPUAccessFlags = 0;
			int iCPUFlag = 0;
			while( !(flagNode = bufferNode.getChildNode(L"cpuAccess", iCPUFlag)).isEmpty() )
			{
				desc.CPUAccessFlags |= flagNode.readCPUAccessFlag(L"flag");
				iCPUFlag++;
			}
			desc.MiscFlags = bufferNode.readLong(L"miscFlags");

			bufferDescDirectory[name] = desc;
		}

		iBuffer++;
	}
}

void ResourceBuilder::loadTexture1Ds(XMLNode& resourcesNode)
{
	int iTexture1D = 0;
	XMLNode texture1DNode;
	while( !(texture1DNode = resourcesNode.getChildNode(L"Texture1D", iTexture1D)).isEmpty() )
	{
		const wchar_t* name = texture1DNode|L"name";
		if(name)
		{
			D3D10_TEXTURE1D_DESC desc;
			desc.Width = texture1DNode.readLong(L"width", 0);
			desc.MipLevels = texture1DNode.readLong(L"mipLevels", 1);
			desc.ArraySize = texture1DNode.readLong(L"arraySize", 1);
			desc.Format = texture1DNode.readFormat(L"format", DXGI_FORMAT_UNKNOWN);
			desc.Usage = texture1DNode.readUsage(L"usage");

			desc.BindFlags = 0;
			int iFlag = 0;
			XMLNode flagNode;
			while( !(flagNode = texture1DNode.getChildNode(L"binding", iFlag)).isEmpty() )
			{
				desc.BindFlags |= flagNode.readBindFlag(L"flag");
				iFlag++;
			}
			desc.CPUAccessFlags = 0;
			int iCPUFlag = 0;
			while( !(flagNode = texture1DNode.getChildNode(L"cpuAccess", iCPUFlag)).isEmpty() )
			{
				desc.CPUAccessFlags |= flagNode.readCPUAccessFlag(L"flag");
				iCPUFlag++;
			}
			desc.MiscFlags = texture1DNode.readLong(L"miscFlags");

			texture1DDescDirectory[name] = desc;
		}

		iTexture1D++;
	}
}


void ResourceBuilder::loadTexture2Ds(XMLNode& resourcesNode)
{
	int iTexture2D = 0;
	XMLNode texture2DNode;
	while( !(texture2DNode = resourcesNode.getChildNode(L"Texture2D", iTexture2D)).isEmpty() )
	{
		const wchar_t* name = texture2DNode|L"name";
		if(name)
		{
			D3D10_TEXTURE2D_DESC desc;
			desc.Width = texture2DNode.readLong(L"width", 0);
			if(desc.Width == 0)
			{
				desc.Width = (unsigned int)texture2DNode.readLong(L"widthDivisionFactor", 1) | 0x80000000;
			}
			desc.Height = texture2DNode.readLong(L"height", 0);
			if(desc.Height == 0)
			{
				desc.Height = (unsigned int)texture2DNode.readLong(L"heightDivisionFactor", 1) | 0x80000000;
			}
			desc.MipLevels = texture2DNode.readLong(L"mipLevels", 1);
			desc.ArraySize = texture2DNode.readLong(L"arraySize", 1);
			desc.Format = texture2DNode.readFormat(L"format", DXGI_FORMAT_UNKNOWN);
			desc.Usage = texture2DNode.readUsage(L"usage");

			desc.BindFlags = 0;
			int iFlag = 0;
			XMLNode flagNode;
			while( !(flagNode = texture2DNode.getChildNode(L"binding", iFlag)).isEmpty() )
			{
				desc.BindFlags |= flagNode.readBindFlag(L"flag");
				iFlag++;
			}
			desc.CPUAccessFlags = 0;
			int iCPUFlag = 0;
			while( !(flagNode = texture2DNode.getChildNode(L"cpuAccess", iCPUFlag)).isEmpty() )
			{
				desc.CPUAccessFlags |= flagNode.readCPUAccessFlag(L"flag");
				iCPUFlag++;
			}
			desc.MiscFlags = texture2DNode.readLong(L"miscFlags");
			desc.SampleDesc.Count = texture2DNode.readLong(L"sampleCount", 1);
			desc.SampleDesc.Quality = texture2DNode.readLong(L"sampleQuality", 0);

			texture2DDescDirectory[name] = desc;
		}

		iTexture2D++;
	}
}

void ResourceBuilder::loadShaderResourceViews(XMLNode& resourcesNode)
{
	int iShaderResourceView = 0;
	XMLNode shaderResourceViewNode;
	while( !(shaderResourceViewNode = resourcesNode.getChildNode(L"ShaderResourceView", iShaderResourceView)).isEmpty() )
	{
		const wchar_t* name = shaderResourceViewNode|L"name";
		const wchar_t* resourceName = shaderResourceViewNode|L"resource";
		if(name && resourceName)
		{
			SRVParameters srvp;
			srvp.resourceName = resourceName;
			D3D10_SHADER_RESOURCE_VIEW_DESC& desc = srvp.desc;
			 // use resource format by default
			desc.Format = shaderResourceViewNode.readFormat(L"format", DXGI_FORMAT_UNKNOWN);
			desc.ViewDimension = shaderResourceViewNode.readShaderResourceViewDimension(L"dimension", D3D10_SRV_DIMENSION_UNKNOWN);
			switch(desc.ViewDimension)
			{
			case D3D10_SRV_DIMENSION_BUFFER:
				desc.Buffer.ElementOffset = shaderResourceViewNode.readLong(L"elementOffset", 0);
				desc.Buffer.ElementWidth = shaderResourceViewNode.readLong(L"elementWidth", 0);
				break;
			case D3D10_SRV_DIMENSION_TEXTURE1D:
				desc.Texture1D.MipLevels = shaderResourceViewNode.readLong(L"mipLevels", 1);
				desc.Texture1D.MostDetailedMip = shaderResourceViewNode.readLong(L"mostDetailedMip", 0);
				break;
			case D3D10_SRV_DIMENSION_TEXTURE2D:
				desc.Texture2D.MipLevels = shaderResourceViewNode.readLong(L"mipLevels", 1);
				desc.Texture2D.MostDetailedMip = shaderResourceViewNode.readLong(L"mostDetailedMip", 0);
				break;
			case D3D10_SRV_DIMENSION_TEXTURE3D:
				desc.Texture3D.MipLevels = shaderResourceViewNode.readLong(L"mipLevels", 1);
				desc.Texture3D.MostDetailedMip = shaderResourceViewNode.readLong(L"mostDetailedMip", 0);
				break;
			case D3D10_SRV_DIMENSION_TEXTURE1DARRAY:
				desc.Texture1DArray.MipLevels = shaderResourceViewNode.readLong(L"mipLevels", 1);
				desc.Texture1DArray.MostDetailedMip = shaderResourceViewNode.readLong(L"mostDetailedMip", 0);
				desc.Texture1DArray.FirstArraySlice = shaderResourceViewNode.readLong(L"firstArraySlice", 0);
				desc.Texture1DArray.ArraySize = shaderResourceViewNode.readLong(L"arraySize", 1);
				break;
			case D3D10_SRV_DIMENSION_TEXTURE2DARRAY:
				desc.Texture2DArray.MipLevels = shaderResourceViewNode.readLong(L"mipLevels", 1);
				desc.Texture2DArray.MostDetailedMip = shaderResourceViewNode.readLong(L"mostDetailedMip", 0);
				desc.Texture2DArray.FirstArraySlice = shaderResourceViewNode.readLong(L"firstArraySlice", 0);
				desc.Texture2DArray.ArraySize = shaderResourceViewNode.readLong(L"arraySize", 1);
				break;
			case D3D10_SRV_DIMENSION_TEXTURE2DMS:
				break;
			case D3D10_SRV_DIMENSION_TEXTURE2DMSARRAY:
				desc.Texture2DMSArray.FirstArraySlice = shaderResourceViewNode.readLong(L"firstArraySlice", 0);
				desc.Texture2DMSArray.ArraySize = shaderResourceViewNode.readLong(L"arraySize", 1);
				break;
			}
			srvDescDirectory[name] = srvp;
		}

		iShaderResourceView++;
	}
}

void ResourceBuilder::loadRenderTargetViews(XMLNode& resourcesNode)
{
	int iRenderTargetView = 0;
	XMLNode renderTargetViewNode;
	while( !(renderTargetViewNode = resourcesNode.getChildNode(L"RenderTargetView", iRenderTargetView)).isEmpty() )
	{
		const wchar_t* name = renderTargetViewNode|L"name";
		const wchar_t* resourceName = renderTargetViewNode|L"resource";
		if(name && resourceName)
		{
			RTVParameters rtvp;
			rtvp.resourceName = resourceName;
			D3D10_RENDER_TARGET_VIEW_DESC& desc = rtvp.desc;
			 // use resource format by default
			desc.Format = renderTargetViewNode.readFormat(L"format", DXGI_FORMAT_UNKNOWN);
			desc.ViewDimension = renderTargetViewNode.readRenderTargetViewDimension(L"dimension", D3D10_RTV_DIMENSION_UNKNOWN);
			switch(desc.ViewDimension)
			{
			case D3D10_RTV_DIMENSION_BUFFER:
				desc.Buffer.ElementOffset = renderTargetViewNode.readLong(L"elementOffset", 0);
				desc.Buffer.ElementWidth = renderTargetViewNode.readLong(L"elementWidth", 0);
				break;
			case D3D10_RTV_DIMENSION_TEXTURE1D:
				desc.Texture1D.MipSlice = renderTargetViewNode.readLong(L"mipSlice", 0);
				break;
			case D3D10_RTV_DIMENSION_TEXTURE2D:
				desc.Texture2D.MipSlice = renderTargetViewNode.readLong(L"mipSlice", 0);
				break;
			case D3D10_RTV_DIMENSION_TEXTURE3D:
				desc.Texture3D.MipSlice = renderTargetViewNode.readLong(L"mipSlice", 0);
				desc.Texture3D.FirstWSlice = renderTargetViewNode.readLong(L"firstWSlice", 0);
				desc.Texture3D.WSize = renderTargetViewNode.readLong(L"WSize", 0);
				break;
			case D3D10_RTV_DIMENSION_TEXTURE1DARRAY:
				desc.Texture1DArray.MipSlice = renderTargetViewNode.readLong(L"mipSlice", 0);
				desc.Texture1DArray.FirstArraySlice = renderTargetViewNode.readLong(L"firstArraySlice", 0);
				desc.Texture1DArray.ArraySize = renderTargetViewNode.readLong(L"arraySize", 1);
				break;
			case D3D10_RTV_DIMENSION_TEXTURE2DARRAY:
				desc.Texture2DArray.MipSlice = renderTargetViewNode.readLong(L"mipSlice", 0);
				desc.Texture2DArray.FirstArraySlice = renderTargetViewNode.readLong(L"firstArraySlice", 0);
				desc.Texture2DArray.ArraySize = renderTargetViewNode.readLong(L"arraySize", 1);
				break;
			case D3D10_RTV_DIMENSION_TEXTURE2DMS:
				break;
			case D3D10_RTV_DIMENSION_TEXTURE2DMSARRAY:
				desc.Texture2DMSArray.FirstArraySlice = renderTargetViewNode.readLong(L"firstArraySlice", 0);
				desc.Texture2DMSArray.ArraySize = renderTargetViewNode.readLong(L"arraySize", 1);
				break;
			}
			rtvDescDirectory[name] = rtvp;
		}

		iRenderTargetView++;
	}
}


void ResourceBuilder::instantiate(ResourceSet* resourceSet, ID3D10Device* device)
{
	BufferDescDirectory::iterator iBufferDesc = bufferDescDirectory.begin();
	while(iBufferDesc != bufferDescDirectory.end())
	{
		D3D10_BUFFER_DESC desc = iBufferDesc->second;
		ID3D10Buffer* buffer;
		device->CreateBuffer(&desc, NULL, &buffer);
		resourceSet->addResource(iBufferDesc->first, buffer, swapChainBound);
		iBufferDesc++;
	}

	D3D10_TEXTURE2D_DESC defaultTexture2DDesc;
	memset(&defaultTexture2DDesc, 0, sizeof(D3D10_TEXTURE2D_DESC));

	if(swapChainBound)
	{
		ScriptRenderTargetViewVariable* defaultVar = resourceSet->getRenderTargetViewVariable(L"default");
		if(defaultVar)
		{
			ID3D10RenderTargetView* defaultRTV = defaultVar->getRenderTargetView();
			if(defaultRTV)
			{
				ID3D10Texture2D* defaultRenderTargetResource;
				defaultRTV->GetResource((ID3D10Resource**)&defaultRenderTargetResource);
				defaultRenderTargetResource->GetDesc(&defaultTexture2DDesc);
				defaultRenderTargetResource->Release();
			}
		}
	}

	Texture2DDescDirectory::iterator iTexture2DDesc = texture2DDescDirectory.begin();
	while(iTexture2DDesc != texture2DDescDirectory.end())
	{
		D3D10_TEXTURE2D_DESC desc = iTexture2DDesc->second;
		if(desc.Width == 0)
			desc.Width = defaultTexture2DDesc.Width;
		else if(desc.Width & 0x80000000)
		{
			desc.Width &= ~0x80000000;
			desc.Width = defaultTexture2DDesc.Width / desc.Width;
		}
		if(desc.Height == 0)
			desc.Height = defaultTexture2DDesc.Height;
		else if(desc.Height & 0x80000000)
		{
			desc.Height &= ~0x80000000;
			desc.Height = defaultTexture2DDesc.Height / desc.Height;
		}
		if(desc.Format == DXGI_FORMAT_UNKNOWN)
			desc.Format = defaultTexture2DDesc.Format;
		ID3D10Texture2D* texture2D;
		if(iTexture2DDesc->first.compare(L"randomMap") == 0)
		{
			
			D3DXVECTOR4* randomData = new D3DXVECTOR4[desc.Width * desc.Height];
			for(unsigned int ir=0; ir<desc.Width * desc.Height; ir++)
			{
				float fi = (float)rand() / RAND_MAX * 6.28; 
				float v = (float)rand() / RAND_MAX;
	// floyd specular hack
/*				randomData[ir].x = cos(fi) * sqrt(v);
				randomData[ir].y = sin(fi) * sqrt(v);
				randomData[ir].z = sqrt( 1-v );*/
				randomData[ir].x = cos(fi) * pow(v, 0.0005f);
				randomData[ir].y = sin(fi) * pow(v, 0.0005f);
				randomData[ir].z = pow( 1-v , 0.0005f);
				randomData[ir].w = ((float)rand() / RAND_MAX);
			}

			D3D10_SUBRESOURCE_DATA randomInitData;
			randomInitData.pSysMem = randomData;
			randomInitData.SysMemPitch = desc.Width * sizeof(D3DXVECTOR4);

			device->CreateTexture2D(&desc, &randomInitData, &texture2D);
			delete [] randomData;
		}
		else if(iTexture2DDesc->first.compare(L"gridDirectionMap") == 0)
		{
			signed short ddata[16*64*4];
			for(int j=0; j<64; j++)
				for(int k=0; k<16; k++)
				{
					float samx, samy, samz;
					samy = sin(j / 64.0 * 3.14 * 2);
					samx = cos(j / 64.0 * 3.14 * 2);
					float sctx, scty;
//					scty = sqrt((k+0.5) / 16.0); //sin((k+1) / 16.0 * 3.14 / 2.0);
//					sctx = sqrt(1 - (k+0.5) / 16.0); //cos((k+1) / 16.0 * 3.14 / 2.0);
// floyd specular hack
					scty = pow((k+0.5) / 16.0, 0.0005);
					sctx = pow(1 - (k+0.5) / 16.0, 0.0005); 
					samx *= scty;
					samy *= scty;
					samz = sctx;
					ddata[4 * (j * 16 + k)+0] = samx * 0x7fff;
					ddata[4 * (j * 16 + k)+1] = samy * 0x7fff;
					ddata[4 * (j * 16 + k)+2] = samz * 0x7fff;
					ddata[4 * (j * 16 + k)+3] = 0;
				}

			D3D10_SUBRESOURCE_DATA dInitData;
			dInitData.pSysMem = ddata;
			dInitData.SysMemPitch = 16 * 4 * sizeof(signed short);

			device->CreateTexture2D(&desc, &dInitData, &texture2D);
		}
		else
			device->CreateTexture2D(&desc, NULL, &texture2D);
		resourceSet->addResource(iTexture2DDesc->first, texture2D, swapChainBound);
		iTexture2DDesc++;
	}

	Texture1DDescDirectory::iterator iTexture1DDesc = texture1DDescDirectory.begin();
	while(iTexture1DDesc != texture1DDescDirectory.end())
	{
		D3D10_TEXTURE1D_DESC desc = iTexture1DDesc->second;
		if(desc.Width == 0)
			desc.Width = defaultTexture2DDesc.Width;
		if(desc.Format == DXGI_FORMAT_UNKNOWN)
			desc.Format = defaultTexture2DDesc.Format;
		ID3D10Texture1D* texture1D;
		device->CreateTexture1D(&desc, NULL, &texture1D);
		resourceSet->addResource(iTexture1DDesc->first, texture1D, swapChainBound);
		iTexture1DDesc++;
	}

	RTVDescDirectory::iterator iRTVDesc = rtvDescDirectory.begin();
	while(iRTVDesc != rtvDescDirectory.end())
	{
		ScriptResourceVariable* resourceVar = resourceSet->getResourceVariable(iRTVDesc->second.resourceName);
		if(resourceVar)
		{
			ID3D10Resource* resource = resourceVar->getResource();
			if(resource)
			{
				D3D10_RENDER_TARGET_VIEW_DESC desc = iRTVDesc->second.desc;
				ID3D10RenderTargetView* rtv;
				HRESULT hr = device->CreateRenderTargetView(resource, &desc, &rtv);
				resourceSet->addRenderTargetView(iRTVDesc->first, rtv, swapChainBound);
			}
		}
		iRTVDesc++;
	}

	SRVDescDirectory::iterator iSRVDesc = srvDescDirectory.begin();
	while(iSRVDesc != srvDescDirectory.end())
	{
		ScriptResourceVariable* resourceVar = resourceSet->getResourceVariable(iSRVDesc->second.resourceName);
		if(resourceVar)
		{
			ID3D10Resource* resource = resourceVar->getResource();
			if(resource)
			{
				D3D10_SHADER_RESOURCE_VIEW_DESC desc = iSRVDesc->second.desc;
				if(desc.Format == DXGI_FORMAT_UNKNOWN)
				{
					switch(desc.ViewDimension)
					{
					case D3D10_SRV_DIMENSION_BUFFER:
						{
							//D3D10_BUFFER_DESC bufiDesc;
							//((ID3D10Buffer*)resource)->GetDesc(&bufiDesc);
							desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
						}
						break;
					case D3D10_SRV_DIMENSION_TEXTURE1D:
						{
							D3D10_TEXTURE1D_DESC texDesc;
							((ID3D10Texture1D*)resource)->GetDesc(&texDesc);
							desc.Format = texDesc.Format;
						}
						break;
					case D3D10_SRV_DIMENSION_TEXTURE2D:
						{
							D3D10_TEXTURE2D_DESC texDesc;
							((ID3D10Texture2D*)resource)->GetDesc(&texDesc);
							desc.Format = texDesc.Format;
						}
						break;
					case D3D10_SRV_DIMENSION_TEXTURE3D:
						{
							D3D10_TEXTURE3D_DESC texDesc;
							((ID3D10Texture3D*)resource)->GetDesc(&texDesc);
							desc.Format = texDesc.Format;
						}
						break;
					}
				}
				ID3D10ShaderResourceView* srv;
				HRESULT hr = device->CreateShaderResourceView(resource, &desc, &srv);
				resourceSet->addShaderResourceView(iSRVDesc->first, srv, swapChainBound);
			}
		}
		iSRVDesc++;
	}
}

void ResourceBuilder::defineVariables(ResourceSet* resourceSet, ID3D10Device* device)
{
	BufferDescDirectory::iterator iBufferDesc = bufferDescDirectory.begin();
	while(iBufferDesc != bufferDescDirectory.end())
	{
		resourceSet->createVariable(ScriptVariableClass::Resource, iBufferDesc->first);
		iBufferDesc++;
	}

	Texture2DDescDirectory::iterator iTexture2DDesc = texture2DDescDirectory.begin();
	while(iTexture2DDesc != texture2DDescDirectory.end())
	{
		resourceSet->createVariable(ScriptVariableClass::Resource, iTexture2DDesc->first);
		iTexture2DDesc++;
	}

	Texture1DDescDirectory::iterator iTexture1DDesc = texture1DDescDirectory.begin();
	while(iTexture1DDesc != texture1DDescDirectory.end())
	{
		resourceSet->createVariable(ScriptVariableClass::Resource, iTexture1DDesc->first);
		iTexture1DDesc++;
	}

	RTVDescDirectory::iterator iRTVDesc = rtvDescDirectory.begin();
	while(iRTVDesc != rtvDescDirectory.end())
	{
		resourceSet->createVariable(ScriptVariableClass::RenderTargetView, iRTVDesc->first);
		iRTVDesc++;
	}

	SRVDescDirectory::iterator iSRVDesc = srvDescDirectory.begin();
	while(iSRVDesc != srvDescDirectory.end())
	{
		resourceSet->createVariable(ScriptVariableClass::ShaderResourceView, iSRVDesc->first);
		iSRVDesc++;
	}
}

void ResourceBuilder::loadVariables(XMLNode& variablesNode)
{

}