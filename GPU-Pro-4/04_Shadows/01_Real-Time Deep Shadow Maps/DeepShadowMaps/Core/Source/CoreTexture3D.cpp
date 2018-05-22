#include "Core.h"
using namespace std;

CoreTexture3D::CoreTexture3D()
{
	width = 0;
	height = 0;
	depth = 0;
	texture = NULL;
	histogram = NULL;
	gradient = NULL;
}

// Load a 3D texture from memory
CoreResult CoreTexture3D::init(Core* core, BYTE* data, UINT width, UINT height, UINT depth, UINT mipLevels,
		  	    DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags)
{
	this->core = core;
	this->width = width;
	this->height = height;
	this->depth = depth;

	this->cpuAccessFlags = cpuAccessFlags;
	this->usage = usage;
	this->bindFlags = bindFlags;
	this->miscFlags = miscFlags;

	if(core)
	{
		D3D11_TEXTURE3D_DESC texDesc;
		ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE3D_DESC));
		texDesc.Height = height;
		texDesc.Width = width;
		texDesc.Depth = depth;
		texDesc.MipLevels = mipLevels;
		texDesc.Format = format;
		texDesc.Usage = usage;
		texDesc.BindFlags = bindFlags;
		texDesc.CPUAccessFlags = cpuAccessFlags;
		texDesc.MiscFlags = miscFlags;

		D3D11_SUBRESOURCE_DATA *subResData = NULL;
		if(data)
		{
			subResData = new D3D11_SUBRESOURCE_DATA;
			subResData->pSysMem = data;
			
			if(format == DXGI_FORMAT_R16G16B16A16_FLOAT)
				subResData->SysMemPitch = width * 8;
			else
				subResData->SysMemPitch = width * 2;
			subResData->SysMemSlicePitch = subResData->SysMemPitch * height;
		}

		
		HRESULT result = core->GetDevice()->CreateTexture3D(&texDesc, subResData, &texture);

		
		delete subResData;
		
		// create the texture
		if(FAILED(result))
		{
			CoreLog::Information(L"Couldn't create D3D Texture, HRESULT = %x!", result);
			return CORE_MISC_ERROR;
		}
		return result;
	}
	else
		texture = NULL;
	

	return CORE_OK;


}

// Load a texture from a stream
CoreResult CoreTexture3D::init(Core* core, std::istream& in, UINT mipLevels,
			  					UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags)
{
	BYTE* data = NULL;

	this->core = core;
	this->mipLevels = mipLevels;
	this->cpuAccessFlags = cpuAccessFlags;
	this->usage = usage;
	this->bindFlags = bindFlags;
	this->miscFlags = miscFlags;

	if(!in.good())
	{
		CoreLog::Information(L"in is not good, skipping!");
		return CORE_MISC_ERROR;
	}

	if(loadDat(in, &data) != CORE_OK)
	{
		CoreLog::Information(L"Couldn't load your 3D Texture (supported types are: *.dat)");
		return CORE_MISC_ERROR;
	}


	CoreResult result = createAndFillTexture(data);
	//if (result == CORE_OK)
	//	result = createAndFillGradient(data);
	

	delete data;

	return result;
}

// Loads a DAT from a stream
CoreResult CoreTexture3D::loadDat(std::istream& in, BYTE** data)
{
	unsigned short uWidth, uHeight, uDepth;

	in.read((char *)&uWidth, sizeof(unsigned short));
	in.read((char *)&uHeight, sizeof(unsigned short));
	in.read((char *)&uDepth, sizeof(unsigned short));

	width = (UINT)uWidth;
	height = (UINT)uHeight;
	depth = (UINT)uDepth;
	maxVal = max(width, max(height, depth));

	UINT padWidth = (maxVal - width) / 2;
	UINT padHeight = (maxVal - height) / 2;
	UINT padDepth = (maxVal - depth) / 2;

	format = DXGI_FORMAT_R32_FLOAT;

	*data = (BYTE *)new float[maxVal * maxVal * maxVal];
	
	unsigned short value;
	
	UINT slice = maxVal * maxVal;

	histogram = new DWORD[4096];		// VisLu doesn't use other values
	histSize = 4096;
	ZeroMemory(histogram, sizeof(DWORD) * 4096);
	maxHistValue = 0;

	float *fdata = (float *)data;

	for (UINT k = 0 ; k < maxVal ; k++)
		for (UINT j = 0 ; j < maxVal ; j++)
			for (UINT i = 0 ; i < maxVal ; i++)
			{
				// Save as DXGI_FORMAT_R32_FLOAT
				if(k >= padDepth && j >= padHeight && i >= padWidth && k - padDepth < depth && j - padHeight < height && i - padWidth < width)
				{
					in.read((char *)&value, sizeof(unsigned short));

					if(value < 4096)
					{
						histogram[value]++;
						if (histogram[value] > maxHistValue)
							maxHistValue = histogram[value];
					}
				}
				else
					value = 0;
			
				((float *)*data)[i + j * maxVal + k * slice] = value / 4096.0f;	// Max val == 4096 -> prescale for texture lookup

				
			}
	
	return CORE_OK;
}

// As the name says...
CoreResult CoreTexture3D::createAndFillTexture(BYTE* data)
{
	if(core)
	{
		D3D11_TEXTURE3D_DESC texDesc;
		ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE3D_DESC));
		texDesc.Height = maxVal;
		texDesc.Width = maxVal;
		texDesc.Depth = maxVal;
		texDesc.MipLevels = mipLevels;
		texDesc.Format = format;
		texDesc.Usage = usage;
		texDesc.BindFlags = bindFlags;
		texDesc.CPUAccessFlags = cpuAccessFlags;
		texDesc.MiscFlags = miscFlags;

		D3D11_SUBRESOURCE_DATA *subResData = NULL;
		if(data)
		{
			subResData = new D3D11_SUBRESOURCE_DATA;
			subResData->pSysMem = data;
			
			subResData->SysMemPitch = maxVal * sizeof(float);
			subResData->SysMemSlicePitch = subResData->SysMemPitch * maxVal;
		}

		
		HRESULT result = core->GetDevice()->CreateTexture3D(&texDesc, subResData, &texture);

		

		delete subResData;
		
		// create the texture
		if(FAILED(result))
		{
			CoreLog::Information(L"Couldn't create D3D Texture, HRESULT = %x!", result);
			return CORE_MISC_ERROR;
		}
		return result;
	}
	else
		texture = NULL;
	

	return CORE_OK;
}

// CleanUp
void CoreTexture3D::finalRelease()
{
	SAFE_RELEASE(texture);
	SAFE_RELEASE(gradient);
	SAFE_DELETE(histogram);
}

// Retrieves the RenderTargetView from the texture
CoreResult CoreTexture3D::CreateRenderTargetView(D3D11_RENDER_TARGET_VIEW_DESC* rtvDesc, ID3D11RenderTargetView** rtv)
{
	HRESULT result = core->GetDevice()->CreateRenderTargetView(texture, rtvDesc, rtv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create RenderTargetView, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}

// Retrieves the DepthStencilView from the texture
CoreResult CoreTexture3D::CreateDepthStencilView(D3D11_DEPTH_STENCIL_VIEW_DESC* dsvDesc, ID3D11DepthStencilView** dsv)
{
	HRESULT result = core->GetDevice()->CreateDepthStencilView(texture, dsvDesc, dsv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create DepthStencilView, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}


// Creates a ShaderResourceView with this texture as resource
CoreResult CoreTexture3D::CreateShaderResourceView(D3D11_SHADER_RESOURCE_VIEW_DESC* srvDesc, ID3D11ShaderResourceView** srv)
{
	HRESULT result = core->GetDevice()->CreateShaderResourceView(texture, srvDesc, srv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create DepthStencilView, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}

// Calculates the histogram for the 3D Texture, do not release the data
void CoreTexture3D::GetHistogram(DWORD **outHist, DWORD *outArrSize, DWORD *outMaxHistValue)
{
	*outHist = histogram;
	*outArrSize = histSize;
	*outMaxHistValue = maxHistValue;
}

CoreResult CoreTexture3D::createAndFillGradient(BYTE* data)
{	
	if(core)
	{
		D3D11_TEXTURE3D_DESC texDesc;
		BYTE *tempgrad = NULL;
		ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE3D_DESC));
		texDesc.Height = maxVal;
		texDesc.Width = maxVal;
		texDesc.Depth = maxVal;
		texDesc.MipLevels = 1;
		texDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
		texDesc.Usage = usage;
		texDesc.BindFlags = bindFlags;
		texDesc.CPUAccessFlags = cpuAccessFlags;
		texDesc.MiscFlags = miscFlags;

		D3D11_SUBRESOURCE_DATA *subResData = NULL;
		if(data)
		{						
			calculateGradient(data, &tempgrad);
			subResData = new D3D11_SUBRESOURCE_DATA;
			subResData->pSysMem = tempgrad;
			
			subResData->SysMemPitch = maxVal * 3 * sizeof(float);
			subResData->SysMemSlicePitch = subResData->SysMemPitch * maxVal;
		}

		
		HRESULT result = core->GetDevice()->CreateTexture3D(&texDesc, subResData, &gradient);

		delete subResData;
		delete tempgrad;

		// create the texture
		if(FAILED(result))
		{
			CoreLog::Information(L"Couldn't create D3D Texture, HRESULT = %x!", result);
			return CORE_MISC_ERROR;
		}
		return result;
	}
	else
		gradient = NULL;
	

	return CORE_OK;
}

void CoreTexture3D::calculateGradient (BYTE* data, BYTE **grad)
{
	*grad = (BYTE *)new float[maxVal * maxVal * maxVal * 3];
	float prevZSlice[9], nextZSlice[9], ZSlice[9];
	int prevPosX, nextPosX;
	int prevPosY, nextPosY;
	int prevPosZ, nextPosZ;
    CoreVector3 gradVector;
	int pos;
	UINT slice = maxVal * maxVal;

	for (int z = 0 ; z < (int)maxVal ; z++)
		for (int y = 0 ; y < (int)maxVal ; y++)
			for (int x = 0 ; x < (int)maxVal ; x++)
			{
				//Calculate positions of surrounding voxels.
				//At edge voxels the surrounding voxels are copies of the edge voxel (mirroring closing)
				prevPosX = ((x - 1) >= 0)?(x-1):0;
				prevPosY = ((y - 1) >= 0)?(y-1):0;
				prevPosZ = ((z - 1) >= 0)?(z-1):0;
				nextPosX = ((x + 1) < (int)maxVal)?(x+1):(maxVal-1);
				nextPosY = ((y + 1) < (int)maxVal)?(y+1):(maxVal-1);
				nextPosZ = ((z + 1) < (int)maxVal)?(z+1):(maxVal-1);
				
				//Get values from voxels around the current voxel 
				fillzSlice (prevZSlice, data, prevPosX, x, nextPosX, prevPosY, y, nextPosY, prevPosZ);
				fillzSlice (ZSlice,     data, prevPosX, x, nextPosX, prevPosY, y, nextPosY, z);
				fillzSlice (nextZSlice, data, prevPosX, x, nextPosX, prevPosY, y, nextPosY, nextPosZ);

				calculateGradientVector(prevZSlice, ZSlice, nextZSlice, gradVector);
				pos = (x + y * maxVal + z * slice) * 3;
				((float *)*grad)[pos] = gradVector.x;
				((float *)*grad)[pos+1] = gradVector.y;
				((float *)*grad)[pos+2] = gradVector.z;
			}
}

void CoreTexture3D::fillzSlice (float *zSlice, BYTE *data, int prevX, int x, int nextX, int prevY, int y, int nextY, int z)
{
	UINT slice = maxVal * maxVal;

	zSlice[0] = ((float *)data)[prevX + prevY * maxVal + z * slice];
	zSlice[1] = ((float *)data)[x +     prevY * maxVal + z * slice];
	zSlice[2] = ((float *)data)[nextX + prevY * maxVal + z * slice];
	zSlice[3] = ((float *)data)[prevX + y     * maxVal + z * slice];
	zSlice[4] = ((float *)data)[x +     y     * maxVal + z * slice];
	zSlice[5] = ((float *)data)[nextX + y     * maxVal + z * slice];
	zSlice[6] = ((float *)data)[prevX + nextY * maxVal + z * slice];
	zSlice[7] = ((float *)data)[x +     nextY * maxVal + z * slice];
	zSlice[8] = ((float *)data)[nextX + nextY * maxVal + z * slice];
}

void CoreTexture3D::calculateGradientVector(float * samplesNegZOffset, float * samplesNoZOffset, float * samplesPosZOffset, CoreVector3 &gradVector)
{
	// Code taken from Stefan Bruckners Computational Aesthetics Framework
	float voxelWeights[3][3][3]= {  
		{ {2.0f,3.0f,2.0f}, {3.0f,6.0f,3.0f}, {2.0f,3.0f,2.0f}  },
		{ {3.0f,6.0f,3.0f},  {6.0f,0.0f,6.0f},  {3.0f,6.0f,3.0f} },
		{ {2.0f,3.0f,2.0f},  {3.0f,6.0f,3.0f},  {2.0f,3.0f,2.0f} } };

		float gradientXWeight = -1.0f / 
			(voxelWeights[0][0][0] + voxelWeights[1][0][0] + voxelWeights[2][0][0]
		+ voxelWeights[0][1][0] + voxelWeights[1][1][0] + voxelWeights[2][1][0]
		+ voxelWeights[0][2][0] + voxelWeights[1][2][0] + voxelWeights[2][2][0]
		+ voxelWeights[0][0][2] + voxelWeights[1][0][2] + voxelWeights[2][0][2]
		+ voxelWeights[0][1][2] + voxelWeights[1][1][2] + voxelWeights[2][1][2]
		+ voxelWeights[0][2][2] + voxelWeights[1][2][2] + voxelWeights[2][2][2]); 

		float gradientYWeight = -1.0f / 
			(voxelWeights[0][0][0] + voxelWeights[1][0][0] + voxelWeights[2][0][0]
		+ voxelWeights[0][2][0] + voxelWeights[1][2][0] + voxelWeights[2][2][0]
		+ voxelWeights[0][0][1] + voxelWeights[1][0][1] + voxelWeights[2][0][1]
		+ voxelWeights[0][2][1] + voxelWeights[1][2][1] + voxelWeights[2][2][1]
		+ voxelWeights[0][0][2] + voxelWeights[1][0][2] + voxelWeights[2][0][2]
		+ voxelWeights[0][2][2] + voxelWeights[1][2][2] + voxelWeights[2][2][2]);

		float gradientZWeight = -1.0f / 
			(voxelWeights[0][0][0] + voxelWeights[2][0][0] + voxelWeights[0][1][0]
		+ voxelWeights[2][1][0] + voxelWeights[0][2][0] + voxelWeights[2][2][0]
		+ voxelWeights[0][0][1] + voxelWeights[2][0][1] + voxelWeights[0][1][1]
		+ voxelWeights[2][1][1] + voxelWeights[0][2][1] + voxelWeights[2][2][1]
		+ voxelWeights[0][0][2] + voxelWeights[2][0][2] + voxelWeights[0][1][2]
		+ voxelWeights[2][1][2] + voxelWeights[0][2][2] + voxelWeights[2][2][2]);


		float fGx(gradientXWeight * 
			(- voxelWeights[0][0][0] * samplesNegZOffset[0] 
		- voxelWeights[1][0][0] * samplesNoZOffset[0] 
		- voxelWeights[2][0][0] * samplesPosZOffset[0] 
		- voxelWeights[0][1][0] * samplesNegZOffset[3]
		- voxelWeights[1][1][0] * samplesNoZOffset[3] 
		- voxelWeights[2][1][0] * samplesPosZOffset[3]
		- voxelWeights[0][2][0] * samplesNegZOffset[6] 
		- voxelWeights[1][2][0] * samplesNoZOffset[6] 
		- voxelWeights[2][2][0] * samplesPosZOffset[6]
		+ voxelWeights[0][0][2] * samplesNegZOffset[2] 
		+ voxelWeights[1][0][2] * samplesNoZOffset[2] 
		+ voxelWeights[2][0][2] * samplesPosZOffset[2]
		+ voxelWeights[0][1][2] * samplesNegZOffset[3] 
		+ voxelWeights[1][1][2] * samplesNoZOffset[5] 
		+ voxelWeights[2][1][2] * samplesPosZOffset[5]
		+ voxelWeights[0][2][2] * samplesNegZOffset[8] 
		+ voxelWeights[1][2][2] * samplesNoZOffset[8] 
		+ voxelWeights[2][2][2] * samplesPosZOffset[8]));

		float fGy(gradientYWeight * 
			(- voxelWeights[0][0][0] * samplesNegZOffset[0]
		- voxelWeights[1][0][0] * samplesNoZOffset[0] 
		- voxelWeights[2][0][0] * samplesPosZOffset[0] 
		+ voxelWeights[0][2][0] * samplesNegZOffset[6]
		+ voxelWeights[1][2][0] * samplesNoZOffset[6] 
		+ voxelWeights[2][2][0] * samplesPosZOffset[6] 
		- voxelWeights[0][0][1] * samplesNegZOffset[1] 
		- voxelWeights[1][0][1] * samplesNoZOffset[1] 
		- voxelWeights[2][0][1] * samplesPosZOffset[1] 
		+ voxelWeights[0][2][1] * samplesNegZOffset[7]
		+ voxelWeights[1][2][1] * samplesNoZOffset[7] 
		+ voxelWeights[2][2][1] * samplesPosZOffset[7] 
		- voxelWeights[0][0][2] * samplesNegZOffset[2] 
		- voxelWeights[1][0][2] * samplesNoZOffset[2] 
		- voxelWeights[2][0][2] * samplesPosZOffset[2] 
		+ voxelWeights[0][2][2] * samplesNegZOffset[8]
		+ voxelWeights[1][2][2] * samplesNoZOffset[8] 
		+ voxelWeights[2][2][2] * samplesPosZOffset[8]));


		float fGz(gradientZWeight * 
			(- voxelWeights[0][0][0] * samplesNegZOffset[0]
		+ voxelWeights[2][0][0] * samplesPosZOffset[0]
		- voxelWeights[0][1][0] * samplesNegZOffset[3] 
		+ voxelWeights[2][1][0] * samplesPosZOffset[3]
		- voxelWeights[0][2][0] * samplesNegZOffset[6]
		+ voxelWeights[2][2][0] * samplesPosZOffset[6] 
		- voxelWeights[0][0][1] * samplesNegZOffset[1]
		+ voxelWeights[2][0][1] * samplesPosZOffset[1]
		- voxelWeights[0][1][1] * samplesNegZOffset[4] 
		+ voxelWeights[2][1][1] * samplesPosZOffset[4] 
		- voxelWeights[0][2][1] * samplesNegZOffset[7]
		+ voxelWeights[2][2][1] * samplesPosZOffset[7] 
		- voxelWeights[0][0][2] * samplesNegZOffset[2]
		+ voxelWeights[2][0][2] * samplesPosZOffset[2]
		- voxelWeights[0][1][2] * samplesNegZOffset[3] 
		+ voxelWeights[2][1][2] * samplesPosZOffset[5]
		- voxelWeights[0][2][2] * samplesNegZOffset[8] 
		+ voxelWeights[2][2][2] * samplesPosZOffset[8]));

		gradVector.x = fGx;
		gradVector.y = fGy;
		gradVector.z = fGz;
}

// Creates a ShaderResourceView with the gradient texture as resource
CoreResult CoreTexture3D::CreateShaderResourceViewGradient(D3D11_SHADER_RESOURCE_VIEW_DESC* srvDesc, ID3D11ShaderResourceView** srv)
{
	HRESULT result = core->GetDevice()->CreateShaderResourceView(gradient, srvDesc, srv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create DepthStencilView, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}