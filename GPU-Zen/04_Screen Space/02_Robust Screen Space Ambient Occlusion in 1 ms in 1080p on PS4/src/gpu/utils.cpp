#include <gpu/utils.h>
#include <gpu/d3d11.h>
#include <system/file.h>
#include <math/main.h>
#include <mesh/main.h>
#include <image/main.h>


using namespace NSystem;
using namespace NMath;
using namespace NImage;


//


NGPU::NUtils::Utils NGPU::NUtils::gpuUtils;


// Texture


bool NGPU::NUtils::Texture::CreateFromFile(const string& path)
{
	string cachedFilePath = path + ".cache";

	if (FileExists(cachedFilePath))
	{
		if (!CreateCPUDataFromCustomFile(cachedFilePath))
			return false;
	}
	else
	{
		if (!CreateCPUDataFromFile(path))
			return false;
		if (!SaveCPUDataToCustomFile(cachedFilePath))
			return false;
	}

	CreateGPUData();
	DestroyCPUData();

	return true;
}


void NGPU::NUtils::Texture::Destroy()
{
	DestroyTexture(texture);
}


bool NGPU::NUtils::Texture::CreateCPUDataFromFile(const string& path)
{
	uint8* imageData;
	uint16 imageWidth, imageHeight;
	NImage::Format imageFormat;

	if (!Load(path, imageData, imageWidth, imageHeight, imageFormat))
		return false;

	if (imageFormat != Format::RGBA8)
	{
		uint8* prevImageData = imageData;
		imageData = Convert(prevImageData, imageWidth, imageHeight, imageFormat, Format::RGBA8);
		imageFormat = Format::RGBA8;
		delete[] prevImageData;
	}

	mipmaps = GenerateMipmaps(imageData, imageWidth, imageHeight, imageFormat, NImage::Filter::Box);
	delete[] imageData;

	return true;
}


bool NGPU::NUtils::Texture::CreateCPUDataFromCustomFile(const string& path)
{
	File file;

	if (file.Open(path, File::OpenMode::ReadBinary))
	{
		uint32 mipmapsCount;

		file.ReadBin((char*)&mipmapsCount, sizeof(uint32));
		mipmaps.resize(mipmapsCount);

		for (uint i = 0; i < mipmapsCount; i++)
		{
			file.ReadBin((char*)&mipmaps[i].width, sizeof(uint16));
			file.ReadBin((char*)&mipmaps[i].height, sizeof(uint16));

			mipmaps[i].data = new uint8[4 * mipmaps[i].width * mipmaps[i].height];

			file.ReadBin((char*)mipmaps[i].data, 4 * mipmaps[i].width * mipmaps[i].height);
		}

		file.Close();

		return true;
	}
	else
	{
		return false;
	}
}


void NGPU::NUtils::Texture::CreateGPUData()
{
	CreateTexture(mipmaps[0].width, mipmaps[0].height, texture);

	for (uint i = 0; i < mipmaps.size(); i++)
		UpdateTexture(texture, i, mipmaps[i].data, 4 * mipmaps[i].width);
}


void NGPU::NUtils::Texture::DestroyCPUData()
{
	for (uint i = 0; i < mipmaps.size(); i++)
		delete[] mipmaps[i].data;
	
	mipmaps.clear();
}


bool NGPU::NUtils::Texture::SaveCPUDataToCustomFile(const string& path)
{
	File file;

	if (file.Open(path, File::OpenMode::WriteBinary))
	{
		uint32 mipmapsCount = (uint32)mipmaps.size();

		file.WriteBin((char*)&mipmapsCount, sizeof(uint32));
		for (uint i = 0; i < mipmapsCount; i++)
		{
			file.WriteBin((char*)&mipmaps[i].width, sizeof(uint16));
			file.WriteBin((char*)&mipmaps[i].height, sizeof(uint16));
			file.WriteBin((char*)mipmaps[i].data, 4 * mipmaps[i].width * mipmaps[i].height);
		}

		file.Close();

		return true;
	}
	else
	{
		return false;
	}
}


// Mesh


void NGPU::NUtils::Mesh::CreatePlane(float width, float height)
{
	verticesCount = 4;
	indicesCount = 6;

	Vertex vertices[] =
	{
		VectorCustom(-0.5f*width, 0.0f, -0.5f*height), VectorCustom(0.0f, 1.0f, 0.0f), VectorCustom(0.0f, 0.0f),
		VectorCustom(-0.5f*width, 0.0f,  0.5f*height), VectorCustom(0.0f, 1.0f, 0.0f), VectorCustom(0.0f, 1.0f),
		VectorCustom( 0.5f*width, 0.0f,  0.5f*height), VectorCustom(0.0f, 1.0f, 0.0f), VectorCustom(1.0f, 1.0f),
		VectorCustom( 0.5f*width, 0.0f, -0.5f*height), VectorCustom(0.0f, 1.0f, 0.0f), VectorCustom(1.0f, 0.0f),
	};

	uint16 indices[] = { 0, 1, 2, 0, 2, 3 };

	CreateVertexBuffer((uint8*)vertices, verticesCount * sizeof(Vertex), vb);
	CreateIndexBuffer((uint8*)indices, indicesCount * sizeof(uint16), ib);
}


void NGPU::NUtils::Mesh::Destroy()
{
	SAFE_RELEASE(vb);
	SAFE_RELEASE(ib);
}


// Scene


bool NGPU::NUtils::Scene::CreateFromFile(const string& path, map<string, Texture>* textures)
{
	string cachedFilePath = path + ".cache";

	if (FileExists(cachedFilePath))
	{
		this->path = path;

		if (!CreateCPUBuffersFromCustomFile(cachedFilePath))
			return false;
	}
	else
	{
		if (!CreateCPUBuffersFromOBJ(path))
			return false;
		if (!SaveCPUBuffersToCustomFile(cachedFilePath))
			return false;
	}

	CreateGPUBuffers();
	DestroyCPUBuffers();

	if (textures)
		UpdateTextures(*textures);

	return true;
}


void NGPU::NUtils::Scene::Destroy()
{
	for (uint i = 0; i < meshes.size(); i++)
		meshes[i].Destroy();
}


bool NGPU::NUtils::Scene::CreateCPUBuffersFromOBJ(const string& path)
{
	vector<NMesh::Mesh> meshes;
	vector<NMesh::Material> materials;

	if (!ImportOBJ(path, meshes, materials))
		return false;

	this->path = path;

	for (uint i = 0; i < meshes.size(); i++)
	{
		ToIndexed(meshes[i]);

		Mesh mesh;

		mesh.verticesCount = meshes[i].vertices.size();
		mesh.indicesCount = meshes[i].indices.size();
		mesh.vertices = new Vertex[mesh.verticesCount];
		mesh.indices = new uint16[mesh.indicesCount];

		for (uint j = 0; j < mesh.verticesCount; j++)
		{
			mesh.vertices[j].position = meshes[i].vertices[j].position;
			mesh.vertices[j].normal = meshes[i].vertices[j].normal;
			mesh.vertices[j].uv = meshes[i].vertices[j].texCoord;
		}

		for (uint j = 0; j < mesh.indicesCount; j++)
		{
			mesh.indices[j] = meshes[i].indices[j];
		}

		if (meshes[i].materialIndex >= 0)
			mesh.textureFileName = materials[meshes[i].materialIndex].textureFileName;
		else
			mesh.textureFileName = "";

		this->meshes.push_back(mesh);
	}

	return true;
}


bool NGPU::NUtils::Scene::CreateCPUBuffersFromCustomFile(const string& path)
{
	File file;

	if (file.Open(path, File::OpenMode::ReadBinary))
	{
		this->path = path;

		uint32 meshesCount;

		file.ReadBin((char*)&meshesCount, sizeof(uint32));
		meshes.resize(meshesCount);

		for (uint i = 0; i < meshesCount; i++)
		{
			file.ReadBin((char*)&meshes[i].verticesCount, sizeof(uint32));
			file.ReadBin((char*)&meshes[i].indicesCount, sizeof(uint32));

			meshes[i].vertices = new Vertex[meshes[i].verticesCount];
			meshes[i].indices = new uint16[meshes[i].indicesCount];

			file.ReadBin((char*)meshes[i].vertices, meshes[i].verticesCount * sizeof(Vertex));
			file.ReadBin((char*)meshes[i].indices, meshes[i].indicesCount * sizeof(uint16));

			file.ReadBin(meshes[i].textureFileName);
		}

		file.Close();

		return true;
	}
	else
	{
		return false;
	}
}


void NGPU::NUtils::Scene::CreateGPUBuffers()
{
	for (uint i = 0; i < meshes.size(); i++)
	{
		CreateVertexBuffer((uint8*)meshes[i].vertices, meshes[i].verticesCount * sizeof(Vertex), meshes[i].vb);
		CreateIndexBuffer((uint8*)meshes[i].indices, meshes[i].indicesCount * sizeof(uint16), meshes[i].ib);
	}
}


void NGPU::NUtils::Scene::DestroyCPUBuffers()
{
	for (uint i = 0; i < meshes.size(); i++)
	{
		delete[] meshes[i].vertices;
		delete[] meshes[i].indices;
	}
}


bool NGPU::NUtils::Scene::SaveCPUBuffersToCustomFile(const string& path)
{
	File file;

	if (file.Open(path, File::OpenMode::WriteBinary))
	{
		uint32 meshesCount = meshes.size();

		file.WriteBin((char*)&meshesCount, sizeof(uint32));
		for (uint i = 0; i < meshesCount; i++)
		{
			file.WriteBin((char*)&meshes[i].verticesCount, sizeof(uint32));
			file.WriteBin((char*)&meshes[i].indicesCount, sizeof(uint32));

			file.WriteBin((char*)meshes[i].vertices, meshes[i].verticesCount * sizeof(Vertex));
			file.WriteBin((char*)meshes[i].indices, meshes[i].indicesCount * sizeof(uint16));

			file.WriteBin(meshes[i].textureFileName);
		}

		file.Close();

		return true;
	}
	else
	{
		return false;
	}
}


void NGPU::NUtils::Scene::UpdateTextures(map<string, Texture>& textures)
{
	for (uint i = 0; i < meshes.size(); i++)
	{
		string& key = meshes[i].textureFileName;

		if (textures.find(key) == textures.end())
		{
			Texture texture;

			if (texture.CreateFromFile(ExtractDir(path) + meshes[i].textureFileName))
				textures[key] = texture;
		}
	}
}


// Utils


void NGPU::NUtils::Utils::Create(const string& frameworkPath)
{
	uint16 indices[] = { 0, 1, 2, 0, 2, 3 };
	CreateIndexBuffer((uint8*)indices, 6 * sizeof(uint16), screenQuadIB);

	CreateConstantBuffer(1 * sizeof(Vector4), oneVectorCB);
	CreateConstantBuffer(2 * sizeof(Vector4), twoVectorsCB);
	CreateConstantBuffer(3 * sizeof(Vector4), threeVectorsCB);
	CreateConstantBuffer(4 * sizeof(Vector4), fourVectorsCB);
	CreateConstantBuffer(5 * sizeof(Vector4), fiveVectorsCB);
	CreateConstantBuffer(6 * sizeof(Vector4), sixVectorsCB);

	//

	ASSERT_FUNCTION(CreateVertexShader(frameworkPath + "/data/gpu/screen_quad_vs.hlsl", "", screenQuadVS));

	ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "COPY", copyTexturePS));

	for (int i = 0; i < 4; i++)
	{
		string channelsCount = ToString(i + 1);

		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "MIN|HORIZONTAL|CHANNELS_COUNT=" + channelsCount, minXPS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "MIN|VERTICAL|CHANNELS_COUNT=" + channelsCount, minYPS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "MIN13|HORIZONTAL|CHANNELS_COUNT=" + channelsCount, minX13PS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "MIN13|VERTICAL|CHANNELS_COUNT=" + channelsCount, minY13PS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "MAX|HORIZONTAL|CHANNELS_COUNT=" + channelsCount, maxXPS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "MAX|VERTICAL|CHANNELS_COUNT=" + channelsCount, maxYPS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "MAX13|HORIZONTAL|CHANNELS_COUNT=" + channelsCount, maxX13PS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "MAX13|VERTICAL|CHANNELS_COUNT=" + channelsCount, maxY13PS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "BLUR|HORIZONTAL|CHANNELS_COUNT=" + channelsCount, blurXPS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "BLUR|VERTICAL|CHANNELS_COUNT=" + channelsCount, blurYPS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "BLUR13|HORIZONTAL|CHANNELS_COUNT=" + channelsCount, blurX13PS[i]));
		ASSERT_FUNCTION(CreatePixelShader(frameworkPath + "/data/gpu/postprocess_ps.hlsl", "BLUR13|VERTICAL|CHANNELS_COUNT=" + channelsCount, blurY13PS[i]));
	}
}


void NGPU::NUtils::Utils::Destroy()
{
	for (int i = 0; i < 4; i++)
	{
		DestroyPixelShader(minXPS[i]);
		DestroyPixelShader(minYPS[i]);
		DestroyPixelShader(minX13PS[i]);
		DestroyPixelShader(minY13PS[i]);
		DestroyPixelShader(maxXPS[i]);
		DestroyPixelShader(maxYPS[i]);
		DestroyPixelShader(maxX13PS[i]);
		DestroyPixelShader(maxY13PS[i]);
		DestroyPixelShader(blurXPS[i]);
		DestroyPixelShader(blurYPS[i]);
		DestroyPixelShader(blurX13PS[i]);
		DestroyPixelShader(blurY13PS[i]);
	}

	DestroyPixelShader(copyTexturePS);

	DestroyVertexShader(screenQuadVS);

	//

	DestroyBuffer(oneVectorCB);
	DestroyBuffer(twoVectorsCB);
	DestroyBuffer(threeVectorsCB);
	DestroyBuffer(fourVectorsCB);
	DestroyBuffer(fiveVectorsCB);
	DestroyBuffer(sixVectorsCB);

	DestroyBuffer(screenQuadIB);
}


void NGPU::NUtils::Utils::CopyTexture(const RenderTarget& renderTarget, const NGPU::Texture& texture)
{
	deviceContext->OMSetRenderTargets(1, &renderTarget.rtv, nullptr);

	deviceContext->VSSetShader(screenQuadVS, nullptr, 0);
	deviceContext->PSSetShader(copyTexturePS, nullptr, 0);
	deviceContext->PSSetShaderResources(0, 1, &texture.srv);

	deviceContext->IASetIndexBuffer(screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

	deviceContext->DrawIndexed(6, 0, 0);
}


void NGPU::NUtils::Utils::MinX(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const Vector2& pixelSize, int32 from, int32 to)
{
	deviceContext->OMSetRenderTargets(1, &renderTarget.rtv, nullptr);

	deviceContext->VSSetShader(screenQuadVS, nullptr, 0);
	if (from == -6 && to == 6)
		deviceContext->PSSetShader(minX13PS[channelsCount-1], nullptr, 0);
	else
		deviceContext->PSSetShader(minXPS[channelsCount-1], nullptr, 0);
	deviceContext->PSSetShaderResources(0, 1, &texture.srv);

	deviceContext->IASetIndexBuffer(screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

	uint8 data[16];
	memcpy(&data[0], &pixelSize, 8);
	memcpy(&data[8], &from, 4);
	memcpy(&data[12], &to, 4);
	deviceContext->UpdateSubresource(oneVectorCB, 0, nullptr, data, 0, 0);
	deviceContext->PSSetConstantBuffers(0, 1, &oneVectorCB);

	deviceContext->DrawIndexed(6, 0, 0);
}


void NGPU::NUtils::Utils::MinY(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const Vector2& pixelSize, int32 from, int32 to)
{
	deviceContext->OMSetRenderTargets(1, &renderTarget.rtv, nullptr);

	deviceContext->VSSetShader(screenQuadVS, nullptr, 0);
	if (from == -6 && to == 6)
		deviceContext->PSSetShader(minY13PS[channelsCount-1], nullptr, 0);
	else
		deviceContext->PSSetShader(minYPS[channelsCount-1], nullptr, 0);
	deviceContext->PSSetShaderResources(0, 1, &texture.srv);

	deviceContext->IASetIndexBuffer(screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

	uint8 data[16];
	memcpy(&data[0], &pixelSize, 8);
	memcpy(&data[8], &from, 4);
	memcpy(&data[12], &to, 4);
	deviceContext->UpdateSubresource(oneVectorCB, 0, nullptr, data, 0, 0);
	deviceContext->PSSetConstantBuffers(0, 1, &oneVectorCB);

	deviceContext->DrawIndexed(6, 0, 0);
}


void NGPU::NUtils::Utils::MaxX(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const Vector2& pixelSize, int32 from, int32 to)
{
	deviceContext->OMSetRenderTargets(1, &renderTarget.rtv, nullptr);

	deviceContext->VSSetShader(screenQuadVS, nullptr, 0);
	if (from == -6 && to == 6)
		deviceContext->PSSetShader(maxX13PS[channelsCount-1], nullptr, 0);
	else
		deviceContext->PSSetShader(maxXPS[channelsCount-1], nullptr, 0);
	deviceContext->PSSetShaderResources(0, 1, &texture.srv);

	deviceContext->IASetIndexBuffer(screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

	uint8 data[16];
	memcpy(&data[0], &pixelSize, 8);
	memcpy(&data[8], &from, 4);
	memcpy(&data[12], &to, 4);
	deviceContext->UpdateSubresource(oneVectorCB, 0, nullptr, data, 0, 0);
	deviceContext->PSSetConstantBuffers(0, 1, &oneVectorCB);

	deviceContext->DrawIndexed(6, 0, 0);
}


void NGPU::NUtils::Utils::MaxY(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const Vector2& pixelSize, int32 from, int32 to)
{
	deviceContext->OMSetRenderTargets(1, &renderTarget.rtv, nullptr);

	deviceContext->VSSetShader(screenQuadVS, nullptr, 0);
	if (from == -6 && to == 6)
		deviceContext->PSSetShader(maxY13PS[channelsCount-1], nullptr, 0);
	else
		deviceContext->PSSetShader(maxYPS[channelsCount-1], nullptr, 0);
	deviceContext->PSSetShaderResources(0, 1, &texture.srv);

	deviceContext->IASetIndexBuffer(screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

	uint8 data[16];
	memcpy(&data[0], &pixelSize, 8);
	memcpy(&data[8], &from, 4);
	memcpy(&data[12], &to, 4);
	deviceContext->UpdateSubresource(oneVectorCB, 0, nullptr, data, 0, 0);
	deviceContext->PSSetConstantBuffers(0, 1, &oneVectorCB);

	deviceContext->DrawIndexed(6, 0, 0);
}


void NGPU::NUtils::Utils::BlurX(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const Vector2& pixelSize, int32 from, int32 to)
{
	deviceContext->OMSetRenderTargets(1, &renderTarget.rtv, nullptr);

	deviceContext->VSSetShader(screenQuadVS, nullptr, 0);
	if (from == -6 && to == 6)
		deviceContext->PSSetShader(blurX13PS[channelsCount-1], nullptr, 0);
	else
		deviceContext->PSSetShader(blurXPS[channelsCount-1], nullptr, 0);
	deviceContext->PSSetShaderResources(0, 1, &texture.srv);

	deviceContext->IASetIndexBuffer(screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

	uint8 data[16];
	memcpy(&data[0], &pixelSize, 8);
	memcpy(&data[8], &from, 4);
	memcpy(&data[12], &to, 4);
	deviceContext->UpdateSubresource(oneVectorCB, 0, nullptr, data, 0, 0);
	deviceContext->PSSetConstantBuffers(0, 1, &oneVectorCB);

	deviceContext->DrawIndexed(6, 0, 0);
}


void NGPU::NUtils::Utils::BlurY(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const Vector2& pixelSize, int32 from, int32 to)
{
	deviceContext->OMSetRenderTargets(1, &renderTarget.rtv, nullptr);

	deviceContext->VSSetShader(screenQuadVS, nullptr, 0);
	if (from == -6 && to == 6)
		deviceContext->PSSetShader(blurY13PS[channelsCount-1], nullptr, 0);
	else
		deviceContext->PSSetShader(blurYPS[channelsCount-1], nullptr, 0);
	deviceContext->PSSetShaderResources(0, 1, &texture.srv);

	deviceContext->IASetIndexBuffer(screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

	uint8 data[16];
	memcpy(&data[0], &pixelSize, 8);
	memcpy(&data[8], &from, 4);
	memcpy(&data[12], &to, 4);
	deviceContext->UpdateSubresource(oneVectorCB, 0, nullptr, data, 0, 0);
	deviceContext->PSSetConstantBuffers(0, 1, &oneVectorCB);

	deviceContext->DrawIndexed(6, 0, 0);
}
