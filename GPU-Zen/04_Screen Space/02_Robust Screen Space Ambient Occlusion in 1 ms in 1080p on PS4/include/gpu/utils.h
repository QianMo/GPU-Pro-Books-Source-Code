#pragma once


#include "types.h"
#include <essentials/main.h>
#include <math/types.h>
#include <image/types.h>

#include <d3d11_1.h>


using namespace NEssentials;
using namespace NImage;


namespace NGPU
{
	namespace NUtils
	{
		struct Vertex;
		class Texture;
		class Mesh;
		class Scene;
		class Utils;


		struct Vertex
		{
			NMath::Vector3 position;
			NMath::Vector3 normal;
			NMath::Vector2 uv;
		};


		class Texture
		{
		public:
			bool CreateFromFile(const string& path);
			void Destroy();

			bool CreateCPUDataFromFile(const string& path);
			bool CreateCPUDataFromCustomFile(const string& path);
			void CreateGPUData();
			void DestroyCPUData();

			bool SaveCPUDataToCustomFile(const string& path);

		public: // readonly
			vector<Mipmap> mipmaps;
			NGPU::Texture texture;
		};


		class Mesh
		{
		public:
			void CreatePlane(float width, float height);
			void Destroy();

		public: // readonly
			uint32 verticesCount;
			uint32 indicesCount;

			Vertex* vertices;
			uint16* indices;

			ID3D11Buffer* vb;
			ID3D11Buffer* ib;

			string textureFileName;
		};


		class Scene
		{
		public:
			bool CreateFromFile(const string& path, map<string, Texture>* textures = nullptr);
			void Destroy();

			bool CreateCPUBuffersFromOBJ(const string& path);
			bool CreateCPUBuffersFromCustomFile(const string& path);
			void CreateGPUBuffers();
			void DestroyCPUBuffers();

			bool SaveCPUBuffersToCustomFile(const string& path);

			void UpdateTextures(map<string, Texture>& textures);

		public: // readonly
			string path;
			vector<Mesh> meshes;
		};


		class Utils
		{
		public:
			void Create(const string& frameworkPath);
			void Destroy();

			void CopyTexture(const RenderTarget& renderTarget, const NGPU::Texture& texture);
			void MinX(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const NMath::Vector2& pixelSize, int32 from, int32 to);
			void MinY(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const NMath::Vector2& pixelSize, int32 from, int32 to);
			void MaxX(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const NMath::Vector2& pixelSize, int32 from, int32 to);
			void MaxY(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const NMath::Vector2& pixelSize, int32 from, int32 to);
			void BlurX(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const NMath::Vector2& pixelSize, int32 from, int32 to);
			void BlurY(const RenderTarget& renderTarget, const NGPU::Texture& texture, byte channelsCount, const NMath::Vector2& pixelSize, int32 from, int32 to);

		public: // readonly
			// buffers

			ID3D11Buffer* screenQuadIB;

			ID3D11Buffer* oneVectorCB;
			ID3D11Buffer* twoVectorsCB;
			ID3D11Buffer* threeVectorsCB;
			ID3D11Buffer* fourVectorsCB;
			ID3D11Buffer* fiveVectorsCB;
			ID3D11Buffer* sixVectorsCB;

			// shaders

			ID3D11VertexShader* screenQuadVS;

			ID3D11PixelShader* copyTexturePS;

			ID3D11PixelShader* minXPS[4];
			ID3D11PixelShader* minYPS[4];
			ID3D11PixelShader* minX13PS[4];
			ID3D11PixelShader* minY13PS[4];
			ID3D11PixelShader* maxXPS[4];
			ID3D11PixelShader* maxYPS[4];
			ID3D11PixelShader* maxX13PS[4];
			ID3D11PixelShader* maxY13PS[4];
			ID3D11PixelShader* blurXPS[4];
			ID3D11PixelShader* blurYPS[4];
			ID3D11PixelShader* blurX13PS[4];
			ID3D11PixelShader* blurY13PS[4];
		};
		extern Utils gpuUtils;
	}
}
