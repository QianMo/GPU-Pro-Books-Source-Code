#pragma once
#include "sas/DxaSas.h"
#include "sas/SasButton.h"
#include "DXUTCamera.h"
#include "Mesh/Binder.h"
#include "Mesh/Cast.h"
#include "Particle.h"

namespace Dxa
{
	/// Application class realizing volumetric transparency demos. (Both linked list objects and ray-trace particles.)
	class VolTransRaster :
		public Dxa::Sas
	{
		CFirstPersonCamera camera;
		Mesh::Binder* binder;

		/// particles / linked list demo toggler
		bool displayParticles;
		/// animation toggler
		bool freeze;
		/// particle distance importor toggler
		bool drawSpheresOnly;

		/// Rendering roles corresponding to passes
		// for linked list fragments demo
		Mesh::Role deferRole;
		Mesh::Role storeRole;
		Mesh::Role sortRole;
		// for the particle demo
		Mesh::Role smokeRole;
		Mesh::Role sphereRole;

		Mesh::Cast::P loadMeshTransparent(const char* geometryFilename);
		Mesh::Cast::P loadMeshOpaque(const char* geometryFilename, bool isShadowCaster);

		/// geometry for full viewport quad
		Mesh::Cast::P quadMesh;
		/// geometry for instanced small quads covering the viewport
		Mesh::Cast::P tileSet;

		/// Draws all entities.
		void drawAll(ID3D11DeviceContext* context, Mesh::Role role);

		struct Entity
		{
			Mesh::Cast::P mesh;
			D3DXMATRIX modelMatrix;
			unsigned int transparentMaterialIndex;
		};
		/// Virtual world entities.
		std::vector<Entity> entities;

		/// Textures.
		typedef std::map<std::string, ID3D11ShaderResourceView*> SrvDirectory;
		SrvDirectory srvs;
		ID3D11ShaderResourceView* envTextureSrv;
		ID3D11ShaderResourceView* puffTextureSrv;

		/// R/W buffers.
		ID3D11Buffer* fragmentLinkBuffer;
		ID3D11Buffer* startOffsetBuffer;

		// Unordered Access views of the buffers
		ID3D11UnorderedAccessView*  fragmentLinkUav;
		ID3D11UnorderedAccessView*  startOffsetUav;         

		// Shader Resource Views
		ID3D11ShaderResourceView*   fragmentLinkSrv;
		ID3D11ShaderResourceView*   startOffsetSrv;

		// RT textures
		ID3D11Texture2D*			opaqueTexture;
		ID3D11RenderTargetView*		opaqueRtv;
		ID3D11ShaderResourceView*	opaqueSrv;

		ID3D11Texture2D*			nearPlaneTexture;
		ID3D11RenderTargetView*		nearPlaneRtv;
		ID3D11ShaderResourceView*	nearPlaneSrv;

		ID3D11Texture2D*			nearPlaneIrradianceTexture;
		ID3D11RenderTargetView*		nearPlaneIrradianceRtv;
		ID3D11ShaderResourceView*	nearPlaneIrradianceSrv;

		// ray-cast particle system rsources
		ID3D11Texture2D* particleTexture;
		ID3D11ShaderResourceView* particleSrv;
		ID3D11Buffer* tileParticleCountBuffer;

		Particle particles[512];
		unsigned int nParticles;

		void loadEffect();

		ID3D11ShaderResourceView* loadTexture(const char* filename);

		/// Adds entities from a file with asset importer.
		void importScene(const char* sceneFileName);

		/// Adds a single entity from an obj file with asset importer.
		Mesh::Geometry::P importObject(const char* objectFileName);

	public:
		VolTransRaster(ID3D11Device* device);
		HRESULT createResources();
		HRESULT releaseResources();
		HRESULT createSwapChainResources();
		HRESULT releaseSwapChainResources();
		void animate(double dt, double t);
		bool processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
		void render(ID3D11DeviceContext* context);
	};

}