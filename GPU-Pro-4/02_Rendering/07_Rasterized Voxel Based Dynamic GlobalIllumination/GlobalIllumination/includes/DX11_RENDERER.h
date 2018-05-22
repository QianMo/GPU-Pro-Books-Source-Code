#ifndef DX11_RENDERER_H
#define DX11_RENDERER_H

#include <vertex_types.h>
#include <LIST.h>
#include <SURFACE.h>
#include <DX11_SAMPLER.h>
#include <DX11_RASTERIZER_STATE.h>
#include <DX11_DEPTH_STENCIL_STATE.h>
#include <DX11_BLEND_STATE.h>
#include <RENDER_TARGET_CONFIG.h>
#include <DX11_RENDER_TARGET.h>
#include <DX11_VERTEX_BUFFER.h>
#include <DX11_INDEX_BUFFER.h>
#include <DX11_UNIFORM_BUFFER.h>
#include <DX11_STRUCTURED_BUFFER.h>
#include <DX11_SHADER.h>
#include <CAMERA.h>
#include <POINT_LIGHT.h>
#include <DIRECTIONAL_LIGHT.h>
#include <MESH.h>
#include <IPOST_PROCESSOR.h>

#define CLEAR_COLOR COLOR(0.0f,0.0f,0.0f,0.0f) // render-target clear color
#define CLEAR_DEPTH 1.0f // render-target clear depth
#define CLEAR_STENCIL 0 // render-target clear stencil 

// predefined IDs for frequently used samplers
enum samplerID
{
	LINEAR_SAMPLER_ID=0,
	TRILINEAR_SAMPLER_ID,
	SHADOW_MAP_SAMPLER_ID
};

// predefined IDs for frequently used render-targets
enum renderTargetID
{
	BACK_BUFFER_RT_ID=0, // back buffer
	GBUFFER_RT_ID, // geometry buffer
	SHADOW_MAP_RT_ID // shadow map
};

// predefined IDs for frequently used cameras 
enum cameraID
{
	MAIN_CAMERA_ID
};

// predefined IDs for frequently used meshes 
enum meshID
{
	SCREEN_QUAD_MESH_ID=0,
	UNIT_SPHERE_MESH_ID
};

// DX11_RENDERER
//   Manages DirectX 11 rendering.
class DX11_RENDERER
{
public:
	DX11_RENDERER()
	{
		device = NULL;
		deviceContext = NULL;
		swapChain = NULL;
		noneCullRS = NULL;
		noDepthTestDSS = NULL;
		defaultBS = NULL;
		frameCleared = false;
	}

	~DX11_RENDERER()
	{
		Destroy();
	}

	void Destroy();	

	bool Create();

	DX11_SAMPLER* CreateSampler(const SAMPLER_DESC &desc);

	DX11_SAMPLER* GetSampler(int ID) const;

	DX11_RASTERIZER_STATE* CreateRasterizerState(const RASTERIZER_DESC &desc);

	DX11_DEPTH_STENCIL_STATE* CreateDepthStencilState(const DEPTH_STENCIL_DESC &desc);

	DX11_BLEND_STATE* CreateBlendState(const BLEND_DESC &desc);

	RENDER_TARGET_CONFIG* CreateRenderTargetConfig(const RT_CONFIG_DESC &desc);

	DX11_RENDER_TARGET* CreateBackBufferRT();

	DX11_RENDER_TARGET* CreateRenderTarget(int width,int height,int depth,texFormats format=TEX_FORMAT_RGB16F,bool depthStencil=false,
		                                     int numColorBuffers=1,DX11_SAMPLER *sampler=NULL,bool useUAV=false);

	DX11_RENDER_TARGET* GetRenderTarget(int ID) const;

	DX11_VERTEX_BUFFER* CreateVertexBuffer(const VERTEX_ELEMENT_DESC *vertexElementDescs,int numVertexElementDescs,
		                                     bool dynamic,int maxVertexCount);

	DX11_INDEX_BUFFER* CreateIndexBuffer(bool dynamic,int maxIndexCount);

	DX11_UNIFORM_BUFFER* CreateUniformBuffer(uniformBufferBP bindingPoint,const UNIFORM_LIST &uniformList);

	DX11_STRUCTURED_BUFFER* CreateStructuredBuffer(int bindingPoint,int elementCount,int elementSize);

	CAMERA* CreateCamera(float fovy,float nearClipDistance,float farClipDistance);

	CAMERA* GetCamera(int ID) const;

	POINT_LIGHT* CreatePointLight(const VECTOR3D &position,float radius,const COLOR &color,float multiplier);

	DIRECTIONAL_LIGHT* CreateDirectionalLight(const VECTOR3D &direction,const COLOR &color,float multiplier);

	ILIGHT* GetLight(int index) const;

	MESH* CreateMesh(primitiveTypes primitiveType,const VERTEX_ELEMENT_DESC *vertexElementDescs,
		               int numVertexElementDescs,bool dynamic,int numVertices,int numIndices);

	MESH* GetMesh(int ID) const;

	template<class T> T* CreatePostProcessor()
	{
		T *postProcessor = new T;
		if(!postProcessor)
			return NULL;
		if(!postProcessor->Create())
		{
			SAFE_DELETE(postProcessor);
			return NULL;
		}
		postProcessors.AddElement((IPOST_PROCESSOR**)(&postProcessor));
		return postProcessor;
	}

	IPOST_PROCESSOR* GetPostProcessor(const char *name) const;

	int GetNumLights() const;

	void UpdateLights();

	void SetupPostProcessSurface(SURFACE &surface);

	// add new surface per frame 
	void AddSurface(SURFACE &surface);

	void ClearFrame();

	// draw all surfaces, which have been passed per frame to renderer
	void DrawSurfaces();

	// save a BMP screen-shot
	void SaveScreenshot() const;

	ID3D11Device* GetDevice() const
	{
		return device;
	}

  ID3D11DeviceContext* GetDeviceContext() const
	{
		return deviceContext;
	}

  IDXGISwapChain* GetSwapChain() const
	{
		return swapChain;
	}

private:  	
	// create frequently used objects
	bool CreateDefaultObjects();
	
	// create mesh for a full-screen quad
	bool CreateScreenQuadMesh();

	// create unit sphere geometry
	bool CreateUnitSphere();

	void ExecutePostProcessors();

	// set render states for passed surface
	void SetRenderStates(SURFACE &surface);

	// set shader params for passed surface
	void SetShaderParams(SURFACE &surface);

	void DrawIndexedElements(primitiveTypes primitiveType,int numElements,int firstIndex,int numInstances);

	void DrawElements(primitiveTypes primitiveType,int numElements,int firstIndex,int numInstances);

	void Dispatch(int numThreadGroupsX,int numThreadGroupsY,int numThreadGroupsZ); 
	
	void UnbindShaderResources();

	// list of all samplers
	LIST<DX11_SAMPLER*> samplers;

	// list of all rasterizer states
	LIST<DX11_RASTERIZER_STATE*> rasterizerStates;

	// list of all depth-stencil states
	LIST<DX11_DEPTH_STENCIL_STATE*> depthStencilStates;

	// list of all blend states
	LIST<DX11_BLEND_STATE*> blendStates;

	// list of all render-target configs
	LIST<RENDER_TARGET_CONFIG*> renderTargetConfigs;

	// list of all render-targets
	LIST<DX11_RENDER_TARGET*> renderTargets;

	// list of all vertex-buffers
	LIST<DX11_VERTEX_BUFFER*> vertexBuffers;

	// list of all index-buffers
	LIST<DX11_INDEX_BUFFER*> indexBuffers;

	// list of all uniform-buffers
	LIST<DX11_UNIFORM_BUFFER*> uniformBuffers;

	// list of all structured buffers
	LIST<DX11_STRUCTURED_BUFFER*> structuredBuffers;

	// list of all cameras
	LIST<CAMERA*> cameras;	

	// list of all dynamic lights
	LIST<ILIGHT*> lights;

	// list of all dynamically created meshes
	LIST<MESH*> meshes;

	// list of all post-processors
	LIST<IPOST_PROCESSOR*> postProcessors;

	// list of all per frame passed surfaces 
	LIST<SURFACE> surfaces;

	// render-states, frequently used by post-processors
	DX11_RASTERIZER_STATE *noneCullRS;
	DX11_DEPTH_STENCIL_STATE *noDepthTestDSS;
	DX11_BLEND_STATE *defaultBS;

	// helper variables
	SURFACE lastSurface;
	bool frameCleared;

	// DirectX 11 objects
	ID3D11Device *device;
	ID3D11DeviceContext *deviceContext;
	IDXGISwapChain *swapChain;

};

#endif 