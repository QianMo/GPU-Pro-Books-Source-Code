#ifndef GLOBAL_ILLUM_H
#define GLOBAL_ILLUM_H

#include <SURFACE.h>
#include <IPOST_PROCESSOR.h>

class DX11_RENDER_TARGET;
class RENDER_TARGET_CONFIG;
class DX11_UNIFORM_BUFFER;
class DX11_STRUCTURED_BUFFER;
class DX11_SHADER;
class DX11_RASTERIZER_STATE;
class DX11_DEPTH_STENCIL_STATE;
class DX11_BLEND_STATE;	

// this demo uses 2 grid cascades
enum gridTypes
{
	FINE_GRID=0, // fine resolution grid
	COARSE_GRID  // coarse resolution grid
};

// different debug views
enum globalIllumModes
{
	DEFAULT_GIM=0,           // default combined output (direct + indirect illumination)
	DIRECT_ILLUM_ONLY_GIM,   // direct illumination only
	INDIRECT_ILLUM_ONLY_GIM, // indirect illumination only
	VISUALIZE_GIM,           // visualization of voxel-grids  
};

// GLOBAL_ILLUM
//   This post-processor performs "Rasterized Voxel based Dynamic Global Illumination".
//   Actually the technique can be divided into 5 steps:
//   1. In an initial step a voxel-representation of the scene geometry is created by 
//      utilizing the hardware rasterizer.
//   2. For each light the low resolution voxel-grid is illuminated and from the illuminated 
//      voxels virtual point-lights (represented as spherical harmonics) are created.
//   3. The virtual point-lights are propagated iteratively until the required light distribution
//      is achieved. Thereby the previously generated voxel-grid is utilized to perform geometric
//      occlusion for the propagated light.
//   4. With the help of the normal-/ depth-buffer (of the Gbuffer) each visible pixel is illuminated,
//      thereby generating the actual indirect illumination.
//   5. Finally the voxel grid is cleared.
//   The voxel- as well as the virtual point-light grid are consisting of 32x32x32 cells. In order to
//   cover the entire scene, therefore two cascades are used, a fine and a coarse resolution grid.
class GLOBAL_ILLUM: public IPOST_PROCESSOR
{
public: 
	GLOBAL_ILLUM()
	{
		strcpy(name,"GLOBAL_ILLUM");
	
    sceneRT = NULL;
	  blendBS = NULL;
		gridUniformBuffer = NULL;
	
    gridRT = NULL;
		gridRTCs[FINE_GRID] = NULL;
		gridRTCs[COARSE_GRID] = NULL;
    gridSBs[FINE_GRID] = NULL;
		gridSBs[COARSE_GRID] = NULL;
	  gridFillShaders[FINE_GRID] = NULL;
    gridFillShaders[COARSE_GRID] = NULL;
		gridRS = NULL;
		gridDSS = NULL;
		gridBS = NULL;	
		
		for(int i=0;i<2;i++)
		{
			lightRTs[FINE_GRID][i] = NULL; 
			lightRTs[COARSE_GRID][i] = NULL; 
		}
		currentLightRTIndices[FINE_GRID] = 0;
		currentLightRTIndices[COARSE_GRID] = 0;

		lightPropagateRTC = NULL;
		for(int i=0;i<2;i++)
		{
			lightPropagateShaders[FINE_GRID][i] = NULL;
			lightPropagateShaders[COARSE_GRID][i] = NULL;
		}

		outputRTC = NULL;
		globalIllumShader = NULL;
		globalIllumNoTexShader = NULL;
		stencilTestDSS = NULL;

		gridVisShader = NULL;

		clearRT = NULL;
		clearRTC = NULL;
		clearShader = NULL;

		mode = DEFAULT_GIM;
		useOcclusion = true;
	}

	virtual bool Create();

	virtual DX11_RENDER_TARGET* GetOutputRT() const;

	virtual void AddSurfaces();

	// configures surface for generating the voxel-grid
	void SetupGridSurface(SURFACE &surface,gridTypes gridType);

	// configures surface for illuminating the voxel-grid
	void SetupLightGridSurface(SURFACE &surface,gridTypes gridType);

	void SetGlobalIllumMode(globalIllumModes mode)
	{
		this->mode = mode;
	}

	bool IsOcclusionEnabled() const
	{
		return useOcclusion;
	}

	void EnableOcclusion(bool enable)
	{
		useOcclusion = enable;
	}

private:
	void Update();

  void UpdateGrid(int gridType);

	// performs illumination of the voxel-grids
	void PerformGridLightingPass();
  
  // performs propagation of virtual point-lights
	void PerformLightPropagatePass(int index,gridTypes gridType);

	// perform actual indirect illumination
	void PerformGlobalIllumPass();	

	// visualizes voxel-grids
	void PerformGridVisPass();

	// clears voxel-grids
  void PerformClearPass();
	
	// commonly used objects
	DX11_RENDER_TARGET *sceneRT;              // GBuffer
	DX11_BLEND_STATE *blendBS;                // use additive blending
	DX11_UNIFORM_BUFFER *gridUniformBuffer;   // uniform buffer with information about the grids  

	// objects used for generating the voxel-grids
	DX11_RENDER_TARGET *gridRT;               // simple 64x64 RGB8 render-target 
	RENDER_TARGET_CONFIG *gridRTCs[2];        // render-target configs for FINE_GRID/ COARSE_GRID
	DX11_STRUCTURED_BUFFER *gridSBs[2];       // structured buffers for FINE_GRID/ COARSE_GRID
	DX11_SHADER *gridFillShaders[2];          // shaders for FINE_GRID/ COARSE_GRID
	DX11_RASTERIZER_STATE *gridRS;            // default rasterizer state (no culling, solid mode)
	DX11_DEPTH_STENCIL_STATE *gridDSS;        // no depth-write/ -test depth-stencil state
	DX11_BLEND_STATE *gridBS;	                // default blend state (blending disabled)

	// objects used for illuminating the voxel-grids
	DX11_RENDER_TARGET *lightRTs[2][2];	      // two 32x32x32 RGBA16F render-targets for each FINE_GRID/ COARSE_GRID
	int currentLightRTIndices[2];             // keep track of currently set render-target for FINE_GRID/ COARSE_GRID

	// objects used for the light propagation
	RENDER_TARGET_CONFIG *lightPropagateRTC;  // render-target config for using the compute shader
	DX11_SHADER *lightPropagateShaders[2][2]; // shaders for FINE_GRID/ COARSE_GRID (with and without occlusion)

	// objects used for generating the indirect illumination
  RENDER_TARGET_CONFIG *outputRTC;          // only render into the accumulation render-target of the GBuffer
	DX11_SHADER *globalIllumShader;           // default shader for generating textured indirect illumination
	DX11_SHADER *globalIllumNoTexShader;      // shader for visualizing indirect illumination only (without texturing)
	DX11_DEPTH_STENCIL_STATE *stencilTestDSS; // only illuminate actual geometry, not the sky

	// objects used for visualizing the voxel-grids
	DX11_SHADER *gridVisShader;               // shader for voxel-grid visualization

	// objects used for clearing the voxel-grids  
	DX11_RENDER_TARGET *clearRT;              // empty render-target
	RENDER_TARGET_CONFIG *clearRTC;           // render-target config to configure above render-target for the compute shader
	DX11_SHADER *clearShader;                 // shader to clear both voxel-grids (FINE_GRID/ COARSE_GRID)

	// data for grid uniform-buffer
	MATRIX4X4 gridViewProjMatrices[6];        // viewProjMatrices for generating the voxel-grids
	VECTOR4D gridCellSizes;                   // (inverse) sizes of grid-cells (FINE_GRID/ COARSE_GRID)
	VECTOR4D gridPositions[2];                // center of FINE_GRID/ COARSE_GRID
	VECTOR4D snappedGridPositions[2];         // center of FINE_GRID/ COARSE_GRID, snapped to the corresponding grid-cell extents 
	
	// helper variables  
	float gridHalfExtents[2];                 // half extents of cubic FINE_GRID/ COARSE_GRID     
	MATRIX4X4 gridProjMatrices[2];            // orthographic projection matrices for FINE_GRID/ COARSE_GRID   
	globalIllumModes mode;
	bool useOcclusion;
	
};

#endif