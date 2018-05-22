// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#ifndef __RAYTRACERCS_H__
#define __RAYTRACERCS_H__

#ifdef WINDOWS

#include "ArgumentsParser.h"
extern ArgumentsParser m_Parser;

#include "Common.h"
#include "D3D11Object.h"
#include "Scene.h"
#include "Camera.h"
#include "Input.h"
#include "Performance.h"
#include "D3DComputeShader.h"
#include "D3DResource.h"
#include "ConstantBuffers.h"

class RayTracerCS
{
public:
	RayTracerCS(Scene *a_Scene, HWND &m_hWnd);
	~RayTracerCS();

	void									Render();
	LRESULT CALLBACK						WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	bool									UpdateWindowTitle(char aux[1024], char* str) { return m_pTimeTracker->updateFPS(aux,str);}

private:
	// UAVs
	D3DResource								m_uRays; // store the generated/bounced rays
	D3DResource								m_uIntersections; // store the intersection information for a given ray
	D3DResource								m_uAccumulation; // store the accumulated color
	D3DResource								m_uPrimitives; // empty
	D3DResource								m_uNodes; // empty
	D3DResource								m_uMortonCode;	// empty
	D3DResource								m_uResult; // store the texture to render

	// SRVs
	D3DResource								m_sVertices; // store the vertices of the model
	D3DResource								m_sIndices;	// store the indices of the model
	D3DResource								m_sPrimitives; // store the ordered list of primitives
	D3DResource								m_sNodes; // store the BVH structure
	D3DResource								m_sLBVHNodes; // store the LBVH structure
	D3DResource								m_sMaterials; // store the material id of a given primitive
	D3DResource								m_sColorTextures; // store an Array2D of textures
	D3DResource								m_sNormalMapTextures; // store an Array2D of normal maps
	D3DResource								m_sSpecularMapTextures; // store an Array2D of sepecular maps
	D3DResource								m_sRandomMapTextures; // store random values for GI app
	D3DResource								m_sEnvMapTextures; // store an environment mapping texture

	// Constant Buffers
	D3DResource								m_cbCamera; // store the camera information
	D3DResource								m_cbUserInput; // store the values of current keyboard/mouse status
	D3DResource								m_cbLight; // store the light information
	D3DResource								m_cbGlobalIllumination; // store the GI values

	// Compute shaders
	D3DComputeShader						m_csPrimaryRays; // generate the primary rays
	D3DComputeShader						m_csIntersections; // compute ray-triangle intersections
	D3DComputeShader						m_csColor; // compute the color of a pixel
	
	UINT									GRID_SIZE[3]; // number of groups to execute (xyz)
	UINT									GROUP_SIZE[3]; // number of threads per group (xyz)
	std::vector<ID3D11UnorderedAccessView*>	m_vpUAViews; // store the UAVs structures used by the CSs
	std::vector<ID3D11ShaderResourceView*>	m_vpSRViews; // store the SRVs structures used by the CSs
	std::vector<ID3D11Buffer*>				m_vpCBuffers; // store the CBs used by the CSs

	// Objects
	D3D11Object*							m_pDXObject; // DirectX11 object
	Scene*									m_pScene; // loaded scene information
	Camera*									m_pCamera; // camera definition
	Light*									m_pLight; // light object
	Input*									m_pInput; // user input (keyboard-mouse) object
	int*									m_pMaterial; // buffer containing material ids
	Performance*							m_pTimeTracker; // time tracking helper class
	
	unsigned  int							m_NumMuestras;

	// Functions
	template <class U> HRESULT				DebugCSBuffer ( ID3D11Resource* pBuffer, int iEnd, int iStart = 0 );
	template <class R, class T> HRESULT		UpdateCB(ID3D11Resource* pResource, T* pObj);
	HRESULT									CreateBVHBuffers();
	HRESULT									Init();
	HRESULT									LoadTextures();
	void									LoadShaders();
	void									SelectAccelerationStructure();
};

#endif
#endif