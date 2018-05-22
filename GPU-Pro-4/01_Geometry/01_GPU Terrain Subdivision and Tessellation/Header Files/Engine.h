//-------------------------------------------------------------------------------------------------
// File: Engine.h
// Author: Ben Mistal
// Copyright 2010-2012 Mistal Research, Inc.
//-------------------------------------------------------------------------------------------------
#include <d3d11.h>
#include <d3dx11.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <math.h>
#include "Helpers.h"
#include <D3DX10math.h>

#define NumTempVertexBuffers	4 // 0=Initial Data, 1=Final Data, 2/3=Temp Data
#define InitialBufferIndex		0
#define FinalBufferIndex		1
#define TempStartBufferIndex	2
#define MaxNumStoredFrameTimes	60
#define NumOctaveWraps			2


#define nHSize 1024
#define nVSize 768
#define fMinEyeHeight 0.1f

class CEngine
{
protected: // Protected Member Definitions

	struct FractalGeneratorInfoStruct
	{
		float fOffset;
		float fGain;
		float fH;
		float fLacunarity;

	}; // end struct FractalGeneratorInfoStruct

	struct FractalOctaveInfoStruct
	{
		float fSinArray[ 16 ];
		float fCosArray[ 16 ];
		float fReverseSinArray[ 16 ];
		float fReverseCosArray[ 16 ];
		float fXOffsetArray[ 16 ];
		float fYOffsetArray[ 16 ];
		float fExponentArray[ 16 ];
		float fOctaveLengthArray[ 16 ];

	}; // end struct FractalGeneratorInfoStruct

	struct GeneralFractalInfoStruct
	{
		FractalGeneratorInfoStruct	fractalGeneratorInfo;
		FractalOctaveInfoStruct		fractalOctaveInfo;

	}; // end struct GeneralFractalInfoStruct

	struct ConstantBuffer
	{
		D3DXMATRIX	matWorld;
		D3DXMATRIX	matView;
		D3DXMATRIX	matProjection;
		D3DXMATRIX	matWorldView;
		D3DXMATRIX	matWorldViewProjection;
		D3DXVECTOR4 fvViewFrustumPlanes[ 6 ];
		D3DXVECTOR4 fvControlPosition;
		D3DXVECTOR4 fvEye;
		D3DXVECTOR4 fvLookAt;
		D3DXVECTOR4 fvUp;

		float		fMaxRegionSpan;
		float		fHSize;
		float		fVSize;
		float		fUnused;

		GeneralFractalInfoStruct generalFractalInfo;

	}; // end struct ConstantBuffer

	struct int4
	{
		_int32 x, y, z, w;

	}; // end struct int4

	struct RegionVertexElement
	{
		int4 nvPosition;
		int4 nvParentInfo;

	}; // end struct RegionVertexElement

	struct CompositeVertexElement
	{
		D3DXVECTOR4 fvPosition;
		D3DXVECTOR2 fvTexcoord;

	}; // end struct CompositeVertexElement

	struct TimerInformation
	{
		LARGE_INTEGER	m_nTimerFrequency;
		unsigned int	m_nFrameCount;
		LARGE_INTEGER	m_nFrameTimeArray[ MaxNumStoredFrameTimes ];
		LARGE_INTEGER	m_nLastPrintedStatTime;

	}; // end struct TimerInformation

	struct OrientationInformation
	{
		float	m_fvViewRotation[ 2 ];
		int 	m_nvPreviousPosition[ 2 ];
		bool	m_bLeftMouseButtonDown;
		bool	m_bLeftArrowButtonDown;
		bool	m_bRightArrowButtonDown;
		bool	m_bUpArrowButtonDown;
		bool	m_bDownArrowButtonDown;
		bool	m_bAddButtonDown;
		bool	m_bSubtractButtonDown;
		bool	m_bPageUpButtonDown;
		bool	m_bPageDownButtonDown;
		unsigned int nInverseMaxRegionSpan;

	}; // end struct OrientationInformation

public: // Construction / Destruction

	CEngine();
	virtual ~CEngine();

public: // Public Member Functions

	HRESULT InitDevice( HWND hWnd );
	void CleanupDevice();
	HRESULT ResizeDevice();

	void MessageHandler( UINT, WPARAM, LPARAM );

	void Render( bool bHardwareDeviceOnly = true );

protected: // Protected Member Functions

	HRESULT InitTextures();
	HRESULT InitBuffers();
	HRESULT InitShaders();
	HRESULT InitStateObjects();

	void SetDefaultSettings();

	HRESULT CreateValueGradientTexture( ID3D11Texture2D*& p2DValueGradientTexture, const unsigned int nNumOctaveWraps );

	static HRESULT CompileShaderFromFile(	WCHAR*				szFileName,
											LPCSTR				szEntryPoint,
											LPCSTR				szShaderModel,
											D3D10_SHADER_MACRO* pDefines,
											ID3DBlob**			ppBlobOut );

	void SetConstants();
	void UpdateConstants();

	void SubdivideRegions();
	void RenderRegions();
	void RenderWireframeRegions();

	void CompositeBackground();
	void CompositeScene();

	void SetFullScreenMode( bool bFullScreen );

	void CalculateTimerStats(	unsigned int &nStoredTimerFrameCount,
								LARGE_INTEGER &nMinimumTime,
								LARGE_INTEGER &nMaximumTime,
								LARGE_INTEGER &nTotalTime );

protected: // Protected Member Variables

	HWND						m_hWnd;

	D3D_DRIVER_TYPE             m_driverType;
	D3D_FEATURE_LEVEL           m_featureLevel;
	ID3D11Device*               m_pd3dDevice;
	ID3D11DeviceContext*        m_pImmediateContext;
	IDXGISwapChain*             m_pSwapChain;
    ID3D11Texture2D*			m_pBackBuffer;
	ID3D11RenderTargetView*     m_pRenderTargetView;
	ID3D11Texture2D*            m_pDepthStencilTexture;
	ID3D11DepthStencilView*     m_pDepthStencilTextureDSV;

	ID3D11InputLayout*          m_pRegionVertexLayout;
	ID3D11VertexShader*         m_pRegionPassThroughVertexShader;
	ID3D11GeometryShader*		m_pRegionSplitGeometryShader;
	ID3D11GeometryShader*		m_pRegionFaceGeometryShader;
	ID3D11GeometryShader*		m_pRegionWireGeometryShader;
	ID3D11PixelShader*          m_pRegionPixelShader;
	ID3D11PixelShader*          m_pRegionWireframePixelShader;

	ID3D11Buffer*               m_pRegionVertexBufferArray[ NumTempVertexBuffers ];

	ID3D11RasterizerState*		m_pRasterizerState;
	ID3D11DepthStencilState*	m_pRegionDepthStencilState;
	ID3D11DepthStencilState*	m_pBackgroundDepthStencilState;
	ID3D11DepthStencilState*	m_pCompositeDepthStencilState;
	ID3D11Buffer*				m_pConstantBuffer;
	ID3D11Buffer*				m_pControlConstantBuffer;
	ConstantBuffer				m_constantBufferData;

	ID3D11Resource*				m_p2DValueGradientTextureResource;
	ID3D11Texture2D*			m_p2DValueGradientTexture;
	ID3D11ShaderResourceView*	m_p2DValueGradientTextureRV;

	ID3D11VertexShader*         m_pCompositeVertexShader;
	ID3D11PixelShader*          m_pBackgroundPixelShader;
	ID3D11PixelShader*          m_pCompositePixelShader;

	ID3D11InputLayout*          m_pCompositeVertexLayout;
	ID3D11Buffer*               m_pCompositeVertexBuffer;
	ID3D11Buffer*               m_pCompositeIndexBuffer;	

	ID3D11Texture2D*			m_pPositionTexture;
	ID3D11ShaderResourceView*	m_pPositionTextureRV;
	ID3D11RenderTargetView*     m_pPositionRenderTargetView;

	ID3D11SamplerState*         m_pSamplerLinearWrap;
	ID3D11SamplerState*         m_pSamplerPointClamp;
	ID3D11SamplerState*         m_pSamplerPointWrap;

	bool						m_bShowWireframe;
	bool						m_bFullScreen;
	bool						m_bHardwareDevice;

	OrientationInformation		m_orientationInformation;
	TimerInformation			m_timerInformation;

}; // end class CEngine