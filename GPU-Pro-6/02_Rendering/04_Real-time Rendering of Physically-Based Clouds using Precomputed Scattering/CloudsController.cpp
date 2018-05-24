#include "StdAfx.h"
#include "CloudsController.h"
#include "Structures.fxh"
#include "Visibility.h"
#include "ShaderMacroHelper.h"
#include "IGFXExtensionsHelper.h"

// This helper template function traverses the 3D cloud lattice
template<typename CProc>
void TraverseCloudLattice(const UINT iNumRings,
                          const UINT iInnerRingDim,
                          const UINT iRingExtension,
                          const UINT iNumLayers,
                          CProc Proc)
{
    UINT iRingDimension = iInnerRingDim + 2*iRingExtension;
    assert( (iInnerRingDim % 4) == 0 );
    for(int iRing = iNumRings-1; iRing >= 0; --iRing)
    {
        for(UINT iRow = 0; iRow < iRingDimension; ++iRow)
        {
            UINT iFirstQuart = iRingExtension + iInnerRingDim*1/4;
            UINT iThirdQuart = iRingExtension + iInnerRingDim*3/4;
            UINT iStartCol[2] = {0,           iThirdQuart   };
            UINT iEndCol[2]   = {iFirstQuart, iRingDimension};
            if( !(iRing > 0 && iRow >= iFirstQuart && iRow < iThirdQuart) )
                iStartCol[1] = iEndCol[0];

            for(int i=0; i < _countof(iStartCol); ++i)
                for(UINT iCol = iStartCol[i]; iCol < iEndCol[i]; ++iCol)
                    for(UINT iLayer = 0; iLayer < iNumLayers; ++iLayer)
                        Proc(iCol, iRow, iRing, iLayer);
        }
    }
}

CCloudsController::CCloudsController() : 
    m_strEffectPath(L"fx\\Clouds.fx"),
    m_strPreprocessingEffectPath(L"fx\\Preprocessing.fx"),
    m_uiCloudDensityTexWidth(1024), 
    m_uiCloudDensityTexHeight(1024),
    m_bPSOrderingAvailable(false),
    m_f3PrevLightDir(0,0,0)
{

}

CCloudsController::~CCloudsController()
{

}

void RenderQuad(ID3D11DeviceContext *pd3dDeviceCtx, 
                CRenderTechnique &State, 
                int iWidth = 0, int iHeight = 0,
                int iTopLeftX = 0, int iTopLeftY = 0,
                int iNumInstances = 1);

// This method handles resize event
void CCloudsController::OnResize(ID3D11Device *pDevice, 
                                 UINT uiWidth, UINT uiHeight)
{
    m_uiBackBufferWidth  = uiWidth;
    m_uiBackBufferHeight = uiHeight;
    UINT uiDownscaledWidth  = m_uiBackBufferWidth /m_CloudAttribs.uiDownscaleFactor;
    UINT uiDownscaledHeight = m_uiBackBufferHeight/m_CloudAttribs.uiDownscaleFactor;
    m_CloudAttribs.uiBackBufferWidth = m_uiBackBufferWidth;
    m_CloudAttribs.uiBackBufferHeight = m_uiBackBufferHeight;
    m_CloudAttribs.uiDownscaledBackBufferWidth  = uiDownscaledWidth;
    m_CloudAttribs.uiDownscaledBackBufferHeight = uiDownscaledHeight;

    m_CloudAttribs.fBackBufferWidth  = (float)m_uiBackBufferWidth;
    m_CloudAttribs.fBackBufferHeight = (float)m_uiBackBufferHeight;
    m_CloudAttribs.fDownscaledBackBufferWidth  = (float)uiDownscaledWidth;
    m_CloudAttribs.fDownscaledBackBufferHeight = (float)uiDownscaledHeight;

    // Release existing resources
    m_ptex2DScreenCloudColorSRV.Release();
    m_ptex2DScreenCloudColorRTV.Release();
    m_ptex2DScrSpaceCloudTransparencySRV.Release();
    m_ptex2DScrSpaceCloudTransparencyRTV.Release();
    m_ptex2DScrSpaceDistToCloudSRV.Release();
    m_ptex2DScrSpaceDistToCloudRTV.Release();

    m_ptex2DDownscaledScrCloudColorSRV.Release();
    m_ptex2DDownscaledScrCloudColorRTV.Release();
    m_ptex2DDownscaledScrCloudTransparencySRV.Release();
    m_ptex2DDownscaledScrCloudTransparencyRTV.Release();
    m_ptex2DDownscaledScrDistToCloudSRV.Release();
    m_ptex2DDownscaledScrDistToCloudRTV.Release();

    m_pbufParticleLayersSRV.Release();
    m_pbufParticleLayersUAV.Release();
    m_pbufClearParticleLayers.Release();

    // Create screen space cloud color buffer
    D3D11_TEXTURE2D_DESC ScreenCloudColorTexDesc = 
    {
        m_uiBackBufferWidth,                //UINT Width;
        m_uiBackBufferHeight,               //UINT Height;
        1,                                  //UINT MipLevels;
        1,                                  //UINT ArraySize;
        DXGI_FORMAT_R11G11B10_FLOAT,        //DXGI_FORMAT Format;
        {1,0},                              //DXGI_SAMPLE_DESC SampleDesc;
        D3D11_USAGE_DEFAULT,                //D3D11_USAGE Usage;
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,           //UINT BindFlags;
        0,                                  //UINT CPUAccessFlags;
        0,                                  //UINT MiscFlags;
    };

    HRESULT hr;
    CComPtr<ID3D11Texture2D> ptex2DScreenCloudColor;
    V( pDevice->CreateTexture2D(&ScreenCloudColorTexDesc, nullptr, &ptex2DScreenCloudColor) );
    V( pDevice->CreateShaderResourceView(ptex2DScreenCloudColor, nullptr, &m_ptex2DScreenCloudColorSRV) );
    V( pDevice->CreateRenderTargetView(ptex2DScreenCloudColor, nullptr, &m_ptex2DScreenCloudColorRTV) );

    if( m_CloudAttribs.uiDownscaleFactor > 1 )
    {
        // Create downscaled screen space cloud color buffer
        D3D11_TEXTURE2D_DESC DownscaledScreenCloudColorTexDesc = ScreenCloudColorTexDesc;
        DownscaledScreenCloudColorTexDesc.Width  /= m_CloudAttribs.uiDownscaleFactor;
        DownscaledScreenCloudColorTexDesc.Height /= m_CloudAttribs.uiDownscaleFactor;
        CComPtr<ID3D11Texture2D> ptex2DDownscaledScrCloudColor;
        V( pDevice->CreateTexture2D(&DownscaledScreenCloudColorTexDesc, nullptr, &ptex2DDownscaledScrCloudColor) );
        V( pDevice->CreateShaderResourceView(ptex2DDownscaledScrCloudColor, nullptr, &m_ptex2DDownscaledScrCloudColorSRV) );
        V( pDevice->CreateRenderTargetView(ptex2DDownscaledScrCloudColor, nullptr, &m_ptex2DDownscaledScrCloudColorRTV) );
    }

    {
        // Create screen space cloud transparency buffer
        D3D11_TEXTURE2D_DESC ScreenTransparencyTexDesc = ScreenCloudColorTexDesc;
        ScreenTransparencyTexDesc.Format = DXGI_FORMAT_R8_UNORM;
        CComPtr<ID3D11Texture2D> ptex2DScreenTransparency;
        V( pDevice->CreateTexture2D(&ScreenTransparencyTexDesc, nullptr, &ptex2DScreenTransparency) );
        V( pDevice->CreateShaderResourceView(ptex2DScreenTransparency, nullptr, &m_ptex2DScrSpaceCloudTransparencySRV) );
        V( pDevice->CreateRenderTargetView(ptex2DScreenTransparency, nullptr, &m_ptex2DScrSpaceCloudTransparencyRTV) );
        if( m_CloudAttribs.uiDownscaleFactor > 1 )
        {
            // Create downscaled screen space cloud transparency buffer
            ScreenTransparencyTexDesc.Width  /= m_CloudAttribs.uiDownscaleFactor;
            ScreenTransparencyTexDesc.Height /= m_CloudAttribs.uiDownscaleFactor;
            CComPtr<ID3D11Texture2D> ptex2DDownscaledScrTransparency;
            V( pDevice->CreateTexture2D(&ScreenTransparencyTexDesc, nullptr, &ptex2DDownscaledScrTransparency) );
            V( pDevice->CreateShaderResourceView(ptex2DDownscaledScrTransparency, nullptr, &m_ptex2DDownscaledScrCloudTransparencySRV) );
            V( pDevice->CreateRenderTargetView(ptex2DDownscaledScrTransparency, nullptr, &m_ptex2DDownscaledScrCloudTransparencyRTV) );
        }
    }

    {
        // Create screen space distance to cloud buffer
        D3D11_TEXTURE2D_DESC ScreenDistToCloudTexDesc = ScreenCloudColorTexDesc;
        ScreenDistToCloudTexDesc.Format = DXGI_FORMAT_R32_FLOAT; // We need only the closest distance to camera
        CComPtr<ID3D11Texture2D> ptex2DScrSpaceDistToCloud;
        V( pDevice->CreateTexture2D(&ScreenDistToCloudTexDesc, nullptr, &ptex2DScrSpaceDistToCloud) );
        V( pDevice->CreateShaderResourceView(ptex2DScrSpaceDistToCloud, nullptr, &m_ptex2DScrSpaceDistToCloudSRV) );
        V( pDevice->CreateRenderTargetView(ptex2DScrSpaceDistToCloud, nullptr, &m_ptex2DScrSpaceDistToCloudRTV) );
        if( m_CloudAttribs.uiDownscaleFactor > 1 )
        {
            // Create downscaled screen space distance to cloud buffer
            ScreenDistToCloudTexDesc.Width  /= m_CloudAttribs.uiDownscaleFactor;
            ScreenDistToCloudTexDesc.Height /= m_CloudAttribs.uiDownscaleFactor;
            CComPtr<ID3D11Texture2D> ptex2DDownscaledScrDistToCloud;
            V( pDevice->CreateTexture2D(&ScreenDistToCloudTexDesc, nullptr, &ptex2DDownscaledScrDistToCloud) );
            V( pDevice->CreateShaderResourceView(ptex2DDownscaledScrDistToCloud, nullptr, &m_ptex2DDownscaledScrDistToCloudSRV) );
            V( pDevice->CreateRenderTargetView(ptex2DDownscaledScrDistToCloud, nullptr, &m_ptex2DDownscaledScrDistToCloudRTV) );
        }
    }

    if( m_bPSOrderingAvailable )
    {
        int iNumElements = (m_uiBackBufferWidth  / m_CloudAttribs.uiDownscaleFactor) * 
                           (m_uiBackBufferHeight / m_CloudAttribs.uiDownscaleFactor) * 
                           m_CloudAttribs.uiNumParticleLayers;
        D3D11_BUFFER_DESC ParticleLayersBufDesc = 
        {
            iNumElements * sizeof(SParticleLayer), //UINT ByteWidth;
            D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,  //UINT MiscFlags;
            sizeof(SParticleLayer)               //UINT StructureByteStride;
        };
    
        CComPtr<ID3D11Buffer> pbufParticleLayers;
        V(pDevice->CreateBuffer( &ParticleLayersBufDesc, nullptr, &pbufParticleLayers));
        D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        ZeroMemory(&SRVDesc, sizeof(SRVDesc));
        SRVDesc.Format = DXGI_FORMAT_UNKNOWN;
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        SRVDesc.Buffer.FirstElement = 0;
        SRVDesc.Buffer.NumElements = iNumElements;
        V(pDevice->CreateShaderResourceView( pbufParticleLayers, &SRVDesc, &m_pbufParticleLayersSRV));
        D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
        UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
        UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        UAVDesc.Buffer.FirstElement = 0;
        UAVDesc.Buffer.NumElements = iNumElements;
        UAVDesc.Buffer.Flags = 0;
        V(pDevice->CreateUnorderedAccessView( pbufParticleLayers, &UAVDesc, &m_pbufParticleLayersUAV));

        std::vector<SParticleLayer> ClearLayers(iNumElements);
        ParticleLayersBufDesc.Usage = D3D11_USAGE_IMMUTABLE;
        ParticleLayersBufDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA InitData = {&ClearLayers[0],0,0};
        V(pDevice->CreateBuffer( &ParticleLayersBufDesc, &InitData, &m_pbufClearParticleLayers));
    }
}

// This method renders maximum density mip map
void CCloudsController::RenderMaxDensityMip( ID3D11Device *pDevice, 
                                             ID3D11DeviceContext *pDeviceContext, 
                                             ID3D11Texture2D *ptex2DMaxDensityMipMap, 
                                             ID3D11Texture2D *ptex2DTmpMaxDensityMipMap, 
                                             const D3D11_TEXTURE2D_DESC &MaxCloudDensityMipDesc )
{
    HRESULT hr;
    CD3DShaderMacroHelper Macros;
    DefineMacros(Macros);
    Macros.Finalize();
     
    // Create techniques
    CRenderTechnique RenderMaxDensityLevel0Tech;
    RenderMaxDensityLevel0Tech.SetDeviceAndContext(pDevice, pDeviceContext);
    RenderMaxDensityLevel0Tech.CreateVGPShadersFromFile(m_strEffectPath, "ScreenSizeQuadVS", nullptr, "RenderMaxMipLevel0PS", Macros);
    RenderMaxDensityLevel0Tech.SetDS( m_pdsDisableDepth );
    RenderMaxDensityLevel0Tech.SetRS( m_prsSolidFillNoCull );
    RenderMaxDensityLevel0Tech.SetBS( m_pbsDefault );

    CRenderTechnique RenderCoarseMaxMipLevelTech;
    RenderCoarseMaxMipLevelTech.SetDeviceAndContext(pDevice, pDeviceContext);
    RenderCoarseMaxMipLevelTech.CreateVGPShadersFromFile(m_strEffectPath, "ScreenSizeQuadVS", nullptr, "RenderCoarseMaxMipLevelPS", Macros);
    RenderCoarseMaxMipLevelTech.SetDS( m_pdsDisableDepth );
    RenderCoarseMaxMipLevelTech.SetRS( m_prsSolidFillNoCull );
    RenderCoarseMaxMipLevelTech.SetBS( m_pbsDefault );

    CComPtr<ID3D11RenderTargetView> pOrigRTV;
    CComPtr<ID3D11DepthStencilView> pOrigDSV;
    pDeviceContext->OMGetRenderTargets(1, &pOrigRTV, &pOrigDSV);

    D3D11_VIEWPORT OrigViewPort;
    UINT iNumOldViewports = 1;
    pDeviceContext->RSGetViewports(&iNumOldViewports, &OrigViewPort);

    UINT uiCurrMipWidth = MaxCloudDensityMipDesc.Width;
    UINT uiCurrMipHeight = MaxCloudDensityMipDesc.Height;
    for(UINT uiMip = 0; uiMip < MaxCloudDensityMipDesc.MipLevels; ++uiMip)
    {
        D3D11_RENDER_TARGET_VIEW_DESC RTVDesc;
        RTVDesc.Format = MaxCloudDensityMipDesc.Format;
        RTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        RTVDesc.Texture2D.MipSlice = uiMip;
        CComPtr<ID3D11RenderTargetView> ptex2DTmpMaxDensityMipMapRTV;
        V(pDevice->CreateRenderTargetView(ptex2DTmpMaxDensityMipMap, &RTVDesc, &ptex2DTmpMaxDensityMipMapRTV));

        pDeviceContext->OMSetRenderTargets(1, &ptex2DTmpMaxDensityMipMapRTV.p, nullptr);

        ID3D11SamplerState *pSamplers[] = {m_psamPointWrap};
        pDeviceContext->PSSetSamplers(2, _countof(pSamplers), pSamplers);

        m_CloudAttribs.f4Parameter.x = (float)uiMip;
        UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));

        ID3D11Buffer *pCBs[] = {m_pcbGlobalCloudAttribs};
        pDeviceContext->PSSetConstantBuffers(0, _countof(pCBs), pCBs);

        if(uiMip == 0)
        {
            ID3D11ShaderResourceView *pSRVs[] = {m_ptex2DCloudDensitySRV};
            pDeviceContext->PSSetShaderResources(1, _countof(pSRVs), pSRVs);
        }
        else
        {
            ID3D11ShaderResourceView *pSRVs[] = {m_ptex2DMaxDensityMipMapSRV};
            pDeviceContext->PSSetShaderResources(3, _countof(pSRVs), pSRVs);
        }

        RenderQuad(pDeviceContext, uiMip == 0 ? RenderMaxDensityLevel0Tech : RenderCoarseMaxMipLevelTech, uiCurrMipWidth, uiCurrMipHeight);
        
        pDeviceContext->CopySubresourceRegion(ptex2DMaxDensityMipMap, uiMip, 0,0,0, ptex2DTmpMaxDensityMipMap, uiMip, nullptr);

        uiCurrMipWidth /= 2;
        uiCurrMipHeight /= 2;
    }
    assert( uiCurrMipWidth == 0 && uiCurrMipHeight == 0 );

    pDeviceContext->OMSetRenderTargets(1, &pOrigRTV.p, pOrigDSV);
    pDeviceContext->RSSetViewports(iNumOldViewports, &OrigViewPort);
}


// Auxiliary method which creates a buffer and views
HRESULT CCloudsController::CreateBufferAndViews(ID3D11Device *pDevice,
                                                const D3D11_BUFFER_DESC &BuffDesc, 
                                                D3D11_SUBRESOURCE_DATA *pInitData, 
                                                ID3D11Buffer**ppBuffer, 
                                                ID3D11ShaderResourceView **ppSRV, 
                                                ID3D11UnorderedAccessView **ppUAV,
                                                UINT UAVFlags /*= 0*/)
{
    HRESULT hr;

    CComPtr< ID3D11Buffer > ptmpBuffer;
    if( ppBuffer == nullptr )
        ppBuffer = &ptmpBuffer;

    V_RETURN(pDevice->CreateBuffer( &BuffDesc, pInitData, ppBuffer));

    if( ppSRV )
    {
        assert( BuffDesc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED );

        D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        ZeroMemory(&SRVDesc, sizeof(SRVDesc));
        SRVDesc.Format =  DXGI_FORMAT_UNKNOWN;
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        SRVDesc.Buffer.FirstElement = 0;
        SRVDesc.Buffer.NumElements = BuffDesc.ByteWidth / BuffDesc.StructureByteStride;
        V_RETURN(pDevice->CreateShaderResourceView( *ppBuffer, &SRVDesc, ppSRV));
    }

    if( ppUAV )
    {
        assert( BuffDesc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED );

        D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
        UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
        UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        UAVDesc.Buffer.FirstElement = 0;
        UAVDesc.Buffer.NumElements = BuffDesc.ByteWidth / BuffDesc.StructureByteStride;
        UAVDesc.Buffer.Flags = UAVFlags;
        V_RETURN(pDevice->CreateUnorderedAccessView( *ppBuffer, &UAVDesc, ppUAV));
    }

    return S_OK;
}

// Method crates particle buffers
HRESULT CCloudsController::CreateParticleDataBuffer(ID3D11Device *pDevice)
{
    m_CloudAttribs.uiRingDimension = m_CloudAttribs.uiInnerRingDim + m_CloudAttribs.uiRingExtension*2;

    // Populate cell locations array
    m_PackedCellLocations.clear();
    m_PackedCellLocations.reserve(m_CloudAttribs.uiRingDimension * m_CloudAttribs.uiRingDimension * m_CloudAttribs.uiNumRings);
    TraverseCloudLattice(m_CloudAttribs.uiNumRings, m_CloudAttribs.uiInnerRingDim, m_CloudAttribs.uiRingExtension, 1, 
                            [&](UINT i, UINT j, UINT ring, UINT layer)
                            {
                                m_PackedCellLocations.push_back( PackParticleIJRing(i,j,ring, layer) );
                            }
                            );
    m_CloudAttribs.uiNumCells = (UINT)m_PackedCellLocations.size();

    // Populate particle locations array
    m_PackedParticleLocations.clear();
    m_PackedParticleLocations.reserve(m_CloudAttribs.uiNumCells * m_CloudAttribs.uiMaxLayers);
    TraverseCloudLattice(m_CloudAttribs.uiNumRings, m_CloudAttribs.uiInnerRingDim, m_CloudAttribs.uiRingExtension, m_CloudAttribs.uiMaxLayers, 
                            [&](UINT i, UINT j, UINT ring, UINT layer)
                            {
                                m_PackedParticleLocations.push_back( PackParticleIJRing(i,j,ring, layer) );
                            }
                            );
    m_CloudAttribs.uiMaxParticles = (UINT)m_PackedParticleLocations.size();

    HRESULT hr;

    // Create cloud cell attributes buffer
    {
        D3D11_BUFFER_DESC CloudGridBuffDesc = 
        {
            m_CloudAttribs.uiNumCells * sizeof(SCloudCellAttribs), //UINT ByteWidth;
            D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,  //UINT MiscFlags;
            sizeof(SCloudCellAttribs)  //UINT StructureByteStride;
        };
        m_pbufCloudGridSRV.Release();
        m_pbufCloudGridUAV.Release();
        V( CreateBufferAndViews( pDevice, CloudGridBuffDesc, nullptr, nullptr, &m_pbufCloudGridSRV, &m_pbufCloudGridUAV ) );
    }

    // Create particle attributes buffer
    {
        D3D11_BUFFER_DESC ParticleBuffDesc = 
        {
            m_CloudAttribs.uiMaxParticles * sizeof(SParticleAttribs), //UINT ByteWidth;
            D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,  //UINT MiscFlags;
            sizeof(SParticleAttribs)  //UINT StructureByteStride;
        };
        m_pbufCloudParticlesSRV.Release();
        m_pbufCloudParticlesUAV.Release();
        V( CreateBufferAndViews( pDevice, ParticleBuffDesc, nullptr, nullptr, &m_pbufCloudParticlesSRV, &m_pbufCloudParticlesUAV ) );
    }

    // Create buffer for storing particle lighting info
    {
	    D3D11_BUFFER_DESC LightingBuffDesc = 
        {
            m_CloudAttribs.uiMaxParticles * sizeof(SCloudParticleLighting), //UINT ByteWidth;
            D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,  //UINT MiscFlags;
            sizeof(SCloudParticleLighting)			//UINT StructureByteStride;
        };
        m_pbufParticlesLightingSRV.Release();
        m_pbufParticlesLightingUAV.Release();
        V( CreateBufferAndViews( pDevice, LightingBuffDesc, nullptr, nullptr, &m_pbufParticlesLightingSRV, &m_pbufParticlesLightingUAV) );
    }

    // Create buffer for storing cell locations
    {
        D3D11_BUFFER_DESC PackedCellLocationsBuffDesc = 
        {
            m_CloudAttribs.uiNumCells * sizeof(UINT), //UINT ByteWidth;
            D3D11_USAGE_IMMUTABLE,                  //D3D11_USAGE Usage;
            D3D11_BIND_SHADER_RESOURCE,             //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,  //UINT MiscFlags;
            sizeof(UINT)                            //UINT StructureByteStride;
        };
        
        m_pbufPackedCellLocationsSRV.Release();
        D3D11_SUBRESOURCE_DATA InitData = {&m_PackedCellLocations[0], 0, 0};
        CreateBufferAndViews( pDevice, PackedCellLocationsBuffDesc, &InitData, nullptr, &m_pbufPackedCellLocationsSRV, nullptr);
    }

    // Create buffer for storing unordered list of valid cell
    {
	    D3D11_BUFFER_DESC ValidCellsBuffDesc = 
        {
            m_CloudAttribs.uiNumCells * sizeof(UINT),//UINT ByteWidth;
            D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,            //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,  //UINT MiscFlags;
            sizeof(UINT)							//UINT StructureByteStride;
        };
        m_pbufValidCellsUnorderedList.Release();
        m_pbufValidCellsUnorderedListSRV.Release();
        m_pbufValidCellsUnorderedListUAV.Release();
        V( CreateBufferAndViews( pDevice, ValidCellsBuffDesc, nullptr, &m_pbufValidCellsUnorderedList, &m_pbufValidCellsUnorderedListSRV, &m_pbufValidCellsUnorderedListUAV, D3D11_BUFFER_UAV_FLAG_APPEND) );
        
		m_pbufVisibleCellsUnorderedListSRV.Release();
        m_pbufVisibleCellsUnorderedListUAV.Release();
        V( CreateBufferAndViews( pDevice, ValidCellsBuffDesc, nullptr, nullptr, &m_pbufVisibleCellsUnorderedListSRV, &m_pbufVisibleCellsUnorderedListUAV, D3D11_BUFFER_UAV_FLAG_APPEND) );
	}

	{
		D3D11_BUFFER_DESC VisibleParticlesBuffDesc =
		{
			m_CloudAttribs.uiMaxParticles * sizeof(SParticleIdAndDist),           //UINT ByteWidth;
			D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
			D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,            //UINT BindFlags;
			0,                                      //UINT CPUAccessFlags;
			D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,  //UINT MiscFlags;
			sizeof(SParticleIdAndDist)			    //UINT StructureByteStride;
		};
		m_pbufVisibleParticlesUnorderedListSRV.Release();
		m_pbufVisibleParticlesUnorderedListUAV.Release();
		V(CreateBufferAndViews(pDevice, VisibleParticlesBuffDesc, nullptr, nullptr, &m_pbufVisibleParticlesUnorderedListSRV, &m_pbufVisibleParticlesUnorderedListUAV, D3D11_BUFFER_UAV_FLAG_APPEND));

        m_pbufVisibleParticlesSortedListSRV.Release();
        m_pbufVisibleParticlesSortedListUAV.Release();
        V(CreateBufferAndViews(pDevice, VisibleParticlesBuffDesc, nullptr, nullptr, &m_pbufVisibleParticlesSortedListSRV, &m_pbufVisibleParticlesSortedListUAV));
        
		m_pbufVisibleParticlesMergedListSRV.Release();
        m_pbufVisibleParticlesMergedListUAV.Release();
        V(CreateBufferAndViews(pDevice, VisibleParticlesBuffDesc, nullptr, nullptr, &m_pbufVisibleParticlesMergedListSRV, &m_pbufVisibleParticlesMergedListUAV));
	}

    // Create buffer for storing streamed out list of visible particles
    {
	    D3D11_BUFFER_DESC SerializedParticlesBuffDesc = 
        {
            m_CloudAttribs.uiMaxParticles * sizeof(UINT),                           //UINT ByteWidth;
            D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
            D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_VERTEX_BUFFER,               //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            0,                                      //UINT MiscFlags;
            0                                       //UINT StructureByteStride;
        };

        m_pbufSerializedVisibleParticles.Release();
		m_pbufSerializedVisibleParticlesUAV.Release();

		V(pDevice->CreateBuffer( &SerializedParticlesBuffDesc, nullptr, &m_pbufSerializedVisibleParticles));

        D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
		UAVDesc.Format = DXGI_FORMAT_R32_UINT;
        UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        UAVDesc.Buffer.FirstElement = 0;
        UAVDesc.Buffer.NumElements = SerializedParticlesBuffDesc.ByteWidth / sizeof(UINT);
        UAVDesc.Buffer.Flags = 0;
        V(pDevice->CreateUnorderedAccessView( m_pbufSerializedVisibleParticles, &UAVDesc, &m_pbufSerializedVisibleParticlesUAV));
    }

	{
		m_ptex3DCellDensitySRV.Release();
		m_ptex3DCellDensityUAV.Release();
		m_ptex3DLightAttenuatingMassSRV.Release();
		m_ptex3DLightAttenuatingMassUAV.Release();
		D3D11_TEXTURE3D_DESC Tex3DDesc = 
		{
			m_CloudAttribs.uiRingDimension * m_CloudAttribs.uiDensityBufferScale, //UINT Width;
			m_CloudAttribs.uiRingDimension * m_CloudAttribs.uiDensityBufferScale, //UINT Height;
			m_CloudAttribs.uiMaxLayers * m_CloudAttribs.uiDensityBufferScale * m_CloudAttribs.uiNumRings,  //UINT Depth;
			1,							//UINT MipLevels;
			DXGI_FORMAT_R16_FLOAT,		//DXGI_FORMAT Format;
			D3D11_USAGE_DEFAULT,		//D3D11_USAGE Usage;
			D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, //UINT BindFlags;
			0,							//UINT CPUAccessFlags;
			0							//UINT MiscFlags;
		};

		// Accuracy of R8_UNORM is not sufficient to provide smooth animation
		Tex3DDesc.Format = DXGI_FORMAT_R16_FLOAT;
		CComPtr<ID3D11Texture3D> ptex3DCellDensity;
		V(pDevice->CreateTexture3D( &Tex3DDesc, nullptr, &ptex3DCellDensity));
		V(pDevice->CreateShaderResourceView( ptex3DCellDensity, nullptr, &m_ptex3DCellDensitySRV));
		V(pDevice->CreateUnorderedAccessView( ptex3DCellDensity, nullptr, &m_ptex3DCellDensityUAV));

		Tex3DDesc.Format = DXGI_FORMAT_R8_UNORM;
		CComPtr<ID3D11Texture3D> ptex3DLightAttenuatingMass;
		V(pDevice->CreateTexture3D( &Tex3DDesc, nullptr, &ptex3DLightAttenuatingMass));
		V(pDevice->CreateShaderResourceView( ptex3DLightAttenuatingMass, nullptr, &m_ptex3DLightAttenuatingMassSRV));
		V(pDevice->CreateUnorderedAccessView( ptex3DLightAttenuatingMass, nullptr, &m_ptex3DLightAttenuatingMassUAV));
	}

    return S_OK;
}

HRESULT CCloudsController::PrecomputParticleDensity(ID3D11Device *pDevice, ID3D11DeviceContext *pDeviceContext)
{
    HRESULT hr;
    int iNumStartPosZenithAngles  = m_PrecomputedOpticalDepthTexDim.iNumStartPosZenithAngles;
    int iNumStartPosAzimuthAngles = m_PrecomputedOpticalDepthTexDim.iNumStartPosAzimuthAngles;
    int iNumDirectionZenithAngles = m_PrecomputedOpticalDepthTexDim.iNumDirectionZenithAngles;
    int iNumDirectionAzimuthAngles= m_PrecomputedOpticalDepthTexDim.iNumDirectionAzimuthAngles;

    D3D11_TEXTURE3D_DESC PrecomputedOpticalDepthTexDesc = 
    {
        iNumStartPosZenithAngles,  //UINT Width;
        iNumStartPosAzimuthAngles,  //UINT Height;
        iNumDirectionZenithAngles * iNumDirectionAzimuthAngles,  //UINT Depth;
        5, //UINT MipLevels;
        DXGI_FORMAT_R8_UNORM,//DXGI_FORMAT_R8G8_UNORM,//DXGI_FORMAT Format;
        D3D11_USAGE_DEFAULT, //D3D11_USAGE Usage;
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,//UINT BindFlags;
        0,//UINT CPUAccessFlags;
        D3D11_RESOURCE_MISC_GENERATE_MIPS //UINT MiscFlags;
    };

    CComPtr<ID3D11Texture3D> ptex3DPrecomputedParticleDensity;
    V_RETURN( pDevice->CreateTexture3D(&PrecomputedOpticalDepthTexDesc, nullptr, &ptex3DPrecomputedParticleDensity));

    m_ptex3DPrecomputedParticleDensitySRV.Release();
    V_RETURN(pDevice->CreateShaderResourceView( ptex3DPrecomputedParticleDensity, nullptr, &m_ptex3DPrecomputedParticleDensitySRV));

    if( !m_ComputeOpticalDepthTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("DENSITY_GENERATION_METHOD", m_CloudAttribs.uiDensityGenerationMethod);
        Macros.Finalize();

        m_ComputeOpticalDepthTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_ComputeOpticalDepthTech.CreateVGPShadersFromFile(m_strPreprocessingEffectPath, "ScreenSizeQuadVS", nullptr, "PrecomputeOpticalDepthPS", Macros);
        m_ComputeOpticalDepthTech.SetDS( m_pdsDisableDepth );
        m_ComputeOpticalDepthTech.SetRS( m_prsSolidFillNoCull );
        m_ComputeOpticalDepthTech.SetBS( m_pbsDefault );
    }

    CComPtr<ID3D11RenderTargetView> pOrigRTV;
    CComPtr<ID3D11DepthStencilView> pOrigDSV;
    pDeviceContext->OMGetRenderTargets(1, &pOrigRTV, &pOrigDSV);
    
    D3D11_VIEWPORT OrigViewPort;
    UINT iNumOldViewports = 1;
    pDeviceContext->RSGetViewports(&iNumOldViewports, &OrigViewPort);

    ID3D11Buffer *pCBs[] = {m_pcbGlobalCloudAttribs/*, RenderAttribs.pcMediaScatteringParams*/};
    pDeviceContext->PSSetConstantBuffers(0, _countof(pCBs), pCBs);

    ID3D11SamplerState *pSamplers[] = {m_psamLinearClamp, m_psamLinearWrap, m_psamPointWrap};
    pDeviceContext->VSSetSamplers(0, _countof(pSamplers), pSamplers);
    pDeviceContext->PSSetSamplers(0, _countof(pSamplers), pSamplers);

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_ptex3DNoiseSRV,
    };
    
    for(UINT Slice = 0; Slice < PrecomputedOpticalDepthTexDesc.Depth; ++Slice)
    {
        D3D11_RENDER_TARGET_VIEW_DESC RTVDesc;
        RTVDesc.Format = PrecomputedOpticalDepthTexDesc.Format;
        RTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
        RTVDesc.Texture3D.MipSlice = 0;
        RTVDesc.Texture3D.FirstWSlice = Slice;
        RTVDesc.Texture3D.WSize = 1;

        CComPtr<ID3D11RenderTargetView> pSliceRTV;
        V_RETURN(pDevice->CreateRenderTargetView( ptex3DPrecomputedParticleDensity, &RTVDesc, &pSliceRTV));
        
        UINT uiDirectionZenith = Slice % iNumDirectionZenithAngles;
        UINT uiDirectionAzimuth= Slice / iNumDirectionZenithAngles;
        m_CloudAttribs.f4Parameter.x = ((float)uiDirectionZenith + 0.5f)  / (float)iNumDirectionZenithAngles;
        m_CloudAttribs.f4Parameter.y = ((float)uiDirectionAzimuth + 0.5f) / (float)iNumDirectionAzimuthAngles;
        assert(0 < m_CloudAttribs.f4Parameter.x && m_CloudAttribs.f4Parameter.x < 1);
        assert(0 < m_CloudAttribs.f4Parameter.y && m_CloudAttribs.f4Parameter.y < 1);
        UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));

        pDeviceContext->OMSetRenderTargets(1, &pSliceRTV.p, nullptr);
        
        pDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

        RenderQuad(pDeviceContext, m_ComputeOpticalDepthTech, PrecomputedOpticalDepthTexDesc.Width, PrecomputedOpticalDepthTexDesc.Height);
    }
	// TODO: need to use proper filtering for coarser mip levels
    pDeviceContext->GenerateMips( m_ptex3DPrecomputedParticleDensitySRV);

    pDeviceContext->OMSetRenderTargets(1, &pOrigRTV.p, pOrigDSV);
    pDeviceContext->RSSetViewports(iNumOldViewports, &OrigViewPort);

    return S_OK;
}

HRESULT CCloudsController::PrecomputeScatteringInParticle(ID3D11Device *pDevice, ID3D11DeviceContext *pDeviceContext)
{
    HRESULT hr;
    
    LPCTSTR SingleSctrTexPath   = L"media\\SingleSctr.dds";
    LPCTSTR MultipleSctrTexPath = L"media\\MultipleSctr.dds";
    HRESULT hr1 = D3DX11CreateShaderResourceViewFromFile(pDevice, SingleSctrTexPath, nullptr, nullptr, &m_ptex3DSingleSctrInParticleLUT_SRV, nullptr);
    HRESULT hr2 = D3DX11CreateShaderResourceViewFromFile(pDevice, MultipleSctrTexPath, nullptr, nullptr, &m_ptex3DMultipleSctrInParticleLUT_SRV, nullptr);
    if( SUCCEEDED(hr1) && SUCCEEDED(hr2) )
        return S_OK;

    D3D11_TEXTURE3D_DESC PrecomputedScatteringTexDesc = 
    {
        m_PrecomputedSctrInParticleLUTDim.iNumStartPosZenithAngles,  //UINT Width;
        m_PrecomputedSctrInParticleLUTDim.iNumViewDirAzimuthAngles,  //UINT Height;
        // We are only interested in rays going into the sphere, which is half of the total number of diections
        m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles/2,  //UINT Depth;
        1, //UINT MipLevels;
        DXGI_FORMAT_R16_FLOAT,//DXGI_FORMAT Format;
        D3D11_USAGE_DEFAULT, //D3D11_USAGE Usage;
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,//UINT BindFlags;
        0,//UINT CPUAccessFlags;
        0 //UINT MiscFlags;
    };

    CComPtr<ID3D11Texture3D> ptex3DSingleScatteringInParticleLUT, ptex3DMultipleScatteringInParticleLUT;
    V_RETURN( pDevice->CreateTexture3D(&PrecomputedScatteringTexDesc, nullptr, &ptex3DSingleScatteringInParticleLUT));
    V_RETURN( pDevice->CreateTexture3D(&PrecomputedScatteringTexDesc, nullptr, &ptex3DMultipleScatteringInParticleLUT));
    m_ptex3DSingleSctrInParticleLUT_SRV.Release();
    m_ptex3DMultipleSctrInParticleLUT_SRV.Release();
    V_RETURN(pDevice->CreateShaderResourceView( ptex3DSingleScatteringInParticleLUT,   nullptr, &m_ptex3DSingleSctrInParticleLUT_SRV));
    V_RETURN(pDevice->CreateShaderResourceView( ptex3DMultipleScatteringInParticleLUT, nullptr, &m_ptex3DMultipleSctrInParticleLUT_SRV));
    
    D3D11_TEXTURE3D_DESC TmpScatteringTexDesc = PrecomputedScatteringTexDesc;
    TmpScatteringTexDesc.Format = DXGI_FORMAT_R32_FLOAT;
    TmpScatteringTexDesc.Depth = m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles * m_PrecomputedSctrInParticleLUTDim.iNumDistancesFromCenter;

    CComPtr<ID3D11Texture3D> ptex3DSingleSctr, ptex3DGatheredScatteringN, ptex3DSctrOrderN, ptex3DMultipeScattering;
    V_RETURN( pDevice->CreateTexture3D(&TmpScatteringTexDesc, nullptr, &ptex3DSingleSctr));
    V_RETURN( pDevice->CreateTexture3D(&TmpScatteringTexDesc, nullptr, &ptex3DGatheredScatteringN));
    V_RETURN( pDevice->CreateTexture3D(&TmpScatteringTexDesc, nullptr, &ptex3DSctrOrderN));
    V_RETURN( pDevice->CreateTexture3D(&TmpScatteringTexDesc, nullptr, &ptex3DMultipeScattering));

    std::vector< CComPtr<ID3D11RenderTargetView> > ptex3DSingleSctrRTVs(TmpScatteringTexDesc.Depth);
    std::vector< CComPtr<ID3D11RenderTargetView> > ptex3DGatheredScatteringN_RTVs(TmpScatteringTexDesc.Depth);
    std::vector< CComPtr<ID3D11RenderTargetView> > ptex3DSctrOrderN_RTVs(TmpScatteringTexDesc.Depth);
    std::vector< CComPtr<ID3D11RenderTargetView> > ptex3DMultipeScatteringRTVs(TmpScatteringTexDesc.Depth);
    
    CComPtr<ID3D11ShaderResourceView> ptex3DSingleSctrSRV;
    CComPtr<ID3D11ShaderResourceView> ptex3DGatheredScatteringN_SRV;
    CComPtr<ID3D11ShaderResourceView> ptex3DSctrOrderN_SRV;
    CComPtr<ID3D11ShaderResourceView> ptex3DMultipeScatteringSRV;
    V_RETURN(pDevice->CreateShaderResourceView( ptex3DSingleSctr,          nullptr, &ptex3DSingleSctrSRV));
    V_RETURN(pDevice->CreateShaderResourceView( ptex3DGatheredScatteringN, nullptr, &ptex3DGatheredScatteringN_SRV));
    V_RETURN(pDevice->CreateShaderResourceView( ptex3DSctrOrderN,          nullptr, &ptex3DSctrOrderN_SRV));
    V_RETURN(pDevice->CreateShaderResourceView( ptex3DMultipeScattering,   nullptr, &ptex3DMultipeScatteringSRV));

    for(UINT Slice = 0; Slice < TmpScatteringTexDesc.Depth; ++Slice)
    {
        D3D11_RENDER_TARGET_VIEW_DESC RTVDesc;
        RTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
        RTVDesc.Texture3D.MipSlice = 0;
        RTVDesc.Texture3D.FirstWSlice = Slice;
        RTVDesc.Texture3D.WSize = 1;
        RTVDesc.Format = TmpScatteringTexDesc.Format;
        V_RETURN(pDevice->CreateRenderTargetView( ptex3DSingleSctr,          &RTVDesc, &ptex3DSingleSctrRTVs[Slice])          );
        V_RETURN(pDevice->CreateRenderTargetView( ptex3DGatheredScatteringN, &RTVDesc, &ptex3DGatheredScatteringN_RTVs[Slice]));
        V_RETURN(pDevice->CreateRenderTargetView( ptex3DSctrOrderN,          &RTVDesc, &ptex3DSctrOrderN_RTVs[Slice])         );
        V_RETURN(pDevice->CreateRenderTargetView( ptex3DMultipeScattering,   &RTVDesc, &ptex3DMultipeScatteringRTVs[Slice])   );
    }

    
    if( !m_ComputeSingleSctrInParticleTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_ComputeSingleSctrInParticleTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_ComputeSingleSctrInParticleTech.CreateVGPShadersFromFile(m_strPreprocessingEffectPath, "ScreenSizeQuadVS", nullptr, "PrecomputeSingleSctrPS", Macros);
        m_ComputeSingleSctrInParticleTech.SetDS( m_pdsDisableDepth );
        m_ComputeSingleSctrInParticleTech.SetRS( m_prsSolidFillNoCull );
        m_ComputeSingleSctrInParticleTech.SetBS( m_pbsDefault );
    }

    if( !m_RenderScatteringLUTSliceTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_RenderScatteringLUTSliceTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_RenderScatteringLUTSliceTech.CreateVGPShadersFromFile(m_strPreprocessingEffectPath, "ScreenSizeQuadVS", nullptr, "RenderScatteringLUTSlicePS", Macros);
        m_RenderScatteringLUTSliceTech.SetDS( m_pdsDisableDepth );
        m_RenderScatteringLUTSliceTech.SetRS( m_prsSolidFillNoCull );
        m_RenderScatteringLUTSliceTech.SetBS( m_pbsDefault );
    }

    if( !m_GatherPrevSctrOrderTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_GatherPrevSctrOrderTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_GatherPrevSctrOrderTech.CreateVGPShadersFromFile(m_strPreprocessingEffectPath, "ScreenSizeQuadVS", nullptr, "GatherScatteringPS", Macros);
        m_GatherPrevSctrOrderTech.SetDS( m_pdsDisableDepth );
        m_GatherPrevSctrOrderTech.SetRS( m_prsSolidFillNoCull );
        m_GatherPrevSctrOrderTech.SetBS( m_pbsDefault );
    }
     
    if( !m_ComputeScatteringOrderTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_ComputeScatteringOrderTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_ComputeScatteringOrderTech.CreateVGPShadersFromFile(m_strPreprocessingEffectPath, "ScreenSizeQuadVS", nullptr, "ComputeScatteringOrderPS", Macros);
        m_ComputeScatteringOrderTech.SetDS( m_pdsDisableDepth );
        m_ComputeScatteringOrderTech.SetRS( m_prsSolidFillNoCull );
        m_ComputeScatteringOrderTech.SetBS( m_pbsDefault );
    }

    if( !m_AccumulateInscatteringTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_AccumulateInscatteringTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_AccumulateInscatteringTech.CreateVGPShadersFromFile(m_strPreprocessingEffectPath, "ScreenSizeQuadVS", nullptr, "AccumulateMultipleScattering", Macros);
        m_AccumulateInscatteringTech.SetDS( m_pdsDisableDepth );
        m_AccumulateInscatteringTech.SetRS( m_prsSolidFillNoCull );

        D3D11_BLEND_DESC AdditiveBlendStateDesc;
        ZeroMemory(&AdditiveBlendStateDesc, sizeof(AdditiveBlendStateDesc));
        AdditiveBlendStateDesc.IndependentBlendEnable = FALSE;
        for(int i=0; i< _countof(AdditiveBlendStateDesc.RenderTarget); i++)
            AdditiveBlendStateDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        AdditiveBlendStateDesc.RenderTarget[0].BlendEnable = TRUE;
        AdditiveBlendStateDesc.RenderTarget[0].BlendOp     = D3D11_BLEND_OP_ADD;
        AdditiveBlendStateDesc.RenderTarget[0].BlendOpAlpha= D3D11_BLEND_OP_ADD;
        AdditiveBlendStateDesc.RenderTarget[0].DestBlend   = D3D11_BLEND_ONE;
        AdditiveBlendStateDesc.RenderTarget[0].DestBlendAlpha= D3D11_BLEND_ONE;
        AdditiveBlendStateDesc.RenderTarget[0].SrcBlend     = D3D11_BLEND_ONE;
        AdditiveBlendStateDesc.RenderTarget[0].SrcBlendAlpha= D3D11_BLEND_ONE;
        CComPtr<ID3D11BlendState> pAdditiveBlendBS;
        V_RETURN( pDevice->CreateBlendState( &AdditiveBlendStateDesc, &pAdditiveBlendBS) );
        m_AccumulateInscatteringTech.SetBS( pAdditiveBlendBS );
    }

    CComPtr<ID3D11RenderTargetView> pOrigRTV;
    CComPtr<ID3D11DepthStencilView> pOrigDSV;
    pDeviceContext->OMGetRenderTargets(1, &pOrigRTV, &pOrigDSV);
    
    D3D11_VIEWPORT OrigViewPort;
    UINT iNumOldViewports = 1;
    pDeviceContext->RSGetViewports(&iNumOldViewports, &OrigViewPort);

    ID3D11Buffer *pCBs[] = {m_pcbGlobalCloudAttribs/*, RenderAttribs.pcMediaScatteringParams*/};
    pDeviceContext->PSSetConstantBuffers(0, _countof(pCBs), pCBs);

    ID3D11SamplerState *pSamplers[] = {m_psamLinearClamp, m_psamLinearWrap, m_psamPointWrap};
    pDeviceContext->VSSetSamplers(0, _countof(pSamplers), pSamplers);
    pDeviceContext->PSSetSamplers(0, _countof(pSamplers), pSamplers);

    for(UINT Slice = 0; Slice < TmpScatteringTexDesc.Depth; ++Slice)
    {
        float Zero[4]={0,0,0,0};
        pDeviceContext->ClearRenderTargetView(ptex3DMultipeScatteringRTVs[Slice], Zero);
    }

    // Precompute single scattering
    for(UINT Slice = 0; Slice < TmpScatteringTexDesc.Depth; ++Slice)
    {
        UINT uiViewDirZenith = Slice % m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles;
        UINT uiDistFromCenter = Slice / m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles;
        m_CloudAttribs.f4Parameter.x = ((float)uiViewDirZenith + 0.5f)  / (float)m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles;
        m_CloudAttribs.f4Parameter.y = ((float)uiDistFromCenter + 0.5f) / (float)m_PrecomputedSctrInParticleLUTDim.iNumDistancesFromCenter;
        assert(0 < m_CloudAttribs.f4Parameter.x && m_CloudAttribs.f4Parameter.x < 1);
        assert(0 < m_CloudAttribs.f4Parameter.y && m_CloudAttribs.f4Parameter.y < 1);
        UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));

        ID3D11RenderTargetView *pSliceRTV = ptex3DSingleSctrRTVs[Slice];
        pDeviceContext->OMSetRenderTargets(1, &pSliceRTV, nullptr);
        
        ID3D11ShaderResourceView *pSRVs[] = 
        {
            m_ptex3DNoiseSRV,
        };

        pDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

        RenderQuad(pDeviceContext, m_ComputeSingleSctrInParticleTech, TmpScatteringTexDesc.Width, TmpScatteringTexDesc.Height);
    }
    
    // Number of scattering orders is chosen so as to obtain reasonable exitance through the particle surface
    const int iMaxScatteringOrder = 18;
    for(int iSctrOrder = 1; iSctrOrder < iMaxScatteringOrder; ++iSctrOrder)
    {
        for(int iPass = 0; iPass < 3; ++iPass)
        {
            // Gather scattering of previous order
            for(UINT Slice = 0; Slice < TmpScatteringTexDesc.Depth; ++Slice)
            {
                if( iPass < 2 )
                {
                    UINT uiViewDirZenith = Slice % m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles;
                    UINT uiDistFromCenter = Slice / m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles;
                    m_CloudAttribs.f4Parameter.x = ((float)uiViewDirZenith + 0.5f)  / (float)m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles;
                    m_CloudAttribs.f4Parameter.y = ((float)uiDistFromCenter + 0.5f) / (float)m_PrecomputedSctrInParticleLUTDim.iNumDistancesFromCenter;
                    assert(0 < m_CloudAttribs.f4Parameter.x && m_CloudAttribs.f4Parameter.x < 1);
                    assert(0 < m_CloudAttribs.f4Parameter.y && m_CloudAttribs.f4Parameter.y < 1);
                    m_CloudAttribs.f4Parameter.w = (float)iSctrOrder;
                }
                else
                {
                    m_CloudAttribs.f4Parameter.x = ((float)Slice + 0.5f) / (float)TmpScatteringTexDesc.Depth;
                    assert(0 < m_CloudAttribs.f4Parameter.x && m_CloudAttribs.f4Parameter.x < 1);
                }
                UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));

                ID3D11RenderTargetView *pSliceRTV = nullptr;
                CRenderTechnique *pTechnique = nullptr;
                ID3D11ShaderResourceView *pSRVs[1] = {nullptr};
                switch(iPass)
                {
                    // Gather scattering of previous order
                    case 0: 
                        pSRVs[0] = iSctrOrder > 1 ? ptex3DSctrOrderN_SRV : ptex3DSingleSctrSRV;
                        pSliceRTV = ptex3DGatheredScatteringN_RTVs[Slice];
                        pTechnique = &m_GatherPrevSctrOrderTech;
                    break;

                    // Compute current scattering order
                    case 1: 
                        pSRVs[0] = ptex3DGatheredScatteringN_SRV;
                        pSliceRTV = ptex3DSctrOrderN_RTVs[Slice];
                        pTechnique = &m_ComputeScatteringOrderTech;
                    break;

                    // Accumulate current scattering order
                    case 2: 
                        pSRVs[0] = ptex3DSctrOrderN_SRV;
                        pSliceRTV = ptex3DMultipeScatteringRTVs[Slice];
                        pTechnique = &m_AccumulateInscatteringTech;
                    break;
                }

                pDeviceContext->OMSetRenderTargets(1, &pSliceRTV, nullptr);
                pDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

                RenderQuad(pDeviceContext, *pTechnique, TmpScatteringTexDesc.Width, TmpScatteringTexDesc.Height);
            }
        }
    }

    // Copy single and multiple scattering to the textures
    for(UINT Slice = 0; Slice < PrecomputedScatteringTexDesc.Depth; ++Slice)
    {
        D3D11_RENDER_TARGET_VIEW_DESC RTVDesc;
        RTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
        RTVDesc.Texture3D.MipSlice = 0;
        RTVDesc.Texture3D.FirstWSlice = Slice;
        RTVDesc.Texture3D.WSize = 1;
        RTVDesc.Format = PrecomputedScatteringTexDesc.Format;
        CComPtr<ID3D11RenderTargetView> pSingleSctrSliceRTV, pMultSctrSliceRTV;
        V_RETURN(pDevice->CreateRenderTargetView( ptex3DSingleScatteringInParticleLUT, &RTVDesc, &pSingleSctrSliceRTV));
        V_RETURN(pDevice->CreateRenderTargetView( ptex3DMultipleScatteringInParticleLUT, &RTVDesc, &pMultSctrSliceRTV));

        m_CloudAttribs.f4Parameter.x = ((float)Slice + 0.5f)  / (float)PrecomputedScatteringTexDesc.Depth;
        UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));

        ID3D11RenderTargetView *pRTVs[] = {pSingleSctrSliceRTV, pMultSctrSliceRTV};
        pDeviceContext->OMSetRenderTargets(_countof(pRTVs), pRTVs, nullptr);

        ID3D11ShaderResourceView *pSRVs[] = 
        {
            ptex3DSingleSctrSRV,
            ptex3DMultipeScatteringSRV
        };
        pDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

        RenderQuad(pDeviceContext, m_RenderScatteringLUTSliceTech, PrecomputedScatteringTexDesc.Width, PrecomputedScatteringTexDesc.Height);
    }

    D3DX11SaveTextureToFile(pDeviceContext, ptex3DSingleScatteringInParticleLUT, D3DX11_IFF_DDS, SingleSctrTexPath);
    D3DX11SaveTextureToFile(pDeviceContext, ptex3DMultipleScatteringInParticleLUT, D3DX11_IFF_DDS, MultipleSctrTexPath);

    pDeviceContext->OMSetRenderTargets(1, &pOrigRTV.p, pOrigDSV);
    pDeviceContext->RSSetViewports(iNumOldViewports, &OrigViewPort);

    return S_OK;
}

float CubicInterpolate(float ym1, float y0, float y1, float y2, float x)
{
    float b0 = 0*ym1 + 6*y0 + 0*y1 + 0*y2;
    float b1 =-2*ym1 - 3*y0 + 6*y1 - 1*y2;
    float b2 = 3*ym1 - 6*y0 + 3*y1 + 0*y2;
    float b3 =-1*ym1 + 3*y0 - 3*y1 + 1*y2;
    float x2 = x*x;
    float x3 = x2*x;
    return 1.f/6.f * (b0 + x*b1 + x2*b2 + x3*b3);
}

HRESULT CCloudsController::Create3DNoise(ID3D11Device *pDevice)
{
    HRESULT hr;
    // Create 3D noise
    UINT uiMips = 8;
    UINT uiDim = 1 << (uiMips-1);
    D3D11_TEXTURE3D_DESC NoiseTexDesc = 
    {
        uiDim,  //UINT Width;
        uiDim,  //UINT Height;
        uiDim,  //UINT Depth;
        uiMips, //UINT MipLevels;
        DXGI_FORMAT_R8_UNORM,//DXGI_FORMAT Format;
        D3D11_USAGE_DEFAULT, //D3D11_USAGE Usage;
        D3D11_BIND_SHADER_RESOURCE,//UINT BindFlags;
        0,//UINT CPUAccessFlags;
        0//UINT MiscFlags;
    };
    size_t DataSize = 0;
    for(UINT Mip=0; Mip < uiMips; ++Mip)
        DataSize += (NoiseTexDesc.Width>>Mip) * (NoiseTexDesc.Height>>Mip) * (NoiseTexDesc.Depth>>Mip);
    std::vector<float> NoiseData(DataSize);

    #define NOISE(i,j,k) NoiseData[i + j * NoiseTexDesc.Width + k * (NoiseTexDesc.Width * NoiseTexDesc.Height)]

    // Populate texture with random noise
    UINT InitialStep = 8;
    for(UINT i=0; i < NoiseTexDesc.Width; i+=InitialStep)
        for(UINT j=0; j < NoiseTexDesc.Height; j+=InitialStep)
            for(UINT k=0; k < NoiseTexDesc.Depth; k+=InitialStep)
                NOISE(i,j,k) = (float)rand() / (float)RAND_MAX;

    // Smooth rows
    for(UINT i=0; i < NoiseTexDesc.Width; ++i)
        for(UINT j=0; j < NoiseTexDesc.Height; j+=InitialStep)
            for(UINT k=0; k < NoiseTexDesc.Depth; k+=InitialStep)
            {
                int i0 = (i/InitialStep)*InitialStep;
                int im1 = i0-InitialStep;
                if( im1 < 0 )im1 += NoiseTexDesc.Width;
                int i1 = (i0+InitialStep) % NoiseTexDesc.Width;
                int i2 = (i0+2*InitialStep) % NoiseTexDesc.Width;
                NOISE(i,j,k) = CubicInterpolate( NOISE(im1,j,k), NOISE(i0,j,k), NOISE(i1,j,k), NOISE(i2,j,k), (float)(i-i0) / (float)InitialStep );
            }

    // Smooth columns
    for(UINT i=0; i < NoiseTexDesc.Width; ++i)
        for(UINT j=0; j < NoiseTexDesc.Height; ++j)
            for(UINT k=0; k < NoiseTexDesc.Depth; k+=InitialStep)
            {
                int j0 = (j/InitialStep)*InitialStep;
                int jm1 = j0 - InitialStep;
                if( jm1 < 0 )jm1+=NoiseTexDesc.Height;
                int j1 = (j0+InitialStep) % NoiseTexDesc.Height;
                int j2 = (j0+2*InitialStep) % NoiseTexDesc.Height;
                NOISE(i,j,k) = CubicInterpolate(NOISE(i,jm1,k), NOISE(i,j0,k), NOISE(i,j1,k), NOISE(i,j2,k), (float)(j-j0) / (float)InitialStep);
            }

    // Smooth in depth direction
    for(UINT i=0; i < NoiseTexDesc.Width; ++i)
        for(UINT j=0; j < NoiseTexDesc.Height; ++j)
            for(UINT k=0; k < NoiseTexDesc.Depth; ++k)
            {
                int k0 = (k/InitialStep)*InitialStep;
                int km1 = k0-InitialStep;
                if( km1 < 0 )km1+=NoiseTexDesc.Depth;
                int k1 = (k0+InitialStep) % NoiseTexDesc.Depth;
                int k2 = (k0+2*InitialStep) % NoiseTexDesc.Depth;
                NOISE(i,j,k) = CubicInterpolate(NOISE(i,j,km1), NOISE(i,j,k0), NOISE(i,j,k1), NOISE(i,j,k2), (float)(k-k0) / (float)InitialStep);
            }
    
    // Generate mips
    auto FinerMipIt = NoiseData.begin();
    for(uint Mip = 1; Mip < uiMips; ++Mip)
    {
        UINT uiFinerMipWidth  = NoiseTexDesc.Width  >> (Mip-1);
        UINT uiFinerMipHeight = NoiseTexDesc.Height >> (Mip-1);
        UINT uiFinerMipDepth  = NoiseTexDesc.Depth  >> (Mip-1);

        auto CurrMipIt = FinerMipIt + uiFinerMipWidth * uiFinerMipHeight * uiFinerMipDepth;
        UINT uiMipWidth  = NoiseTexDesc.Width  >> Mip;
        UINT uiMipHeight = NoiseTexDesc.Height >> Mip;
        UINT uiMipDepth  = NoiseTexDesc.Depth  >> Mip;
        for(UINT i=0; i < uiMipWidth; ++i)
            for(UINT j=0; j < uiMipHeight; ++j)
                for(UINT k=0; k < uiMipDepth; ++k)
                {
                    float fVal=0;
                    for(int x=0; x<2;++x)
                        for(int y=0; y<2;++y)
                            for(int z=0; z<2;++z)
                            {
                                fVal += FinerMipIt[(i*2+x) + (j*2 + y) * uiFinerMipWidth + (k*2+z) * (uiFinerMipWidth * uiFinerMipHeight)];
                            }
                    CurrMipIt[i + j * uiMipWidth + k * (uiMipWidth * uiMipHeight)] = fVal / 8.f;
                }
        FinerMipIt = CurrMipIt;
    }
    assert(FinerMipIt+1 == NoiseData.end());

    // Convert to 8-bit
    std::vector<BYTE> NoiseDataR8(NoiseData.size());
    for(auto it=NoiseData.begin(); it != NoiseData.end(); ++it)
        NoiseDataR8[it-NoiseData.begin()] = (BYTE)min(max((int)( *it*255.f), 0),255);

    // Prepare init data
    std::vector<D3D11_SUBRESOURCE_DATA>InitData(uiMips);
    auto CurrMipIt = NoiseDataR8.begin();
    for( UINT Mip = 0; Mip < uiMips; ++Mip )
    {
        UINT uiMipWidth  = NoiseTexDesc.Width  >> Mip;
        UINT uiMipHeight = NoiseTexDesc.Height >> Mip;
        UINT uiMipDepth  = NoiseTexDesc.Depth  >> Mip;
        InitData[Mip].pSysMem = &(*CurrMipIt);
        InitData[Mip].SysMemPitch = uiMipWidth*sizeof(NoiseDataR8[0]);
        InitData[Mip].SysMemSlicePitch = uiMipWidth*uiMipHeight*sizeof(NoiseDataR8[0]);
        CurrMipIt += uiMipWidth * uiMipHeight * uiMipDepth;
    }
    assert(CurrMipIt == NoiseDataR8.end());
    
    // TODO: compress to BC1

    // Create 3D texture
    CComPtr<ID3D11Texture3D> ptex3DNoise;
    V( pDevice->CreateTexture3D(&NoiseTexDesc, &InitData[0], &ptex3DNoise));
    V( pDevice->CreateShaderResourceView(ptex3DNoise, nullptr, &m_ptex3DNoiseSRV));

    return S_OK;
}

HRESULT CCloudsController::OnCreateDevice(ID3D11Device *pDevice, ID3D11DeviceContext *pDeviceContext)
{
    HRESULT hr;

    // Detect and report Intel extensions on this system
    hr = IGFX::Init( pDevice );
	if ( FAILED(hr) )
	{
		//CPUTOSServices::GetOSServices()->OpenMessageBox( _L("Error"), _L("Failed hardware detection initialization: incorrect vendor or device.\n\n") );
	}
    // detect the available extensions
    IGFX::Extensions extensions = IGFX::getAvailableExtensions( pDevice );

    m_bPSOrderingAvailable = extensions.PixelShaderOrdering;  
    
    // Disable the AVSM extension method if the hardware/driver does not support Pixel Shader Ordering feature
    if ( !extensions.PixelShaderOrdering ) 
    {
        CPUTOSServices::GetOSServices()->OpenMessageBox(_L("Pixel Shader Ordering feature not found"), _L("Your hardware or graphics driver does not support the pixel shader ordering feature. Volume-aware blending will be disabled. Please update your driver or run on a system that supports the required feature to see that option."));      
    }

    CreateParticleDataBuffer(pDevice);

    // Create buffer for storing number of valid cells
    {
	    D3D11_BUFFER_DESC ValidCellsCounterBuffDesc = 
        {
            sizeof(UINT)*4,                           //UINT ByteWidth;
            D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
            D3D11_BIND_SHADER_RESOURCE,             //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            0,                                      //UINT MiscFlags;
            0	            						//UINT StructureByteStride;
        };
        V( CreateBufferAndViews( pDevice, ValidCellsCounterBuffDesc, nullptr, &m_pbufValidCellsCounter) );
		V(CreateBufferAndViews( pDevice, ValidCellsCounterBuffDesc, nullptr, &m_pbufVisibleParticlesCounter));
        
        D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        ZeroMemory(&SRVDesc, sizeof(SRVDesc));
        SRVDesc.Format =  DXGI_FORMAT_R32_UINT;
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        SRVDesc.Buffer.ElementOffset = 0;
        SRVDesc.Buffer.ElementWidth = sizeof(UINT);
        V_RETURN(pDevice->CreateShaderResourceView( m_pbufValidCellsCounter, &SRVDesc, &m_pbufValidCellsCounterSRV));
		V_RETURN(pDevice->CreateShaderResourceView( m_pbufVisibleParticlesCounter, &SRVDesc, &m_pbufVisibleParticlesCounterSRV));
    }
    
    // Create buffer for storing DispatchIndirect() arguments
    {
        UINT DispatchArgs[] = 
        {
            1, // UINT ThreadGroupCountX
            1, // UINT ThreadGroupCountY
            1, // UINT ThreadGroupCountZ
        };

	    D3D11_BUFFER_DESC DispatchArgsBuffDesc = 
        {
            sizeof(DispatchArgs),                   //UINT ByteWidth;
            D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
            D3D11_BIND_UNORDERED_ACCESS,            //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS,  //UINT MiscFlags;
            0                                       //UINT StructureByteStride;
        };

        D3D11_SUBRESOURCE_DATA InitData = {&DispatchArgs, 0, 0};
        V( CreateBufferAndViews( pDevice, DispatchArgsBuffDesc, &InitData, &m_pbufDispatchArgs, nullptr, nullptr) );

        D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
        UAVDesc.Format = DXGI_FORMAT_R32_UINT;
        UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        UAVDesc.Buffer.FirstElement = 0;
        UAVDesc.Buffer.NumElements = _countof(DispatchArgs);
        UAVDesc.Buffer.Flags = 0;
        V_RETURN(pDevice->CreateUnorderedAccessView( m_pbufDispatchArgs, &UAVDesc, &m_pbufDispatchArgsUAV));
    }

    // Create buffer for storing DrawIndirect() arguments
    {
        UINT DrawInstancedArgs[] = 
        {
            0, // UINT VertexCountPerInstance,
            1, // UINT InstanceCount,
            0, // StartVertexLocation,
            0  // StartInstanceLocation
        };

	    D3D11_BUFFER_DESC DrawArgsBuffDesc = 
        {
            sizeof(DrawInstancedArgs),              //UINT ByteWidth;
            D3D11_USAGE_DEFAULT,                    //D3D11_USAGE Usage;
            0,                                      //UINT BindFlags;
            0,                                      //UINT CPUAccessFlags;
            D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS,  //UINT MiscFlags;
            0                                       //UINT StructureByteStride;
        };

        D3D11_SUBRESOURCE_DATA InitData = {&DrawInstancedArgs, 0, 0};
        V( CreateBufferAndViews( pDevice, DrawArgsBuffDesc, &InitData, &m_pbufDrawIndirectArgs, nullptr, nullptr) );
    }

    D3D11_BUFFER_DESC GlobalCloudAttribsCBDesc = 
    {
        sizeof(SGlobalCloudAttribs), //UINT ByteWidth;
        D3D11_USAGE_DYNAMIC,         //D3D11_USAGE Usage;
        D3D11_BIND_CONSTANT_BUFFER,  //UINT BindFlags;
        D3D11_CPU_ACCESS_WRITE,      //UINT CPUAccessFlags;
        0,                                      //UINT MiscFlags;
        0                                       //UINT StructureByteStride;
    };
    V(pDevice->CreateBuffer( &GlobalCloudAttribsCBDesc, nullptr, &m_pcbGlobalCloudAttribs));
    
    // Create depth stencil states
    D3D11_DEPTH_STENCIL_DESC EnableDepthTestDSDesc;
    ZeroMemory(&EnableDepthTestDSDesc, sizeof(EnableDepthTestDSDesc));
    EnableDepthTestDSDesc.DepthEnable = TRUE;
    EnableDepthTestDSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    EnableDepthTestDSDesc.DepthFunc = D3D11_COMPARISON_GREATER;
    V( pDevice->CreateDepthStencilState(  &EnableDepthTestDSDesc, &m_pdsEnableDepth) );

    D3D11_DEPTH_STENCIL_DESC DisableDepthTestDSDesc;
    ZeroMemory(&DisableDepthTestDSDesc, sizeof(DisableDepthTestDSDesc));
    DisableDepthTestDSDesc.DepthEnable = FALSE;
    DisableDepthTestDSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    DisableDepthTestDSDesc.DepthFunc = D3D11_COMPARISON_GREATER;
    V( pDevice->CreateDepthStencilState(  &DisableDepthTestDSDesc, &m_pdsDisableDepth) );
    
    // Create rasterizer states
    D3D11_RASTERIZER_DESC SolidFillCullBackRSDesc;
    ZeroMemory(&SolidFillCullBackRSDesc, sizeof(SolidFillCullBackRSDesc));
    SolidFillCullBackRSDesc.FillMode = D3D11_FILL_SOLID;
    SolidFillCullBackRSDesc.CullMode = D3D11_CULL_FRONT;
    SolidFillCullBackRSDesc.DepthClipEnable = FALSE; // TODO: temp
    V( pDevice->CreateRasterizerState( &SolidFillCullBackRSDesc, &m_prsSolidFillCullFront) );

    D3D11_RASTERIZER_DESC SolidFillNoCullRSDesc;
    ZeroMemory(&SolidFillNoCullRSDesc, sizeof(SolidFillNoCullRSDesc));
    SolidFillNoCullRSDesc.FillMode = D3D11_FILL_SOLID;
    SolidFillNoCullRSDesc.CullMode = D3D11_CULL_NONE;
    SolidFillNoCullRSDesc.DepthClipEnable = TRUE;
    V( pDevice->CreateRasterizerState( &SolidFillNoCullRSDesc, &m_prsSolidFillNoCull) );
   
    // Create default blend state
    D3D11_BLEND_DESC DefaultBlendStateDesc;
    ZeroMemory(&DefaultBlendStateDesc, sizeof(DefaultBlendStateDesc));
    DefaultBlendStateDesc.IndependentBlendEnable = FALSE;
    for(int i=0; i< _countof(DefaultBlendStateDesc.RenderTarget); i++)
        DefaultBlendStateDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    V( pDevice->CreateBlendState( &DefaultBlendStateDesc, &m_pbsDefault) );

    // Create blend state for rendering particles
    D3D11_BLEND_DESC AlphaBlendStateDesc;
    ZeroMemory(&AlphaBlendStateDesc, sizeof(AlphaBlendStateDesc));
    AlphaBlendStateDesc.IndependentBlendEnable = TRUE;
    for(int i=0; i< _countof(AlphaBlendStateDesc.RenderTarget); i++)
        AlphaBlendStateDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    AlphaBlendStateDesc.RenderTarget[0].BlendEnable = TRUE;
    AlphaBlendStateDesc.RenderTarget[0].BlendOp        = D3D11_BLEND_OP_ADD;
    AlphaBlendStateDesc.RenderTarget[0].SrcBlend       = D3D11_BLEND_ZERO;
    AlphaBlendStateDesc.RenderTarget[0].DestBlend      = D3D11_BLEND_SRC_COLOR;

    AlphaBlendStateDesc.RenderTarget[0].BlendOpAlpha  = D3D11_BLEND_OP_ADD;
    AlphaBlendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
    AlphaBlendStateDesc.RenderTarget[0].DestBlendAlpha= D3D11_BLEND_SRC_ALPHA;

    AlphaBlendStateDesc.RenderTarget[1].BlendEnable    = TRUE;
    AlphaBlendStateDesc.RenderTarget[1].BlendOp        = D3D11_BLEND_OP_MIN;
    AlphaBlendStateDesc.RenderTarget[1].SrcBlend       = D3D11_BLEND_ONE;
    AlphaBlendStateDesc.RenderTarget[1].DestBlend      = D3D11_BLEND_ONE;
                                     
    AlphaBlendStateDesc.RenderTarget[1].BlendOpAlpha  = D3D11_BLEND_OP_MIN;
    AlphaBlendStateDesc.RenderTarget[1].SrcBlendAlpha = D3D11_BLEND_ONE;
    AlphaBlendStateDesc.RenderTarget[1].DestBlendAlpha= D3D11_BLEND_ONE;

    AlphaBlendStateDesc.RenderTarget[2].BlendEnable = TRUE;
    AlphaBlendStateDesc.RenderTarget[2].BlendOp        = D3D11_BLEND_OP_ADD;
    AlphaBlendStateDesc.RenderTarget[2].SrcBlend       = D3D11_BLEND_ONE;
    AlphaBlendStateDesc.RenderTarget[2].DestBlend      = D3D11_BLEND_SRC_ALPHA;
                                     
    AlphaBlendStateDesc.RenderTarget[2].BlendOpAlpha   = D3D11_BLEND_OP_ADD;
    AlphaBlendStateDesc.RenderTarget[2].SrcBlendAlpha  = D3D11_BLEND_ONE;
    AlphaBlendStateDesc.RenderTarget[2].DestBlendAlpha = D3D11_BLEND_ONE;

    V( pDevice->CreateBlendState( &AlphaBlendStateDesc, &m_pbsRT0MulRT1MinRT2Over) );

    D3D11_SAMPLER_DESC SamLinearWrap = 
    {
        D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        D3D11_TEXTURE_ADDRESS_WRAP,
        D3D11_TEXTURE_ADDRESS_WRAP,
        D3D11_TEXTURE_ADDRESS_WRAP,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_NEVER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    V( pDevice->CreateSamplerState( &SamLinearWrap, &m_psamLinearWrap) );

    D3D11_SAMPLER_DESC SamPointWrap = SamLinearWrap;
    SamPointWrap.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    V( pDevice->CreateSamplerState( &SamPointWrap, &m_psamPointWrap) );
    

    SamLinearWrap.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    SamLinearWrap.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    SamLinearWrap.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    V( pDevice->CreateSamplerState( &SamLinearWrap, &m_psamLinearClamp) );

    D3DX11_IMAGE_LOAD_INFO LoadInfo;
    ZeroMemory(&LoadInfo, sizeof(D3DX11_IMAGE_LOAD_INFO));
    LoadInfo.Width          = D3DX11_FROM_FILE;
    LoadInfo.Height         = D3DX11_FROM_FILE;
    LoadInfo.Depth          = D3DX11_FROM_FILE;
    LoadInfo.FirstMipLevel  = D3DX11_FROM_FILE;
    LoadInfo.MipLevels      = D3DX11_DEFAULT;
    LoadInfo.Usage          = D3D11_USAGE_IMMUTABLE;
    LoadInfo.BindFlags      = D3D11_BIND_SHADER_RESOURCE;
    LoadInfo.CpuAccessFlags = 0;
    LoadInfo.MiscFlags      = 0;
    LoadInfo.MipFilter      = D3DX11_FILTER_LINEAR;
    LoadInfo.pSrcInfo       = NULL;
    LoadInfo.Format         = DXGI_FORMAT_BC4_UNORM;
    LoadInfo.Filter         = D3DX11_FILTER_LINEAR;
    
    // Load noise textures. Important to use BC4 compression
    LoadInfo.Format         = DXGI_FORMAT_BC4_UNORM;
    D3DX11CreateShaderResourceViewFromFile(pDevice, L"media\\Noise.png", &LoadInfo, nullptr, &m_ptex2DCloudDensitySRV, nullptr);

    // Noise is not compressed well. Besides, it seems like there are some strange unstable results when using BC1 (?)
    LoadInfo.Format         = DXGI_FORMAT_R8G8B8A8_UNORM;//DXGI_FORMAT_BC1_UNORM;
    D3DX11CreateShaderResourceViewFromFile(pDevice, L"media\\WhiteNoise.png", &LoadInfo, nullptr, &m_ptex2DWhiteNoiseSRV, nullptr);
    
    {
        // Create maximum density mip map
        CComPtr<ID3D11Resource> pCloudDensityRes;
        m_ptex2DCloudDensitySRV->GetResource(&pCloudDensityRes);
        D3D11_TEXTURE2D_DESC CloudDensityTexDesc;
        CComQIPtr<ID3D11Texture2D>(pCloudDensityRes)->GetDesc(&CloudDensityTexDesc);
        m_uiCloudDensityTexWidth = CloudDensityTexDesc.Width;
        m_uiCloudDensityTexHeight = CloudDensityTexDesc.Height;

        D3D11_TEXTURE2D_DESC MaxCloudDensityMipDesc = CloudDensityTexDesc;
        MaxCloudDensityMipDesc.Format = DXGI_FORMAT_R8_UNORM;
        MaxCloudDensityMipDesc.Usage = D3D11_USAGE_DEFAULT;
        MaxCloudDensityMipDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        CComPtr<ID3D11Texture2D> ptex2DMaxDensityMipMap, ptex2DTmpMaxDensityMipMap;
        V(pDevice->CreateTexture2D(&MaxCloudDensityMipDesc, nullptr, &ptex2DMaxDensityMipMap));
        V(pDevice->CreateShaderResourceView(ptex2DMaxDensityMipMap, nullptr, &m_ptex2DMaxDensityMipMapSRV));

        MaxCloudDensityMipDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
        V(pDevice->CreateTexture2D(&MaxCloudDensityMipDesc, nullptr, &ptex2DTmpMaxDensityMipMap));

        RenderMaxDensityMip( pDevice, pDeviceContext, 
                             ptex2DMaxDensityMipMap, ptex2DTmpMaxDensityMipMap, 
                             MaxCloudDensityMipDesc );
    }

    Create3DNoise(pDevice);

    return S_OK;
}

void CCloudsController::OnDestroyDevice()
{
    m_pcbGlobalCloudAttribs.Release();
    for(int i=0; i < _countof(m_RenderCloudsTech); ++i)
        m_RenderCloudsTech[i].Release();
    for(int i=0; i < _countof(m_RenderFlatCloudsTech); ++i)
        m_RenderFlatCloudsTech[i].Release();
    m_CombineWithBBTech.Release();
    m_RenderCloudDetphToShadowMap.Release();
    m_ProcessCloudGridTech.Release();
    for(int i=0; i < _countof(m_ComputeParticleVisibilityTech); ++i)
        m_ComputeParticleVisibilityTech[i].Release();
    m_GenerateVisibleParticlesTech.Release();
	m_ProcessVisibleParticlesTech.Release();
	m_EvaluateDensityTech.Release();
	m_ComputeLightAttenuatingMass.Release();
	m_Clear3DTexTech.Release();
	for(int i=0; i < _countof(m_ComputeDispatchArgsTech); ++i)
		m_ComputeDispatchArgsTech[i].Release();
    m_ComputeOpticalDepthTech.Release();
    m_ApplyParticleLayersTech.Release();
    m_ComputeSingleSctrInParticleTech.Release();
    m_GatherPrevSctrOrderTech.Release();
    m_ComputeScatteringOrderTech.Release();
    m_AccumulateInscatteringTech.Release();
    m_RenderScatteringLUTSliceTech.Release();
    m_SortSubsequenceBitonicTech.Release();
	m_WriteSortedPariclesToVBTech.Release();
	m_MergeSubsequencesTech.Release();

    m_pdsEnableDepth.Release();
    m_pdsDisableDepth.Release();
    m_prsSolidFillCullFront.Release();
    m_prsSolidFillNoCull.Release();
    m_pbsDefault.Release();
    m_pbsRT0MulRT1MinRT2Over.Release();
    m_ptex2DCloudDensitySRV.Release();
    m_ptex2DWhiteNoiseSRV.Release();
    m_ptex2DMaxDensityMipMapSRV.Release();
    m_ptex3DNoiseSRV.Release();
    m_psamLinearWrap.Release();
    m_psamPointWrap.Release();
    m_psamLinearClamp.Release();

    m_pbufCloudGridSRV.Release();
    m_pbufCloudGridUAV.Release();
    m_pbufCloudParticlesUAV.Release();
    m_pbufCloudParticlesSRV.Release();
    m_pbufParticlesLightingSRV.Release();
    m_pbufParticlesLightingUAV.Release();
    m_pbufValidCellsUnorderedList.Release();
    m_pbufValidCellsUnorderedListUAV.Release();
    m_pbufValidCellsUnorderedListSRV.Release();
	m_pbufVisibleCellsUnorderedListSRV.Release();
    m_pbufVisibleCellsUnorderedListUAV.Release();
    m_pbufValidCellsCounter.Release();
    m_pbufValidCellsCounterSRV.Release();
	m_pbufVisibleParticlesCounter.Release();
	m_pbufVisibleParticlesCounterSRV.Release();
	m_ptex3DCellDensitySRV.Release();
	m_ptex3DCellDensityUAV.Release();
	m_ptex3DLightAttenuatingMassSRV.Release();
	m_ptex3DLightAttenuatingMassUAV.Release();
	m_pbufVisibleParticlesUnorderedListUAV.Release();
	m_pbufVisibleParticlesUnorderedListSRV.Release();
	m_pbufVisibleParticlesSortedListUAV.Release();
	m_pbufVisibleParticlesSortedListSRV.Release();
	m_pbufVisibleParticlesMergedListSRV.Release();
    m_pbufVisibleParticlesMergedListUAV.Release();

    m_pbufSerializedVisibleParticles.Release();
	m_pbufSerializedVisibleParticlesUAV.Release();

    m_pbufDispatchArgsUAV.Release();
    m_pbufDispatchArgs.Release();
	m_pbufDrawIndirectArgs.Release();

    m_pbufPackedCellLocationsSRV.Release();

    m_ptex2DScreenCloudColorSRV.Release();
    m_ptex2DScreenCloudColorRTV.Release();
    m_ptex2DScrSpaceCloudTransparencySRV.Release();
    m_ptex2DScrSpaceCloudTransparencyRTV.Release();
    m_ptex2DScrSpaceDistToCloudSRV.Release();
    m_ptex2DScrSpaceDistToCloudRTV.Release();

    m_ptex2DDownscaledScrCloudColorSRV.Release();
    m_ptex2DDownscaledScrCloudColorRTV.Release();
    m_ptex2DDownscaledScrCloudTransparencySRV.Release();
    m_ptex2DDownscaledScrCloudTransparencyRTV.Release();
    m_ptex2DDownscaledScrDistToCloudSRV.Release();
    m_ptex2DDownscaledScrDistToCloudRTV.Release();

    m_pbufParticleLayersSRV.Release();
    m_pbufParticleLayersUAV.Release();
    m_pbufClearParticleLayers.Release();

    m_ptex3DPrecomputedParticleDensitySRV.Release();
    m_ptex3DSingleSctrInParticleLUT_SRV.Release();
    m_ptex3DMultipleSctrInParticleLUT_SRV.Release();

    
    m_pRenderCloudsInputLayout.Release();
}

// Method sorts all visible particles and writes them into the buffer suitable for 
// binding as vertex buffer
void CCloudsController::SortVisibileParticles(SRenderAttribs &RenderAttribs)
{
    ID3D11DeviceContext *pDeviceContext = RenderAttribs.pDeviceContext;
    ID3D11Device *pDevice = RenderAttribs.pDevice;

    if( !m_SortSubsequenceBitonicTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
        Macros.Finalize();

        m_SortSubsequenceBitonicTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_SortSubsequenceBitonicTech.CreateComputeShaderFromFile(L"fx\\Sort.fx", "SortSubsequenceBitonicCS", Macros);
    }
    
	if( !m_WriteSortedPariclesToVBTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
        Macros.Finalize();

        m_WriteSortedPariclesToVBTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_WriteSortedPariclesToVBTech.CreateComputeShaderFromFile(L"fx\\Sort.fx", "WriteSortedPariclesToVBCS", Macros);
    }

	if( !m_MergeSubsequencesTech.IsValid() )
	{
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
        Macros.Finalize();

        m_MergeSubsequencesTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_MergeSubsequencesTech.CreateComputeShaderFromFile(L"fx\\Sort.fx", "MergeSubsequencesCS", Macros);
	}

	PrepareDispatchArgsBuffer(RenderAttribs, m_pbufVisibleParticlesCounterSRV, 0);

	// Perform bitonic sorting of subsequences
    {
        ID3D11ShaderResourceView *pSRVs[] = {m_pbufVisibleParticlesCounterSRV, m_pbufVisibleParticlesUnorderedListSRV};
        pDeviceContext->CSSetShaderResources(0, _countof(pSRVs), pSRVs);
        ID3D11UnorderedAccessView *pUAVs[] = {m_pbufVisibleParticlesSortedListUAV};
        pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
        m_SortSubsequenceBitonicTech.Apply();
        pDeviceContext->DispatchIndirect(m_pbufDispatchArgs, 0);
        pUAVs[0] = nullptr;
        pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
		pSRVs[0] = nullptr;
		pSRVs[1] = nullptr;
		pDeviceContext->CSSetShaderResources(0, _countof(pSRVs), pSRVs);
    }

	// Merge sorted subsequences
	{
        // We do not know how many passes we need to perform, because only the GPU knows the particle counter
        // We thus perform enough passes to sort maximum possible particles. The last passes do nothing and
        // have very little performance impact
		for(UINT iSubseqLen = sm_iCSThreadGroupSize; iSubseqLen < m_CloudAttribs.uiMaxParticles; iSubseqLen*=2)
		{
			m_CloudAttribs.uiParameter = iSubseqLen;
			UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));

			ID3D11Buffer *pCBs[] = {m_pcbGlobalCloudAttribs};
			pDeviceContext->CSSetConstantBuffers(0, _countof(pCBs), pCBs);

			ID3D11ShaderResourceView *pSRVs[] = {m_pbufVisibleParticlesCounterSRV, m_pbufVisibleParticlesSortedListSRV};
			pDeviceContext->CSSetShaderResources(0, _countof(pSRVs), pSRVs);
			ID3D11UnorderedAccessView *pUAVs[] = {m_pbufVisibleParticlesMergedListUAV};
			pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
			m_MergeSubsequencesTech.Apply();
			pDeviceContext->DispatchIndirect(m_pbufDispatchArgs, 0);
			pUAVs[0] = nullptr;
			pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
			pSRVs[0] = nullptr;
			pSRVs[1] = nullptr;
			pDeviceContext->CSSetShaderResources(0, _countof(pSRVs), pSRVs);
			std::swap(m_pbufVisibleParticlesMergedListUAV, m_pbufVisibleParticlesSortedListUAV);
			std::swap(m_pbufVisibleParticlesMergedListSRV, m_pbufVisibleParticlesSortedListSRV);
		}
	}

	{
		// Write sorted particle indices into the buffer suitable for binding as vertex buffer
        ID3D11ShaderResourceView *pSRVs[] = {m_pbufVisibleParticlesCounterSRV, m_pbufVisibleParticlesSortedListSRV};
        pDeviceContext->CSSetShaderResources(0, _countof(pSRVs), pSRVs);
        ID3D11UnorderedAccessView *pUAVs[] = {m_pbufSerializedVisibleParticlesUAV};
        pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
        m_WriteSortedPariclesToVBTech.Apply();
        pDeviceContext->DispatchIndirect(m_pbufDispatchArgs, 0);
        pUAVs[0] = nullptr;
        pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
        pSRVs[0] = nullptr;
        pDeviceContext->CSSetShaderResources(0, 1, pSRVs);
	}

	pDeviceContext->CopyStructureCount(m_pbufDrawIndirectArgs, 0, m_pbufVisibleParticlesUnorderedListUAV);
}


void CCloudsController::Update( const SGlobalCloudAttribs &NewAttribs,
                                const D3DXVECTOR3 &CameraPos, 
                                const D3DXVECTOR3 &LightDir,
                                ID3D11Device *pDevice, 
                                ID3D11DeviceContext *pDeviceContext, 
                                ID3D11Buffer *pcbCameraAttribs, 
                                ID3D11Buffer *pcbLightAttribs, 
                                ID3D11Buffer *pcMediaScatteringParams)
{
    if(GetAsyncKeyState(VK_F7))
    {
        for(int i=0; i < _countof(m_RenderCloudsTech); ++i)
            m_RenderCloudsTech[i].Release();
        for(int i=0; i < _countof(m_RenderFlatCloudsTech); ++i)
            m_RenderFlatCloudsTech[i].Release();
        m_CombineWithBBTech.Release();
        m_RenderCloudDetphToShadowMap.Release();
        m_ProcessCloudGridTech.Release();
        for(int i=0; i < _countof(m_ComputeParticleVisibilityTech); ++i)
            m_ComputeParticleVisibilityTech[i].Release();
        m_GenerateVisibleParticlesTech.Release();
		m_ProcessVisibleParticlesTech.Release();
		m_EvaluateDensityTech.Release();
		m_ComputeLightAttenuatingMass.Release();
		m_Clear3DTexTech.Release();
		for(int i=0; i < _countof(m_ComputeDispatchArgsTech); ++i)
			m_ComputeDispatchArgsTech[i].Release();
        m_ComputeOpticalDepthTech.Release();
        m_ApplyParticleLayersTech.Release();
        m_SortSubsequenceBitonicTech.Release();
		m_WriteSortedPariclesToVBTech.Release();
		m_MergeSubsequencesTech.Release();
        m_ComputeSingleSctrInParticleTech.Release();
        m_GatherPrevSctrOrderTech.Release();
        m_ComputeScatteringOrderTech.Release();
        m_AccumulateInscatteringTech.Release();
        m_RenderScatteringLUTSliceTech.Release();
        m_ptex3DPrecomputedParticleDensitySRV.Release();
        m_ptex3DSingleSctrInParticleLUT_SRV.Release();
        m_ptex3DMultipleSctrInParticleLUT_SRV.Release();
    }

    m_CloudAttribs.fCloudDensityThreshold = NewAttribs.fCloudDensityThreshold;
    m_CloudAttribs.fCloudAltitude         = NewAttribs.fCloudAltitude;
    m_CloudAttribs.fCloudThickness        = NewAttribs.fCloudThickness;
    m_CloudAttribs.fParticleCutOffDist    = NewAttribs.fParticleCutOffDist;

    if( m_CloudAttribs.uiNumRings     != NewAttribs.uiNumRings ||
        m_CloudAttribs.uiInnerRingDim != NewAttribs.uiInnerRingDim ||
        m_CloudAttribs.uiMaxLayers    != NewAttribs.uiMaxLayers )
    {
        m_CloudAttribs.uiNumRings     = NewAttribs.uiNumRings;
        m_CloudAttribs.uiInnerRingDim = NewAttribs.uiInnerRingDim;
        m_CloudAttribs.uiMaxLayers    = NewAttribs.uiMaxLayers;
        CreateParticleDataBuffer(pDevice);
        m_f3PrevLightDir = D3DXVECTOR3(0,0,0);
    }

    if( m_CloudAttribs.uiDownscaleFactor != NewAttribs.uiDownscaleFactor )
    {
        m_CloudAttribs.uiDownscaleFactor = NewAttribs.uiDownscaleFactor;
        OnResize(pDevice, m_uiBackBufferWidth, m_uiBackBufferHeight);
        for(int i=0; i < _countof(m_RenderCloudsTech); ++i)
            m_RenderCloudsTech[i].Release();
        for(int i=0; i < _countof(m_RenderFlatCloudsTech); ++i)
            m_RenderFlatCloudsTech[i].Release();
        m_CombineWithBBTech.Release();
    }

    if( m_CloudAttribs.uiNumCascades != NewAttribs.uiNumCascades )
    {
        m_CloudAttribs.uiNumCascades = NewAttribs.uiNumCascades;
        for(int i=0; i < _countof(m_RenderCloudsTech); ++i)
            m_RenderCloudsTech[i].Release();
        for(int i=0; i < _countof(m_RenderFlatCloudsTech); ++i)
            m_RenderFlatCloudsTech[i].Release();
    }

    if( m_CloudAttribs.bVolumetricBlending != NewAttribs.bVolumetricBlending )
    {
        m_CloudAttribs.bVolumetricBlending = NewAttribs.bVolumetricBlending;
        m_RenderCloudsTech[0].Release();
    }

    if( m_CloudAttribs.uiDensityGenerationMethod != NewAttribs.uiDensityGenerationMethod )
    {
        m_CloudAttribs.uiDensityGenerationMethod = NewAttribs.uiDensityGenerationMethod;
        m_ptex3DPrecomputedParticleDensitySRV.Release();
        m_ComputeOpticalDepthTech.Release();
    }

    // Process cloud grid
    if( !m_ProcessCloudGridTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
        Macros.Finalize();

        m_ProcessCloudGridTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_ProcessCloudGridTech.CreateComputeShaderFromFile(m_strEffectPath, "ProcessCloudGridCS", Macros);
    }
    
    UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));
}

void CCloudsController :: PrepareDispatchArgsBuffer(SRenderAttribs &RenderAttribs, ID3D11ShaderResourceView *pCounterSRV, int iTechInd)
{
    ID3D11DeviceContext *pDeviceContext = RenderAttribs.pDeviceContext;
    ID3D11Device *pDevice = RenderAttribs.pDevice;

	auto &ComputeDispatchArgsTech = m_ComputeDispatchArgsTech[iTechInd];
	// Compute DispatchIndirect() arguments
    if( !ComputeDispatchArgsTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
        Macros.Finalize();

		ComputeDispatchArgsTech.SetDeviceAndContext(pDevice, pDeviceContext);
		std::stringstream ss;
		ss << "ComputeDispatchArgs" << iTechInd << "CS";
		ComputeDispatchArgsTech.CreateComputeShaderFromFile(m_strEffectPath, ss.str().c_str(), Macros);
    }
    ID3D11Buffer *pCBs[] = {m_pcbGlobalCloudAttribs};
    pDeviceContext->CSSetConstantBuffers(0, _countof(pCBs), pCBs);
	ID3D11ShaderResourceView *pSRVs[] = {pCounterSRV};
    pDeviceContext->CSSetShaderResources(0, 1, pSRVs);
	ID3D11UnorderedAccessView *pUAVs[] = {m_pbufDispatchArgsUAV};
    pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
    ComputeDispatchArgsTech.Apply();
    pDeviceContext->Dispatch(1,1,1);
    pUAVs[0] = nullptr;
    pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
}

void CCloudsController :: ClearCellDensityAndAttenuationTextures(SRenderAttribs &RenderAttribs)
{
    ID3D11DeviceContext *pDeviceContext = RenderAttribs.pDeviceContext;
    ID3D11Device *pDevice = RenderAttribs.pDevice;

	if( !m_Clear3DTexTech.IsValid() )
	{
		CD3DShaderMacroHelper Macros;
		DefineMacros(Macros);
		Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
		Macros.Finalize();

		m_Clear3DTexTech.SetDeviceAndContext(pDevice, pDeviceContext);
		m_Clear3DTexTech.CreateComputeShaderFromFile(m_strEffectPath, "Clear3DTextureCS", Macros);
	}

	ID3D11UnorderedAccessView * pUAVs[] = {m_ptex3DCellDensityUAV, m_ptex3DLightAttenuatingMassUAV};
	pDeviceContext->CSSetUnorderedAccessViews(0, _countof(pUAVs), pUAVs, nullptr);
	uint TotalVoxels = 
		m_CloudAttribs.uiRingDimension * m_CloudAttribs.uiDensityBufferScale *
		m_CloudAttribs.uiRingDimension * m_CloudAttribs.uiDensityBufferScale *
		m_CloudAttribs.uiMaxLayers * m_CloudAttribs.uiDensityBufferScale * m_CloudAttribs.uiNumRings;

	m_Clear3DTexTech.Apply();
	pDeviceContext->Dispatch( (TotalVoxels + (sm_iCSThreadGroupSize-1)) / sm_iCSThreadGroupSize, 1, 1);
	pUAVs[0] = nullptr;
	pUAVs[1] = nullptr;
	pDeviceContext->CSSetUnorderedAccessViews(0, _countof(pUAVs), pUAVs, nullptr);
}

void CCloudsController :: GenerateParticles(SRenderAttribs &RenderAttribs)
{
    ID3D11DeviceContext *pDeviceContext = RenderAttribs.pDeviceContext;
    ID3D11Device *pDevice = RenderAttribs.pDevice;

    ID3D11Buffer *pCBs[] = {m_pcbGlobalCloudAttribs, RenderAttribs.pcMediaScatteringParams, RenderAttribs.pcbCameraAttribs, RenderAttribs.pcbLightAttribs};
    pDeviceContext->CSSetConstantBuffers(0, _countof(pCBs), pCBs);

    ID3D11SamplerState *pSamplers[] = {m_psamLinearClamp, m_psamLinearWrap, m_psamPointWrap};
    pDeviceContext->CSSetSamplers(0, _countof(pSamplers), pSamplers);

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_pbufPackedCellLocationsSRV, // StructuredBuffer<uint> g_PackedCellLocations : register( t0 );
        m_ptex2DCloudDensitySRV,
        m_ptex3DNoiseSRV,
        m_ptex2DMaxDensityMipMapSRV,
        nullptr,
        nullptr, // Texture2D<float2> g_tex2DOccludedNetDensityToAtmTop : register( t5 );
    };
    pDeviceContext->CSSetShaderResources(0, _countof(pSRVs), pSRVs);
    
    ID3D11UnorderedAccessView *pUAVs[] = {m_pbufCloudGridUAV, m_pbufValidCellsUnorderedListUAV, m_pbufVisibleCellsUnorderedListUAV};
    UINT uiZeroCounters[_countof(pUAVs)] =  {0};
    pDeviceContext->CSSetUnorderedAccessViews(0, _countof(pUAVs), pUAVs, uiZeroCounters);

    m_ProcessCloudGridTech.Apply();
    pDeviceContext->Dispatch( (m_CloudAttribs.uiNumCells + (sm_iCSThreadGroupSize-1)) / sm_iCSThreadGroupSize, 1, 1);
    
    memset(pUAVs, 0, sizeof(pUAVs));
    pDeviceContext->CSSetUnorderedAccessViews(0, _countof(pUAVs), pUAVs, nullptr);
	
	pDeviceContext->CopyStructureCount(m_pbufValidCellsCounter, 0, m_pbufValidCellsUnorderedListUAV);

	// It is more efficient to clear both UAVs simultaneously using CS
	//ClearCellDensityAndAttenuationTextures(RenderAttribs);
	float Zero[4]={0,0,0,0};
	pDeviceContext->ClearUnorderedAccessViewFloat(m_ptex3DCellDensityUAV,Zero);
	pDeviceContext->ClearUnorderedAccessViewFloat(m_ptex3DLightAttenuatingMassUAV,Zero);

	if( !m_EvaluateDensityTech.IsValid() )
	{
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
        Macros.Finalize();

        m_EvaluateDensityTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_EvaluateDensityTech.CreateComputeShaderFromFile(m_strEffectPath, "EvaluateDensityCS", Macros);
	}

	// TODO: need to process only VISIBLE AND LIGHT OCCLUDING cells here!
	PrepareDispatchArgsBuffer(RenderAttribs, m_pbufValidCellsCounterSRV, 1);

    pSRVs[0] = m_pbufValidCellsCounterSRV;       // Buffer<uint> g_ValidCellsCounter                 : register( t0 );
    pSRVs[1] = m_pbufValidCellsUnorderedListSRV; // StructuredBuffer<uint> g_ValidCellsUnorderedList : register( t1 );
    pSRVs[2] = m_pbufCloudGridSRV;               // StructuredBuffer<SCloudCellAttribs> g_CloudCells : register( t2 );
    pSRVs[3] = nullptr;						     // t3
	pSRVs[4] = m_ptex3DNoiseSRV;				 // t4
    pDeviceContext->CSSetShaderResources(0, 5, pSRVs);
	pUAVs[0] = m_ptex3DCellDensityUAV;
    pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
    m_EvaluateDensityTech.Apply();
    pDeviceContext->DispatchIndirect(m_pbufDispatchArgs, 0);
    pUAVs[0] = nullptr;
    pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
	
	
	pDeviceContext->CopyStructureCount(m_pbufValidCellsCounter, 0, m_pbufVisibleCellsUnorderedListUAV);
	PrepareDispatchArgsBuffer(RenderAttribs, m_pbufValidCellsCounterSRV, 1);

	if( !m_ComputeLightAttenuatingMass.IsValid() )
	{
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
        Macros.Finalize();

        m_ComputeLightAttenuatingMass.SetDeviceAndContext(pDevice, pDeviceContext);
        m_ComputeLightAttenuatingMass.CreateComputeShaderFromFile(m_strEffectPath, "ComputeLightAttenuatingMassCS", Macros);
	}

    pSRVs[0] = m_pbufValidCellsCounterSRV;       // Buffer<uint> g_ValidCellsCounter                 : register( t0 );
	pSRVs[1] = m_pbufVisibleCellsUnorderedListSRV; // StructuredBuffer<uint> g_ValidCellsUnorderedList : register( t1 );
    pSRVs[2] = m_pbufCloudGridSRV;               // StructuredBuffer<SCloudCellAttribs> g_CloudCells : register( t2 );
    pSRVs[3] = nullptr;						     // t3
	pSRVs[4] = m_ptex3DCellDensitySRV;			 // t4
    pDeviceContext->CSSetShaderResources(0, 5, pSRVs);
	pUAVs[0] = m_ptex3DLightAttenuatingMassUAV;
    pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);
    m_ComputeLightAttenuatingMass.Apply();
    pDeviceContext->DispatchIndirect(m_pbufDispatchArgs, 0);
    pUAVs[0] = nullptr;
    pDeviceContext->CSSetUnorderedAccessViews(0, 1, pUAVs, nullptr);

    // Process all valid cells and generate visible particles
    if(!m_GenerateVisibleParticlesTech.IsValid())
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
        Macros.Finalize();

        m_GenerateVisibleParticlesTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_GenerateVisibleParticlesTech.CreateComputeShaderFromFile(m_strEffectPath, "GenerateVisibleParticlesCS", Macros);
    }

	// We now need to use the first method to calculate dispatch args
	PrepareDispatchArgsBuffer(RenderAttribs, m_pbufValidCellsCounterSRV, 0);

    pUAVs[0] = m_pbufCloudParticlesUAV;
    pUAVs[1] = m_pbufVisibleParticlesUnorderedListUAV;
    pDeviceContext->CSSetUnorderedAccessViews(0, 2, pUAVs, uiZeroCounters);
    pSRVs[0] = m_pbufValidCellsCounterSRV;       // Buffer<uint> g_ValidCellsCounter                 : register( t0 );
    pSRVs[1] = m_pbufVisibleCellsUnorderedListSRV;// StructuredBuffer<uint> g_ValidCellsUnorderedList : register( t1 );
    pSRVs[2] = m_pbufCloudGridSRV;               // StructuredBuffer<SCloudCellAttribs> g_CloudCells : register( t2 );
    pSRVs[3] = m_ptex2DWhiteNoiseSRV;            // Texture2D<float3> g_tex2DWhiteNoise              : register( t3 );
	pSRVs[4] = m_ptex3DCellDensitySRV;			 // t4
    pDeviceContext->CSSetShaderResources(0, 5, pSRVs);
    m_GenerateVisibleParticlesTech.Apply();
    pDeviceContext->DispatchIndirect(m_pbufDispatchArgs, 0);
    memset(pUAVs, 0, sizeof(pUAVs));
    pDeviceContext->CSSetUnorderedAccessViews(0, 2, pUAVs, nullptr);

	pDeviceContext->CopyStructureCount(m_pbufVisibleParticlesCounter, 0, m_pbufVisibleParticlesUnorderedListUAV);

	{
		// Process all valid cells and generate visible particles
		if(!m_ProcessVisibleParticlesTech.IsValid())
		{
			CD3DShaderMacroHelper Macros;
			DefineMacros(Macros);
			Macros.AddShaderMacro("THREAD_GROUP_SIZE", sm_iCSThreadGroupSize);
			Macros.Finalize();

			m_ProcessVisibleParticlesTech.SetDeviceAndContext(pDevice, pDeviceContext);
			m_ProcessVisibleParticlesTech.CreateComputeShaderFromFile(m_strEffectPath, "ProcessVisibleParticlesCS", Macros);
		}

		PrepareDispatchArgsBuffer(RenderAttribs, m_pbufVisibleParticlesCounterSRV, 0);
		ID3D11UnorderedAccessView *pUAVs[] = 
		{
			m_pbufParticlesLightingUAV
		};

		pDeviceContext->CSSetUnorderedAccessViews(0, _countof(pUAVs), pUAVs, uiZeroCounters);
		ID3D11ShaderResourceView *pSRVs[] = 
		{
			m_pbufVisibleParticlesCounterSRV,       // Buffer<uint> g_ValidCellsCounter                 : register( t0 );
			m_pbufVisibleParticlesUnorderedListSRV, // StructuredBuffer<SParticleIdAndDist>  g_VisibleParticlesUnorderedList : register( t1 );
			m_pbufCloudGridSRV,				        // StructuredBuffer<SCloudCellAttribs> g_CloudCells : register( t2 );
			m_pbufCloudParticlesSRV,	            // StructuredBuffer<SParticleAttribs>  g_Particles     : register( t3 );
			nullptr,					  		    // t4
			RenderAttribs.pPrecomputedNetDensitySRV,// Texture2D<float2> g_tex2DOccludedNetDensityToAtmTop : register( t5 );
			m_ptex3DLightAttenuatingMassSRV,        // Texture3D<float>       g_tex3DLightAttenuatingMass      : register( t6 );
			RenderAttribs.pAmbientSkylightSRV		// Texture2D<float3>       g_tex2DAmbientSkylight       : register( t7 );
		};
		pDeviceContext->CSSetShaderResources(0, _countof(pSRVs), pSRVs);
		m_ProcessVisibleParticlesTech.Apply();
		pDeviceContext->DispatchIndirect(m_pbufDispatchArgs, 0);
		memset(pUAVs, 0, sizeof(pUAVs));
		pDeviceContext->CSSetUnorderedAccessViews(0, _countof(pUAVs), pUAVs, nullptr);
		memset(pSRVs, 0, sizeof(pSRVs));
		pDeviceContext->CSSetShaderResources(0, _countof(pSRVs), pSRVs);
	}
}

void CCloudsController :: DefineMacros(class CD3DShaderMacroHelper &Macros)
{
    {
        std::stringstream ss;
        ss<<"float2("<<m_uiCloudDensityTexWidth<<","<<m_uiCloudDensityTexHeight<<")";
        Macros.AddShaderMacro("CLOUD_DENSITY_TEX_DIM", ss.str());
    }
    {
        std::stringstream ss;
        ss<<"float4("<< m_PrecomputedOpticalDepthTexDim.iNumStartPosZenithAngles  <<","
                     << m_PrecomputedOpticalDepthTexDim.iNumStartPosAzimuthAngles <<","
                     << m_PrecomputedOpticalDepthTexDim.iNumDirectionZenithAngles <<","
                     << m_PrecomputedOpticalDepthTexDim.iNumDirectionAzimuthAngles<< ")";
        Macros.AddShaderMacro("OPTICAL_DEPTH_LUT_DIM", ss.str());
    }

    Macros.AddShaderMacro("NUM_PARTICLE_LAYERS", m_CloudAttribs.uiNumParticleLayers);
    Macros.AddShaderMacro("PS_ORDERING_AVAILABLE", m_bPSOrderingAvailable);

    {
        std::stringstream ss;
        ss<<"float4("<< m_PrecomputedSctrInParticleLUTDim.iNumStartPosZenithAngles <<","
                     << m_PrecomputedSctrInParticleLUTDim.iNumViewDirAzimuthAngles <<","
                     << m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles <<","
                     << m_PrecomputedSctrInParticleLUTDim.iNumDistancesFromCenter << ")";
        Macros.AddShaderMacro("VOL_SCATTERING_IN_PARTICLE_LUT_DIM", ss.str());
    }
    {
        std::stringstream ss;
        ss<<"float3("<< m_PrecomputedSctrInParticleLUTDim.iNumStartPosZenithAngles <<","
                     << m_PrecomputedSctrInParticleLUTDim.iNumViewDirAzimuthAngles <<","
                     << m_PrecomputedSctrInParticleLUTDim.iNumViewDirZenithAngles/2<<")";
        Macros.AddShaderMacro("SRF_SCATTERING_IN_PARTICLE_LUT_DIM", ss.str());
    }
    Macros.AddShaderMacro("BACK_BUFFER_DOWNSCALE_FACTOR", m_CloudAttribs.uiDownscaleFactor);
}

// Renders all visible particles
void CCloudsController::RenderParticles(SRenderAttribs &RenderAttribs)
{
    ID3D11DeviceContext *pDeviceContext = RenderAttribs.pDeviceContext;
    ID3D11Device *pDevice = RenderAttribs.pDevice;

    bool bLightSpacePass = RenderAttribs.bLightSpacePass;

    auto &RenderCloudsTech = m_RenderCloudsTech[bLightSpacePass ? 1 : 0];
    
    if( !RenderCloudsTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("LIGHT_SPACE_PASS", bLightSpacePass);
        Macros.AddShaderMacro("NUM_SHADOW_CASCADES", m_CloudAttribs.uiNumCascades);
        Macros.AddShaderMacro("BEST_CASCADE_SEARCH", false);
        Macros.AddShaderMacro("TILING_MODE", false);
        Macros.AddShaderMacro("VOLUMETRIC_BLENDING", m_bPSOrderingAvailable && m_CloudAttribs.bVolumetricBlending);
        Macros.Finalize();

        RenderCloudsTech.SetDeviceAndContext(pDevice, pDeviceContext);
        RenderCloudsTech.CreateVGPShadersFromFile(m_strEffectPath, "RenderCloudsVS", "RenderCloudsGS", "RenderCloudsPS", Macros);
        RenderCloudsTech.SetDS( m_pdsDisableDepth /*m_pdsEnableDepth*/ );
        RenderCloudsTech.SetRS( m_prsSolidFillCullFront );
        RenderCloudsTech.SetBS( m_pbsRT0MulRT1MinRT2Over );

        if( !m_pRenderCloudsInputLayout )
        {
            // Create vertex input layout
            const D3D11_INPUT_ELEMENT_DESC layout[] =
            {
                { "PARTICLE_ID",  0, DXGI_FORMAT_R32_UINT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
            };

	        auto pVSByteCode = RenderCloudsTech.GetVSByteCode();
            HRESULT hr;
            V( pDevice->CreateInputLayout( layout, ARRAYSIZE( layout ),
                                            pVSByteCode->GetBufferPointer(),
										    pVSByteCode->GetBufferSize(),
                                            &m_pRenderCloudsInputLayout ) );
        }
    }

    SortVisibileParticles(RenderAttribs);

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        RenderAttribs.pDepthBufferSRV,
        m_ptex2DCloudDensitySRV,
        m_pbufCloudGridSRV,                // StructuredBuffer<SCloudCellAttribs> g_CloudCells : register( t2 );
        m_pbufCloudParticlesSRV,           // StructuredBuffer<SParticleAttribs> g_Particles : register( t3 );
		m_ptex3DCellDensitySRV,                  // Texture3D<float> g_tex3DNoise                  : register(t4);
		nullptr,							// t5
		m_ptex3DLightAttenuatingMassSRV,       // Texture3D<float>       g_tex3DLightAttenuatingMass      : register( t6 );
        m_pbufParticlesLightingSRV,         // t7
        nullptr,         // t8
        nullptr,         // t9
        m_ptex3DPrecomputedParticleDensitySRV, // t10
        m_ptex3DSingleSctrInParticleLUT_SRV,   // t11
        m_ptex3DMultipleSctrInParticleLUT_SRV  // t12
	};

    pDeviceContext->VSSetShaderResources(0, _countof(pSRVs), pSRVs);
    pDeviceContext->GSSetShaderResources(0, _countof(pSRVs), pSRVs);
    pDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

    pDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
    RenderCloudsTech.Apply();
    UINT Strides[] = {sizeof(UINT)};
    UINT Offsets[] = {0};
    pDeviceContext->IASetVertexBuffers(0, 1, &m_pbufSerializedVisibleParticles.p, Strides, Offsets);
    pDeviceContext->IASetInputLayout(m_pRenderCloudsInputLayout);
    pDeviceContext->DrawInstancedIndirect(m_pbufDrawIndirectArgs, 0);

    if( !RenderAttribs.bLightSpacePass && m_bPSOrderingAvailable && m_CloudAttribs.bVolumetricBlending )
    {
        if( !m_ApplyParticleLayersTech.IsValid() )
        {
            CD3DShaderMacroHelper Macros;
            DefineMacros(Macros);
            Macros.Finalize();

            m_ApplyParticleLayersTech.SetDeviceAndContext(pDevice, pDeviceContext);
            m_ApplyParticleLayersTech.CreateVGPShadersFromFile(m_strEffectPath, "ScreenSizeQuadVS", nullptr, "ApplyParticleLayersPS", Macros);
            m_ApplyParticleLayersTech.SetDS( m_pdsDisableDepth );
            m_ApplyParticleLayersTech.SetRS( m_prsSolidFillNoCull );
            m_ApplyParticleLayersTech.SetBS( m_pbsRT0MulRT1MinRT2Over );
        }

        // We need to remove UAVs from the pipeline to be able to bind it as shader resource
        ID3D11RenderTargetView *pRTVs[] = {m_ptex2DScrSpaceCloudTransparencyRTV, m_ptex2DScrSpaceDistToCloudRTV, m_ptex2DScreenCloudColorRTV};
        ID3D11RenderTargetView *pDwnsclRTVs[] = {m_ptex2DDownscaledScrCloudTransparencyRTV, m_ptex2DDownscaledScrDistToCloudRTV, m_ptex2DDownscaledScrCloudColorRTV};
        if(m_CloudAttribs.uiDownscaleFactor > 1 )
            pDeviceContext->OMSetRenderTargets(_countof(pDwnsclRTVs), pDwnsclRTVs, nullptr);
        else
            pDeviceContext->OMSetRenderTargets(_countof(pRTVs), pRTVs, nullptr);

        ID3D11ShaderResourceView *pSRVs[] = 
        {
            m_pbufParticleLayersSRV
        };
        pDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

        RenderQuad(pDeviceContext, m_ApplyParticleLayersTech);
    }

    UnbindPSResources(pDeviceContext);
    UnbindVSResources(pDeviceContext);
    UnbindGSResources(pDeviceContext);
}

// Renders flat clouds on a spherical layer
void CCloudsController::RenderFlatClouds(SRenderAttribs &RenderAttribs)
{
    ID3D11DeviceContext *pDeviceContext = RenderAttribs.pDeviceContext;
    ID3D11Device *pDevice = RenderAttribs.pDevice;
    
    bool bLightSpacePass = RenderAttribs.bLightSpacePass;

    auto &RenderFlatCloudsTech = m_RenderFlatCloudsTech[bLightSpacePass ? 1 : 0];
    if(!RenderFlatCloudsTech.IsValid())
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("LIGHT_SPACE_PASS", bLightSpacePass);
        Macros.AddShaderMacro("NUM_SHADOW_CASCADES", m_CloudAttribs.uiNumCascades);
        Macros.AddShaderMacro("BEST_CASCADE_SEARCH", false);
        Macros.Finalize();

        RenderFlatCloudsTech.SetDeviceAndContext(pDevice, pDeviceContext);
        RenderFlatCloudsTech.CreateVGPShadersFromFile(m_strEffectPath, "ScreenSizeQuadVS", nullptr, "RenderFlatCloudsPS", Macros);
        RenderFlatCloudsTech.SetDS( m_pdsDisableDepth );
        RenderFlatCloudsTech.SetRS( m_prsSolidFillNoCull );
        RenderFlatCloudsTech.SetBS( m_pbsDefault );
    }


    ID3D11ShaderResourceView *pSRVs[] = 
    {
        RenderAttribs.pDepthBufferSRV,
        m_ptex2DCloudDensitySRV,
        nullptr,                                 
        m_ptex2DMaxDensityMipMapSRV,             // Texture2D<float> g_tex2MaxDensityMip           : register(t3);
        m_ptex3DNoiseSRV,                        // Texture3D<float> g_tex3DNoise                  : register(t4);
        RenderAttribs.pPrecomputedNetDensitySRV, // Texture2D<float2> g_tex2DOccludedNetDensityToAtmTop : register( t5 );
        nullptr,                                 // t6
        RenderAttribs.pAmbientSkylightSRV        // t7
	};
    pDeviceContext->VSSetShaderResources(0, _countof(pSRVs), pSRVs);
    pDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);
        
    if( !bLightSpacePass && m_CloudAttribs.uiDownscaleFactor > 1 )
    {
        ID3D11ShaderResourceView *pSRVs2[] = 
        {
            m_ptex2DDownscaledScrCloudTransparencySRV,
            m_ptex2DDownscaledScrDistToCloudSRV,
            m_ptex2DDownscaledScrCloudColorSRV
        };
        pDeviceContext->PSSetShaderResources(11, _countof(pSRVs2), pSRVs2);
    }

    RenderQuad(pDeviceContext, RenderFlatCloudsTech);

    UnbindPSResources(pDeviceContext);
    UnbindVSResources(pDeviceContext);
    UnbindGSResources(pDeviceContext);
}

// Renders light space density from light
void CCloudsController::RenderLightSpaceDensity(SRenderAttribs &RenderAttribs)
{
    ID3D11DeviceContext *pDeviceContext = RenderAttribs.pDeviceContext;
    ID3D11Device *pDevice = RenderAttribs.pDevice;

    m_CloudAttribs.fTime = RenderAttribs.fCurrTime;
    m_CloudAttribs.f4Parameter.x = (float)RenderAttribs.iCascadeIndex;
    UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));

    ID3D11Buffer *pCBs[] = {m_pcbGlobalCloudAttribs, RenderAttribs.pcMediaScatteringParams, RenderAttribs.pcbCameraAttribs, RenderAttribs.pcbLightAttribs};
    pDeviceContext->VSSetConstantBuffers(0, _countof(pCBs), pCBs);
    pDeviceContext->GSSetConstantBuffers(0, _countof(pCBs), pCBs);
    pDeviceContext->PSSetConstantBuffers(0, _countof(pCBs), pCBs);
    pDeviceContext->CSSetConstantBuffers(0, _countof(pCBs), pCBs);

    ID3D11SamplerState *pSamplers[] = {m_psamLinearClamp, m_psamLinearWrap, m_psamPointWrap};
    pDeviceContext->VSSetSamplers(0, _countof(pSamplers), pSamplers);
    pDeviceContext->GSSetSamplers(0, _countof(pSamplers), pSamplers);
    pDeviceContext->PSSetSamplers(0, _countof(pSamplers), pSamplers);
    pDeviceContext->CSSetSamplers(0, _countof(pSamplers), pSamplers);


    ID3D11RenderTargetView *ppOrigRTVs[2];
    CComPtr<ID3D11DepthStencilView> pOrigDSV;
    pDeviceContext->OMGetRenderTargets(_countof(ppOrigRTVs), ppOrigRTVs, &pOrigDSV);
    CComPtr<ID3D11RenderTargetView> pTransparencyRTV, pMinMaxDepthRTV;
    pTransparencyRTV.Attach(ppOrigRTVs[0]);
    pMinMaxDepthRTV.Attach(ppOrigRTVs[1]);
    
    m_CloudAttribs.f2LiSpCloudDensityDim.x = static_cast<float>(RenderAttribs.uiLiSpCloudDensityDim);
    m_CloudAttribs.f2LiSpCloudDensityDim.y = static_cast<float>(RenderAttribs.uiLiSpCloudDensityDim);

    float fOneMinusEpsilon = 1.f;
    --((int&)fOneMinusEpsilon);
    const float One[4] = {1, 1, 1, fOneMinusEpsilon};
    pDeviceContext->ClearRenderTargetView(pTransparencyRTV, One);

    const float InitialMinMaxDepth[4] = {0, 0, 0, FLT_MIN};
    pDeviceContext->ClearRenderTargetView(pMinMaxDepthRTV, InitialMinMaxDepth);

    RenderAttribs.bLightSpacePass = true;
    RenderFlatClouds(RenderAttribs);
}

// Merges light space distance to cloud with the shadow map
void CCloudsController::MergeLiSpDensityWithShadowMap(SRenderAttribs &RenderAttribs)
{
    ID3D11DeviceContext *pDeviceContext = RenderAttribs.pDeviceContext;
    ID3D11Device *pDevice = RenderAttribs.pDevice;

    if( !m_RenderCloudDetphToShadowMap.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_RenderCloudDetphToShadowMap.SetDeviceAndContext(pDevice, pDeviceContext);
        m_RenderCloudDetphToShadowMap.CreateVGPShadersFromFile(m_strEffectPath, "ScreenSizeQuadVS", nullptr, "RenderCloudDepthToShadowMapPS", Macros);
        m_RenderCloudDetphToShadowMap.SetDS( m_pdsEnableDepth );
        m_RenderCloudDetphToShadowMap.SetRS( m_prsSolidFillNoCull );
        m_RenderCloudDetphToShadowMap.SetBS( m_pbsDefault );
    }

    CComPtr<ID3D11RenderTargetView> pOrigRTV;
    CComPtr<ID3D11DepthStencilView> pOrigDSV;
    pDeviceContext->OMGetRenderTargets(1, &pOrigRTV, &pOrigDSV);
    
    D3D11_VIEWPORT OrigViewPort;
    UINT iNumOldViewports = 1;
    pDeviceContext->RSGetViewports(&iNumOldViewports, &OrigViewPort);

    pDeviceContext->OMSetRenderTargets(0, nullptr, RenderAttribs.pShadowMapDSV);
    ID3D11ShaderResourceView *pSRVs[] = 
    {
        RenderAttribs.pLiSpCloudTransparencySRV,
        RenderAttribs.pLiSpCloudMinMaxDepthSRV
    };
    pDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

    ID3D11SamplerState *pSamplers[] = {m_psamLinearClamp, m_psamLinearWrap, m_psamPointWrap};
    pDeviceContext->VSSetSamplers(0, _countof(pSamplers), pSamplers);
    pDeviceContext->PSSetSamplers(0, _countof(pSamplers), pSamplers);

    m_CloudAttribs.f4Parameter.x = (float)RenderAttribs.iCascadeIndex;
    UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));

    ID3D11Buffer *pCBs[] = {m_pcbGlobalCloudAttribs};
    pDeviceContext->PSSetConstantBuffers(0, _countof(pCBs), pCBs);

    RenderQuad(pDeviceContext, m_RenderCloudDetphToShadowMap);

    pDeviceContext->OMSetRenderTargets(1, &pOrigRTV.p, pOrigDSV);
    pDeviceContext->RSSetViewports(iNumOldViewports, &OrigViewPort);
}

// Renders cloud color, transparency and distance to clouds from camera
void CCloudsController::RenderScreenSpaceDensityAndColor(SRenderAttribs &RenderAttribs)
{
    ID3D11DeviceContext *pDeviceContext = RenderAttribs.pDeviceContext;
    ID3D11Device *pDevice = RenderAttribs.pDevice;

    if( !m_ptex3DPrecomputedParticleDensitySRV )
    {
        HRESULT hr;
        V( PrecomputParticleDensity(pDevice, pDeviceContext) );
    }

    if( !m_ptex3DSingleSctrInParticleLUT_SRV || !m_ptex3DMultipleSctrInParticleLUT_SRV )
    {
        HRESULT hr;
        V( PrecomputeScatteringInParticle(pDevice, pDeviceContext) );
    }
	GenerateParticles(RenderAttribs);

    m_CloudAttribs.fTime = RenderAttribs.fCurrTime;
    m_CloudAttribs.f4Parameter.x = (float)RenderAttribs.iCascadeIndex;
    UpdateConstantBuffer(pDeviceContext, m_pcbGlobalCloudAttribs, &m_CloudAttribs, sizeof(m_CloudAttribs));

    ID3D11Buffer *pCBs[] = {m_pcbGlobalCloudAttribs, RenderAttribs.pcMediaScatteringParams, RenderAttribs.pcbCameraAttribs, RenderAttribs.pcbLightAttribs};
    pDeviceContext->VSSetConstantBuffers(0, _countof(pCBs), pCBs);
    pDeviceContext->GSSetConstantBuffers(0, _countof(pCBs), pCBs);
    pDeviceContext->PSSetConstantBuffers(0, _countof(pCBs), pCBs);
    pDeviceContext->CSSetConstantBuffers(0, _countof(pCBs), pCBs);

    ID3D11SamplerState *pSamplers[] = {m_psamLinearClamp, m_psamLinearWrap, m_psamPointWrap};
    pDeviceContext->VSSetSamplers(0, _countof(pSamplers), pSamplers);
    pDeviceContext->GSSetSamplers(0, _countof(pSamplers), pSamplers);
    pDeviceContext->PSSetSamplers(0, _countof(pSamplers), pSamplers);
    pDeviceContext->CSSetSamplers(0, _countof(pSamplers), pSamplers);

    CComPtr<ID3D11RenderTargetView> pOrigRTV;
    CComPtr<ID3D11DepthStencilView> pOrigDSV;
    pDeviceContext->OMGetRenderTargets(1, &pOrigRTV, &pOrigDSV);

    D3D11_VIEWPORT OrigViewPort;
    UINT iNumOldViewports = 1;
    pDeviceContext->RSGetViewports(&iNumOldViewports, &OrigViewPort);

    float Zero[4]={0,0,0,FLT_MIN};
    pDeviceContext->ClearRenderTargetView(m_ptex2DScreenCloudColorRTV, Zero);

    float fOneMinusEpsilon = 1.f;
    --((int&)fOneMinusEpsilon);
    const float One[4] = {1, 1, 1, fOneMinusEpsilon}; // Use 1-Epsilon to block fast clear path
    pDeviceContext->ClearRenderTargetView(m_ptex2DScrSpaceCloudTransparencyRTV, One);

    if( m_bPSOrderingAvailable && m_CloudAttribs.bVolumetricBlending )
    {
        CComPtr<ID3D11Resource> pDstRes;
        m_pbufParticleLayersUAV->GetResource(&pDstRes);
        pDeviceContext->CopyResource(pDstRes, m_pbufClearParticleLayers);
    }
    // With complimentary depth buffer 0 is the far clipping plane
    // TODO: output some distance from shader (or clear with distanc to horizon?). (Do not forget about sample refinement!)
    const float InitialMinMaxZ[4] = {+FLT_MAX, -FLT_MAX, 0, 0};
    pDeviceContext->ClearRenderTargetView(m_ptex2DScrSpaceDistToCloudRTV, InitialMinMaxZ);

    RenderAttribs.bLightSpacePass = false;
    if(m_CloudAttribs.uiDownscaleFactor > 1 )
    {
        D3D11_VIEWPORT NewViewPort = OrigViewPort;
        NewViewPort.Width  = m_CloudAttribs.fDownscaledBackBufferWidth;
        NewViewPort.Height = m_CloudAttribs.fDownscaledBackBufferHeight;
        pDeviceContext->RSSetViewports(1, &NewViewPort);

        pDeviceContext->ClearRenderTargetView(m_ptex2DDownscaledScrCloudColorRTV, Zero);
        pDeviceContext->ClearRenderTargetView(m_ptex2DDownscaledScrCloudTransparencyRTV, One);
        pDeviceContext->ClearRenderTargetView(m_ptex2DDownscaledScrDistToCloudRTV, InitialMinMaxZ);
        ID3D11RenderTargetView *pRTVs[] = {m_ptex2DDownscaledScrCloudTransparencyRTV, m_ptex2DDownscaledScrDistToCloudRTV, m_ptex2DDownscaledScrCloudColorRTV};
        if( m_bPSOrderingAvailable && m_CloudAttribs.bVolumetricBlending )
        {
            ID3D11UnorderedAccessView *pUAVs[] = {m_pbufParticleLayersUAV};
            UINT puiInitialCounts[_countof(pUAVs)] = {0};
            pDeviceContext->OMSetRenderTargetsAndUnorderedAccessViews(_countof(pRTVs), pRTVs, nullptr, 3, _countof(pUAVs), pUAVs, puiInitialCounts);
        }
        else
        {
            pDeviceContext->OMSetRenderTargets(_countof(pRTVs), pRTVs, nullptr);
        }
        RenderParticles(RenderAttribs);
        
        pDeviceContext->RSSetViewports(1, &OrigViewPort);
    }

    ID3D11RenderTargetView *pRTVs[] = {m_ptex2DScrSpaceCloudTransparencyRTV, m_ptex2DScrSpaceDistToCloudRTV, m_ptex2DScreenCloudColorRTV};
    pDeviceContext->OMSetRenderTargets(_countof(pRTVs), pRTVs, nullptr);

    RenderFlatClouds(RenderAttribs);
    if(m_CloudAttribs.uiDownscaleFactor == 1 )
    {
        if( m_bPSOrderingAvailable && m_CloudAttribs.bVolumetricBlending )
        {
            ID3D11RenderTargetView *pRTVs[] = {m_ptex2DScrSpaceCloudTransparencyRTV, nullptr, m_ptex2DScreenCloudColorRTV};
            ID3D11UnorderedAccessView *pUAVs[] = {m_pbufParticleLayersUAV};
            UINT puiInitialCounts[_countof(pUAVs)] = {0};
            pDeviceContext->OMSetRenderTargetsAndUnorderedAccessViews(_countof(pRTVs), pRTVs, nullptr, 3, _countof(pUAVs), pUAVs, puiInitialCounts);
        }
        RenderParticles(RenderAttribs);
    }

    pDeviceContext->OMSetRenderTargets(1, &pOrigRTV.p, pOrigDSV);
    pDeviceContext->RSSetViewports(iNumOldViewports, &OrigViewPort);
}

// Combines cloud color & transparancy with back buffer
void CCloudsController::CombineWithBackBuffer(ID3D11Device *pDevice, 
                                              ID3D11DeviceContext *pDeviceContext, 
                                              ID3D11ShaderResourceView *pDepthBufferSRV,
                                              ID3D11ShaderResourceView *pBackBufferSRV)
{
    if( !m_CombineWithBBTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_CombineWithBBTech.SetDeviceAndContext(pDevice, pDeviceContext);
        m_CombineWithBBTech.CreateVGPShadersFromFile(m_strEffectPath, "ScreenSizeQuadVS", nullptr, "CombineWithBackBufferPS", Macros);
        m_CombineWithBBTech.SetDS( m_pdsDisableDepth );
        m_CombineWithBBTech.SetRS( m_prsSolidFillNoCull );
        m_CombineWithBBTech.SetBS( m_pbsDefault );
    }

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        pDepthBufferSRV,
        pBackBufferSRV
    };
    pDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);
    
    ID3D11ShaderResourceView *pSRVs2[] = 
    {
        m_ptex2DScrSpaceCloudTransparencySRV,
        m_ptex2DScrSpaceDistToCloudSRV,
        m_ptex2DScreenCloudColorSRV
    };
    pDeviceContext->PSSetShaderResources(11, _countof(pSRVs2), pSRVs2);

    RenderQuad(pDeviceContext, m_CombineWithBBTech);
}
