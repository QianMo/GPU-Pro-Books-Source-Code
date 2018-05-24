#pragma once

struct Vec2
{
    Vec2()
    {
    }

    Vec2( float x_, float y_ )
        : x( x_ )
        , y( y_ )
    {
    }

    float x;
    float y;
};

struct Vec3
{
    float x;
    float y;
    float z;
};

UINT const MAX_QUERY_FRAME_NUM = 5;

class CApp
{
public:
    CApp();
    ~CApp();

    bool Init( HWND windowHandle );
    void Release();
    void Render();
    void OnKeyDown( WPARAM wParam );
    void OnLButtonDown( int mouseX, int mouseY );
    void OnLButtonUp( int mouseX, int mouseY );
    void OnMouseMove( int mouseX, int mouseY );
    void OnMouseWheel( int zDelta );
    void OnResize();

    ID3D11Device*           GetDevice()     { return m_device; }
    ID3D11DeviceContext*    GetCtx()        { return m_ctx; }


private:
    unsigned                        m_backbufferWidth;
    unsigned                        m_backbufferHeight;
    ID3D11Device*                   m_device;
    ID3D11DeviceContext*            m_ctx;
    IDXGISwapChain*                 m_swapChain;
    ID3D11RenderTargetView*         m_backBufferView;
    ID3D11SamplerState*             m_pointSampler;
    ID3D11Buffer*                   m_constantBuffer;

    ID3D11Query*                    m_disjointQueries[ MAX_QUERY_FRAME_NUM ];
    ID3D11Query*                    m_timeBeginQueries[ MAX_QUERY_FRAME_NUM ];
    ID3D11Query*                    m_timeEndQueries[ MAX_QUERY_FRAME_NUM ];
    float                           m_timeAcc;
    unsigned                        m_timeAccSampleNum;
    float                           m_compressionTime;

    ID3D11VertexShader*             m_blitVS;
    ID3D11PixelShader*              m_blitPS;
    ID3D11VertexShader*             m_compressVS;
    ID3D11PixelShader*              m_compressFastPS;
    ID3D11PixelShader*              m_compressQualityPS;

    ID3D11Buffer*                   m_ib;
    ID3D11Texture2D*                m_srcTextureRes;
    ID3D11ShaderResourceView*       m_srcTextureView;
    ID3D11Texture2D*                m_dstTextureRes;
    ID3D11ShaderResourceView*       m_dstTextureView;
    ID3D11Texture2D*                m_compressTargetRes;
    ID3D11RenderTargetView*         m_compressTargetView;

    ID3D11Texture2D*                m_tmpTargetRes;
    ID3D11RenderTargetView*         m_tmpTargetView;
    ID3D11Texture2D*                m_tmpStagingRes;

    HWND                            m_windowHandle;
    Vec2                            m_texelBias;
    float                           m_texelScale;
    float                           m_imageZoom;
    float                           m_imageExposure;
    bool                            m_dragEnabled;
    Vec2                            m_dragStart;
    bool                            m_showCompressed;
    bool                            m_qualityMode;
    bool                            m_updateRMSE;
    bool                            m_updateTitle;
    unsigned                        m_imageID;
    unsigned                        m_imageWidth;
    unsigned                        m_imageHeight;
    uint64_t                        m_frameID;
    float                           m_rmsle;

    void CreateImage();
    void DestoryImage();
    void CreateShaders();
    void DestroyShaders();
    void CreateTargets();
    void DestroyTargets();
    void CreateQueries();
    void CreateConstantBuffer();
    void UpdateRMSE();
    void UpdateTitle();
    void CopyTexture( Vec3* image, ID3D11ShaderResourceView* srcView );
};

extern CApp gApp;