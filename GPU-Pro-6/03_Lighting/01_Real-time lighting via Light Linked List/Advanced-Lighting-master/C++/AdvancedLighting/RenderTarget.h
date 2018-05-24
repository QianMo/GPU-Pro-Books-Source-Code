#pragma once

#include <d3d11.h>
 
struct Texture
{
  Texture() : m_Resource(0),
              m_View(0)
  {
  }

  union
  {
   ID3D11Resource*  m_Resource;
   ID3D11Texture2D* m_Texture;
   ID3D11Texture1D* m_Texture1D;
   ID3D11Texture3D* m_Texture3D;
  };
  ID3D11ShaderResourceView*  m_View;

  void AddRef();
  void ReleaseResources();
};


//--------------------------------------------------------------------------------------------------
//
//  RenderTarget
//
//--------------------------------------------------------------------------------------------------

class RenderTarget
{
public:

  enum
  {
    kColorBufferMax = 4,
  };

  //--------------------------------------------------------------------------------------------------
  enum ClearFlags
  {
    kClearColor0               = 0x00000001,
    kClearColor1               = 0x00000002,
    kClearColor2               = 0x00000004,
    kClearColor3               = 0x00000008,
    kClearDepth                = 0x00000010,
    kClearStencil              = 0x00000020, 

    kClearColorAll             = (kClearColor0   | kClearColor1 | kClearColor2 | kClearColor3),
    kClearDepthStencil         = (kClearDepth    | kClearStencil),
    kClearAll                  = (kClearColorAll | kClearDepthStencil),
  };


  //--------------------------------------------------------------------------------------------------
  RenderTarget();

  //--------------------------------------------------------------------------------------------------
  ~RenderTarget()
  {
    Release();
  }

  //--------------------------------------------------------------------------------------------------
  inline bool           IsInitialized() const         { return m_Width != 0; }

  //--------------------------------------------------------------------------------------------------
  inline int            GetDepth()  const             { return m_Depth;  }

  //--------------------------------------------------------------------------------------------------
  inline int            GetWidth()  const             { return m_Width;  }

  //--------------------------------------------------------------------------------------------------
  inline int            GetHeight() const             { return m_Height; }

  //--------------------------------------------------------------------------------------------------
  inline int            GetColorBufferCount() const   { return m_ColorBufferCount;    }
 
  //--------------------------------------------------------------------------------------------------
  inline const Texture* GetColorTexture( int idx = 0 ) const
  {
    return  &m_ColorTex[idx];
  }

  //--------------------------------------------------------------------------------------------------
  inline const Texture* GetStencilTexture() const
  { 
    return &m_StencilTex;
  }

  //--------------------------------------------------------------------------------------------------
  inline const Texture* GetDepthTexture() const
  {
    return &m_DepthTex;
  }


  //--------------------------------------------------------------------------------------------------
  void Release();
  
  // Non-MRT initialization
  inline void  Init( int width, int height, DXGI_FORMAT rt_format, DXGI_FORMAT ds_format          )  { SharedInit( width, height, rt_format, NULL, ds_format, NULL);                 }
  inline void  Init( int width, int height, DXGI_FORMAT rt_format, ID3D11DepthStencilView* ds_view)  { SharedInit( width, height, rt_format, NULL, (DXGI_FORMAT)0, ds_view);         } 

  // MRT initialization
  inline void  Init( int width, int height, int rt_count, DXGI_FORMAT* rt_formats, DXGI_FORMAT             ds_format){ SharedInit( width, height, rt_count, rt_formats, ds_format, NULL        );  }  
  inline void  Init( int width, int height, int rt_count, DXGI_FORMAT* rt_formats, ID3D11DepthStencilView* ds_view)  { SharedInit( width, height, rt_count, rt_formats, (DXGI_FORMAT)0, ds_view);  }  

  // Accessors
  DXGI_FORMAT                             GetRTViewFormat( int index ) const;
  inline ID3D11RenderTargetView*          GetRTView( int index ) const   { return m_RTViews[ index ]; }
  inline ID3D11RenderTargetView* const *  GetRTViews() const             { return m_RTViews; }
  inline ID3D11DepthStencilView*          GetDSView() const              { return m_DSView; }

private:

  // Internal helpers
  void  InitColorTexture( int idx );
  void  InitColorBuffer( int idx, int width, int height, DXGI_FORMAT rt_format);
  void  InitColorBuffer( int idx, ID3D11RenderTargetView* rt_view );

  void  InitDepthStencilTexture( DXGI_FORMAT ds_format );
  void  InitDepthStencil( int width, int height, DXGI_FORMAT ds_format);
  void  InitDepthStencil( ID3D11DepthStencilView* ds_view );

  // Internal functions to handle initialization either through a format specification or a shared resource view
  void  SharedInit( int width, int height, DXGI_FORMAT rt_format, ID3D11RenderTargetView* rt_view, DXGI_FORMAT ds_format, ID3D11DepthStencilView* ds_view );
  void  SharedInit( int width, int height, int rt_count, DXGI_FORMAT* rt_formats, DXGI_FORMAT ds_format, ID3D11DepthStencilView* ds_viewe );

  // Platform specific members
  ID3D11RenderTargetView*   m_RTViews[ kColorBufferMax ];
  ID3D11DepthStencilView*   m_DSView;

  Texture   m_ColorTex[ kColorBufferMax ];
  Texture   m_StencilTex;
  Texture   m_DepthTex;      

  int32_t   m_ColorBufferCount;
  int32_t   m_Depth;
  int32_t   m_Width;
  int32_t   m_Height;
};

//--------------------------------------------------------------------------------------------------
// 
//  LightLinkedListTarget
// 
//--------------------------------------------------------------------------------------------------

class LightLinkedListTarget
{ 
public:
  LightLinkedListTarget();

  //---------------------------------------------------------------------------------------------
  bool  Init( int32_t full_width, int32_t full_height); 

  //---------------------------------------------------------------------------------------------
  void  Release();

  //---------------------------------------------------------------------------------------------                                       
  inline  ID3D11UnorderedAccessView*  GetFragmentLinkUAV()   const  { return m_FragmentLinkUAV;    }

  //---------------------------------------------------------------------------------------------                                       
  inline  ID3D11ShaderResourceView*   GetFragmentLinkSRV()   const  { return m_FragmentLinkSRV;    }

  //---------------------------------------------------------------------------------------------                                     
  inline  ID3D11UnorderedAccessView*  GetStartOffsetUAV()    const  { return m_StartOffsetUAV;     }

  //---------------------------------------------------------------------------------------------                                     
  inline  ID3D11ShaderResourceView*   GetStartOffsetSRV()    const  { return m_StartOffsetSRV;     }

  //---------------------------------------------------------------------------------------------                                     
  inline  ID3D11UnorderedAccessView*  GetBoundsUAV()         const  { return m_BoundsUAV;          }

  //---------------------------------------------------------------------------------------------
  inline const RenderTarget*          GetLinearDepthTarget()  const { return &m_LinearDepthTarget; }

  //---------------------------------------------------------------------------------------------
  inline const Texture*               GetLinearDepthTexture() const { return m_LinearDepthTarget.GetColorTexture(); }

  //----------------------------------------------------------------------------------------
  inline int                          GetWidth()  const             { return m_Width;                               }

  //---------------------------------------------------------------------------------------------
  inline int                          GetHeight() const             { return m_Height;                              }
  
  //---------------------------------------------------------------------------------------------
         uint32_t                     GetMaxLinkedElements() const;

  struct LightFragmentLink
  {
    uint32_t m_DepthInfo; // High bits min depth, low bits max depth
    uint32_t m_IndexLink; // Light index and link to the next fragment  
  };

private:
  // Linear depth
  RenderTarget               m_LinearDepthTarget;

  // Fragment And Link
  ID3D11Buffer*              m_FragmentLink;                // structured buffer
  ID3D11ShaderResourceView*  m_FragmentLinkSRV;             // srv
  ID3D11UnorderedAccessView* m_FragmentLinkUAV;             // uav

  // Start Offset
  ID3D11Buffer*              m_StartOffsetBuffer;           // RWByteAddress buffer
  ID3D11ShaderResourceView*  m_StartOffsetSRV;              // srv
  ID3D11UnorderedAccessView* m_StartOffsetUAV;              // uav

  // Bounds
  ID3D11Buffer*              m_BoundsBuffer;                // RWByteAddress buffer
  ID3D11UnorderedAccessView* m_BoundsUAV;                   // uav

  int32_t                    m_Width;
  int32_t                    m_Height;
};