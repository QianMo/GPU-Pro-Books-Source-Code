#include "dxut.h"
#include "ShaderShared.h"
#include "RenderTarget.h"

//--------------------------------------------------------------------------------------------------
RenderTarget::RenderTarget()
{
  for (int i = 0; i < kColorBufferMax; i++)
  {
    m_RTViews[i]            = NULL;
    m_ColorTex[i].m_View    = NULL;
    m_ColorTex[i].m_Texture = NULL;
  }

  m_DSView                = NULL;
  m_DepthTex.m_View       = NULL;
  m_DepthTex.m_Texture    = NULL;
  m_StencilTex.m_View     = NULL;
  m_StencilTex.m_Texture  = NULL;

  m_ColorBufferCount      = 0;
  m_Depth                 = 1;
  m_Width                 = 0;
  m_Height                = 0;
}

//--------------------------------------------------------------------------------------------------
void RenderTarget::InitColorTexture( int idx )
{
  ID3D11Device* d3d_device = DXUTGetD3D11Device();
  HRESULT       hr         = S_OK;

  if(m_Depth > 1)
  {
    D3D11_TEXTURE3D_DESC tdesc;
    m_ColorTex[idx].m_Texture3D->GetDesc(&tdesc);

    if (tdesc.BindFlags & D3D11_BIND_SHADER_RESOURCE)
    {
      D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
      srv_desc.Format                     = tdesc.Format;
      srv_desc.ViewDimension              = D3D11_SRV_DIMENSION_TEXTURE3D;
      srv_desc.Texture3D.MostDetailedMip  = 0;
      srv_desc.Texture3D.MipLevels        = 1;

      V( d3d_device->CreateShaderResourceView( m_ColorTex[idx].m_Texture, &srv_desc, &m_ColorTex[idx].m_View ) );
    }
  }
  else
  {
    // Initialize the texture resource view
    D3D11_TEXTURE2D_DESC tdesc;
    m_ColorTex[idx].m_Texture->GetDesc(&tdesc);

    if (tdesc.BindFlags & D3D11_BIND_SHADER_RESOURCE)
    {
      D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
      srv_desc.Format                      = tdesc.Format;
      srv_desc.ViewDimension               = D3D11_SRV_DIMENSION_TEXTURE2D;
      srv_desc.Texture2D.MostDetailedMip   = 0;
      srv_desc.Texture2D.MipLevels         = 1;

      V( d3d_device->CreateShaderResourceView( m_ColorTex[idx].m_Texture, &srv_desc, &m_ColorTex[idx].m_View ) );
    }
  }
}

//--------------------------------------------------------------------------------------------------
void RenderTarget::InitColorBuffer( int idx, int width, int height, DXGI_FORMAT rt_format)
{
  if (rt_format == DXGI_FORMAT_UNKNOWN)
  {
    return;
  }

  ID3D11Device* d3d_device = DXUTGetD3D11Device();
  HRESULT       hr         = S_OK;

  // Create texture resource
  D3D11_TEXTURE2D_DESC desc= { 0 };
  desc.Width               = width;
  desc.Height              = height;
  desc.MipLevels           = 1;
  desc.ArraySize           = 1;
  desc.Format              = rt_format;
  desc.SampleDesc.Count    = 1;
  desc.SampleDesc.Quality  = 0;
  desc.Usage               = D3D11_USAGE_DEFAULT;
  desc.BindFlags           = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
  desc.CPUAccessFlags      = 0;
  desc.MiscFlags           = 0;

  V( d3d_device->CreateTexture2D(&desc, NULL, &m_ColorTex[idx].m_Texture) );

  // Create render target view
  D3D11_RENDER_TARGET_VIEW_DESC view_desc;
  view_desc.Format               = desc.Format;
  view_desc.ViewDimension        = D3D11_RTV_DIMENSION_TEXTURE2D;
  view_desc.Texture2D.MipSlice  = 0;

  V( d3d_device->CreateRenderTargetView( m_ColorTex[idx].m_Texture, &view_desc, &m_RTViews[idx] ) );
  InitColorTexture( idx );
}


//--------------------------------------------------------------------------------------------------
void RenderTarget::InitColorBuffer( int idx, ID3D11RenderTargetView* rt_view )
{
  if (rt_view == NULL)
  {
    return;
  }

  // Validate that the d3d resources aren't already initialized
  assert( !m_RTViews[idx] && !m_ColorTex[idx].m_View && !m_ColorTex[idx].m_Texture );

  m_RTViews[idx] = rt_view;

  // Add a reference to the shared render target view
  rt_view->AddRef();

  // Access the texture resource from the shared render target view
  ID3D11Resource* resource;
  HRESULT         hr = S_OK;

  rt_view->GetResource(&resource);
  V( resource->QueryInterface( __uuidof(ID3D11Texture2D), (void **)&m_ColorTex[idx].m_Texture) );
  resource->Release();

  InitColorTexture( idx );
}


//--------------------------------------------------------------------------------------------------
void RenderTarget::InitDepthStencilTexture( DXGI_FORMAT ds_format )
{
  ID3D11Device* d3d_device = DXUTGetD3D11Device();
  HRESULT       hr         = S_OK;

  D3D11_TEXTURE2D_DESC tex_dsc;
  m_DepthTex.m_Texture->GetDesc(&tex_dsc);

  // Check if this render target can be used as a shader resource
  if ((tex_dsc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0)
  {
    m_DepthTex.m_View       = NULL;
    m_StencilTex.m_Texture  = NULL;
    m_StencilTex.m_View     = NULL;
    return;
  }

  // The srv format happens to always follow the dsv format in the dxgi enum
  DXGI_FORMAT srv_format = (DXGI_FORMAT)(ds_format + 1);

  // Create the depth texture shader resource view
  D3D11_SHADER_RESOURCE_VIEW_DESC desc_srv;
  desc_srv.Format                    = srv_format;
  desc_srv.ViewDimension             = D3D11_SRV_DIMENSION_TEXTURE2D;
  desc_srv.Texture2D.MipLevels       = 1;
  desc_srv.Texture2D.MostDetailedMip = 0;

  V(d3d_device->CreateShaderResourceView( m_DepthTex.m_Texture, &desc_srv, &m_DepthTex.m_View ) );

  // The stencil texture aliases the depth (only D24S8 actually has valid stencil information to be viewed)
  m_DepthTex.m_Texture->AddRef();
  m_StencilTex.m_Texture      = m_DepthTex.m_Texture;

  // Stencil default format
  DXGI_FORMAT stencil_format  = srv_format;

  // Check
  switch(ds_format)
  {
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
      stencil_format  = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    break;

   case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
      stencil_format  = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    break;
  }

  // Set the format
  desc_srv.Format        = stencil_format;
  V(d3d_device->CreateShaderResourceView( m_StencilTex.m_Texture, &desc_srv, &m_StencilTex.m_View ) );
}

//--------------------------------------------------------------------------------------------------
void RenderTarget::InitDepthStencil( int width, int height, DXGI_FORMAT ds_format)
{
  ID3D11Device* d3d_device = DXUTGetD3D11Device();
  HRESULT       hr         = S_OK;
 
  // Validate the format
  if (ds_format == DXGI_FORMAT_UNKNOWN)
  {
    return;
  }

  // Choose the texture format that corresponds to the depth-stencil format
  DXGI_FORMAT tex_format;
  switch (ds_format)
  {
    case DXGI_FORMAT_D32_FLOAT:
    {
      tex_format = DXGI_FORMAT_R32_TYPELESS;
      break;
    }

    case DXGI_FORMAT_D16_UNORM:
    {
      tex_format = DXGI_FORMAT_R16_TYPELESS;
      break;
    }

    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    {
      tex_format = DXGI_FORMAT_R32G8X24_TYPELESS;
      break;
    }

    default:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    {
      tex_format = DXGI_FORMAT_R24G8_TYPELESS;
      break;
    }
  }

  // Create the texture resource
  D3D11_TEXTURE2D_DESC desc = { 0 };
  desc.Width                = width;
  desc.Height               = height;
  desc.MipLevels            = 1;
  desc.ArraySize            = 1;
  desc.Format               = tex_format;
  desc.SampleDesc.Count     = 1;
  desc.SampleDesc.Quality   = 0;
  desc.Usage                = D3D11_USAGE_DEFAULT;
  desc.BindFlags            = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL;
  desc.CPUAccessFlags       = 0;
  desc.MiscFlags            = 0;

  V( d3d_device->CreateTexture2D( &desc, NULL, &m_DepthTex.m_Texture) );

  // Create the depth stencil view
  D3D11_DEPTH_STENCIL_VIEW_DESC dsvd;
  dsvd.Format             = ds_format;
  dsvd.ViewDimension      = D3D11_DSV_DIMENSION_TEXTURE2D;
  dsvd.Flags              = 0;
  dsvd.Texture2D.MipSlice = 0;

  V( d3d_device->CreateDepthStencilView( m_DepthTex.m_Texture, &dsvd, &m_DSView ));
  
  // Initialize the depth-stencil shader resource views
  InitDepthStencilTexture( ds_format );
}


//--------------------------------------------------------------------------------------------------
void RenderTarget::InitDepthStencil( ID3D11DepthStencilView* ds_view )
{
  if (ds_view == NULL)
  {
    return;
  }
  
  // Add a reference to the depth-stencil view
  m_DSView = ds_view;
  m_DSView->AddRef();

  // Query to find the depth-stencil format
  ID3D11Resource* resource;
  HRESULT         hr  = S_OK;

  m_DSView->GetResource(&resource);
  V( resource->QueryInterface( __uuidof(ID3D11Texture2D), (void **)&m_DepthTex.m_Texture) );
  resource->Release();

  D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc;
  m_DSView->GetDesc(&dsv_desc);

  // Initialize the depth-stencil shader resource views
  InitDepthStencilTexture( dsv_desc.Format );
}

//--------------------------------------------------------------------------------------------------
DXGI_FORMAT RenderTarget::GetRTViewFormat( int index ) const  
{ 
  // Validate
  if((index >= m_ColorBufferCount) || (index < 0))
  {
    return DXGI_FORMAT_UNKNOWN;
  }
  
  // Fetch the format
  D3D11_RENDER_TARGET_VIEW_DESC desc;
  m_RTViews[ index ]->GetDesc(&desc);

  // Done
  return desc.Format;
}


//--------------------------------------------------------------------------------------------------
void RenderTarget::SharedInit( int width, int height, DXGI_FORMAT rt_format, ID3D11RenderTargetView* rt_view, DXGI_FORMAT ds_format, ID3D11DepthStencilView* ds_view)
{
  Release();

  m_Depth            = 1;
  m_Width            = width;
  m_Height           = height;
  m_ColorBufferCount = 0;

  // Color
  if (rt_view)
  {
    m_ColorBufferCount = 1;
    InitColorBuffer( 0, rt_view );
  }
  else if (rt_format != DXGI_FORMAT_UNKNOWN)
  {
    m_ColorBufferCount = 1;
    InitColorBuffer( 0, width, height, rt_format );
  } 

  // Depth
  if (ds_view)
  {
    InitDepthStencil( ds_view );
  }
  else if (ds_format != DXGI_FORMAT_UNKNOWN)
  {
    InitDepthStencil( width, height, ds_format);
  }
}

//--------------------------------------------------------------------------------------------------
void RenderTarget::SharedInit( int width, int height, int rt_count, DXGI_FORMAT* rt_formats, DXGI_FORMAT ds_format, ID3D11DepthStencilView* ds_view)
{
  Release();

  m_Depth  = 1;
  m_Width  = width;
  m_Height = height;

  m_ColorBufferCount = rt_count;

  // Depth buffer first
  if (ds_view)
  {
    InitDepthStencil( ds_view );
  }
  else if (ds_format != DXGI_FORMAT_UNKNOWN)
  {
    InitDepthStencil( width, height, ds_format);
  }

  // Color buffers second
  for (int ibuffer = 0; ibuffer < rt_count; ++ibuffer)
  {
    InitColorBuffer( ibuffer, width, height, rt_formats[ibuffer]);
  }
}

//--------------------------------------------------------------------------------------------------
void RenderTarget::Release()
{
  if ( !IsInitialized() )
  {
    return;
  }

  SAFE_RELEASE(m_DSView);
  SAFE_RELEASE(m_DepthTex.m_Texture);
  SAFE_RELEASE(m_DepthTex.m_View);
  SAFE_RELEASE(m_StencilTex.m_Texture);
  SAFE_RELEASE(m_StencilTex.m_View);

  for (int i = 0; i < kColorBufferMax; i++)
  {
    SAFE_RELEASE(m_RTViews[i]);
    SAFE_RELEASE(m_ColorTex[i].m_Texture);
    SAFE_RELEASE(m_ColorTex[i].m_View); 
  }

  m_ColorBufferCount = 0;
  m_Depth            = 1;
  m_Width            = 0;
  m_Height           = 0;
}

//--------------------------------------------------------------------------------------------------
//
// LightLinkedListTarget
//
//--------------------------------------------------------------------------------------------------
LightLinkedListTarget::LightLinkedListTarget()
{
  // Fragment And Link
  m_FragmentLink          = NULL;
  m_FragmentLinkSRV       = NULL;
  m_FragmentLinkUAV       = NULL;
                                                
  // Start Offset                                     
  m_StartOffsetBuffer     = NULL;
  m_StartOffsetSRV        = NULL;
  m_StartOffsetUAV        = NULL;

  // Bounds                                 
  m_BoundsBuffer          = NULL; 
  m_BoundsUAV             = NULL;
}

//--------------------------------------------------------------------------------------------------
bool LightLinkedListTarget::Init(int32_t full_width, int32_t full_height)
{
  Release();

  uint32_t  edge_divide     = 8;

  // Set the correct width and height
  m_Height                 = full_height / edge_divide;
  m_Width                  = full_width  / edge_divide;

  int32_t       num_elements   = m_Width * m_Height;
  ID3D11Device* d3d_device     = DXUTGetD3D11Device();
  HRESULT       hr             = S_OK;

  // Initialize the linear depth target
  {
    m_LinearDepthTarget.Init(m_Width, m_Height, DXGI_FORMAT_R32_FLOAT, NULL);
  }

  // Buffer format
  DXGI_FORMAT       u32BufFormat = DXGI_FORMAT_R32_TYPELESS;

  // Bounds
  {
    // We create N (MAX_LLL_BLAYERS) layers to support geometry instancing
    uint32_t          bnum_elements = num_elements * MAX_LLL_BLAYERS;

    // Declare buffer               
    D3D11_BUFFER_DESC u32Buf        = { 0 };
    u32Buf.StructureByteStride      = sizeof( unsigned int );
    u32Buf.ByteWidth                = bnum_elements * u32Buf.StructureByteStride;
    u32Buf.MiscFlags                = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    u32Buf.BindFlags                = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
                                    
    // Create the buffer            
    hr                              = d3d_device->CreateBuffer(&u32Buf, NULL, &m_BoundsBuffer);

    // Validate
    if(hr != S_OK) {  assert( !"IGCreateRenderResource Failed" ); return false; }

    // create UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV;
    memset( &descUAV, 0, sizeof( descUAV ) );
    descUAV.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    descUAV.Buffer.FirstElement = 0;
    descUAV.Format              = u32BufFormat;
    descUAV.Buffer.NumElements  = bnum_elements;
    descUAV.Buffer.Flags        = D3D11_BUFFER_UAV_FLAG_RAW;

    hr                          = d3d_device->CreateUnorderedAccessView( m_BoundsBuffer, &descUAV, &m_BoundsUAV );

    // Validate
    if(hr != S_OK) {  assert( !"CreateUnorderedAccessView Failed"); return false; } 
  }

  // Offset
  {
    // Declare buffer
    D3D11_BUFFER_DESC u32Buf    = { 0 };
    u32Buf.StructureByteStride  = sizeof( unsigned int );
    u32Buf.ByteWidth            = num_elements * u32Buf.StructureByteStride;
    u32Buf.MiscFlags            = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    u32Buf.BindFlags            = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;

    // Create the buffer
    hr                          = d3d_device->CreateBuffer(&u32Buf, NULL, &m_StartOffsetBuffer);

    // Validate
    if(hr != S_OK) {  assert( !"IGCreateRenderResource Failed"); return false; }

    // create UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV;
    memset( &descUAV, 0, sizeof( descUAV ) );
    descUAV.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    descUAV.Buffer.FirstElement = 0;
    descUAV.Format              = u32BufFormat;
    descUAV.Buffer.NumElements  = num_elements;
    descUAV.Buffer.Flags        = D3D11_BUFFER_UAV_FLAG_RAW;
    hr                          = d3d_device->CreateUnorderedAccessView( m_StartOffsetBuffer, &descUAV, &m_StartOffsetUAV );

    // Validate
    if(hr != S_OK) {  assert( !"CreateUnorderedAccessView Failed"); return false; }

    // create SRV
    D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
    descSRV.ViewDimension       = D3D11_SRV_DIMENSION_BUFFER;
    descSRV.Buffer.FirstElement = 0;
    descSRV.Format              = DXGI_FORMAT_R32_UINT;
    descSRV.Buffer.NumElements  = num_elements;
    hr                          = d3d_device->CreateShaderResourceView( m_StartOffsetBuffer, &descSRV, &m_StartOffsetSRV );
    // Validate
    if(hr != S_OK) {  assert( !"CreateShaderResourceView Failed"); return false; }
  }
  
  // Create the LinkedList buffer
  { 
    uint32_t  min_byte_width    = GetMaxLinkedElements() * sizeof( LightFragmentLink );

    D3D11_BUFFER_DESC descBuf   = { 0 };
    descBuf.StructureByteStride = sizeof( LightFragmentLink );
    descBuf.ByteWidth           = min_byte_width;
    descBuf.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    descBuf.BindFlags           = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;

    // Create the buffer
    hr                          = d3d_device->CreateBuffer(&descBuf, NULL, &m_FragmentLink);

    // Validate
    if(hr != S_OK) {  assert( !"IGCreateRenderResource Failed"); return false; }

    // create UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV;
    memset( &descUAV, 0, sizeof( descUAV ) );
    descUAV.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    descUAV.Buffer.FirstElement = 0;
    descUAV.Format              = DXGI_FORMAT_UNKNOWN;
    descUAV.Buffer.NumElements  = num_elements  * MAX_LINKED_LIGHTS_PER_PIXEL;
    descUAV.Buffer.Flags        = D3D11_BUFFER_UAV_FLAG_COUNTER;
    hr                          = d3d_device->CreateUnorderedAccessView( m_FragmentLink, &descUAV, &m_FragmentLinkUAV );
    // Validate
    if(hr != S_OK) {  assert( !"CreateUnorderedAccessView Failed"); return false; }

    // create SRV
    D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
    descSRV.ViewDimension        = D3D11_SRV_DIMENSION_BUFFER;
    descSRV.Buffer.FirstElement  = 0;
    descSRV.Format               = DXGI_FORMAT_UNKNOWN;
    descSRV.Buffer.NumElements   = num_elements * MAX_LINKED_LIGHTS_PER_PIXEL;
    hr                           = d3d_device->CreateShaderResourceView( m_FragmentLink, &descSRV, &m_FragmentLinkSRV );
    // Validate
    if(hr != S_OK) {  assert( !"CreateShaderResourceView Failed"); return false; }
  }
 
  // Done
  return true;
}

//--------------------------------------------------------------------------------------------------
uint32_t  LightLinkedListTarget::GetMaxLinkedElements() const
{
  return (uint32_t)std::min(m_Height * m_Width * MAX_LINKED_LIGHTS_PER_PIXEL, MAX_LLL_ELEMENTS);
}

//--------------------------------------------------------------------------------------------------
void LightLinkedListTarget::Release()
{
  // Fragment And Link
  SAFE_RELEASE( m_FragmentLink      );
  SAFE_RELEASE( m_FragmentLinkSRV   );
  SAFE_RELEASE( m_FragmentLinkUAV   ); 
                                                                         
  // Start Offset                                                        
  SAFE_RELEASE( m_StartOffsetBuffer );
  SAFE_RELEASE( m_StartOffsetSRV    );
  SAFE_RELEASE( m_StartOffsetUAV    ); 

  // Bounds                                                      
  SAFE_RELEASE( m_BoundsBuffer    ); 
  SAFE_RELEASE( m_BoundsUAV       ); 

  // Clean up the linear depth target
  m_LinearDepthTarget.Release();

  // Fragment And Link
  m_FragmentLink          = NULL;
  m_FragmentLinkSRV       = NULL;
  m_FragmentLinkUAV       = NULL; 

  // Start Offset                                     
  m_StartOffsetBuffer     = NULL;
  m_StartOffsetSRV        = NULL;
  m_StartOffsetUAV        = NULL; 

  // Bounds                                 
  m_BoundsBuffer          = NULL; 
  m_BoundsUAV             = NULL; 
}