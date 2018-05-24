//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------

#define MAX_LINKED_LIGHTS_PER_PIXEL 64
#define MAX_LLL_ELEMENTS            0xFFFFFF
#define MAX_LLL_BLAYERS             12
#define MAX_LLL_LIGHTS              256

#define MAX_CASCADES                8

//--------------------------------------------------------------------------------------
// Constant Buffers 
//--------------------------------------------------------------------------------------
#define CB_FRAME             0
#define CB_SIMPLE            1
#define CB_LIGHT_INSTANCES   1

#define CB_SHADOW_DATA       2

//--------------------------------------------------------------------------------------
// Textures 
//--------------------------------------------------------------------------------------
#define TEX_DIFFUSE          0 

#define TEX_DEPTH            0
#define TEX_NRM              1
#define TEX_COL              2

#define TEX_SHADOW           5

#define SRV_LIGHT_LINKED     10
#define SRV_LIGHT_OFFSET     11
#define SRV_LIGHT_ENV        12

//--------------------------------------------------------------------------------------
// Samplers 
//--------------------------------------------------------------------------------------
#define SAM_LINEAR           0 
#define SAM_POINT            1
#define SAM_SHADOW           5

//--------------------------------------------------------------------------------------
// UAVS 
//--------------------------------------------------------------------------------------
#define UAV_LIGHT_LINKED     3
#define UAV_LIGHT_OFFSET     4
#define UAV_LIGHT_BOUNDS     5
 
// The number of cascades 
#define CASCADE_COUNT_FLAG   3
#define CASCADE_BUFFER_SIZE  1536

// C/C++ side of things
#if !defined(__HLSL_SHADER__)
  typedef DirectX::XMFLOAT4X4  float4x4;
  typedef DirectX::XMFLOAT4X3  float4x3;

  typedef DirectX::XMFLOAT4    float4;
  typedef DirectX::XMFLOAT3    float3;
  typedef DirectX::XMINT4      int4;
  typedef uint32_t             uint;

  #define cbuffer              struct 

  #define B_REGISTER( reg_ )
  #define T_REGISTER( reg_ )
  #define S_REGISTER( reg_ )
  #define U_REGISTER( reg_ )

// HLSL
#else
  #pragma pack_matrix( row_major )

  #define B_REGISTER( reg_ ) : register(b##reg_)
  #define T_REGISTER( reg_ )   register(t##reg_)
  #define S_REGISTER( reg_ )   register(s##reg_)
  #define U_REGISTER( reg_ )   register(u##reg_)

#endif


//--------------------------------------------------------------------------
struct GPULightEnv
{
  float3    m_WorldPos;
  float     m_Radius;

  float3    m_LinearColor;
  float     m_SpecIntensity; 
};

//-------------------------------------------------------------------------
cbuffer FrameCB     B_REGISTER( CB_FRAME )
{
    float4x4        m_mViewToWorld;
    float4          m_vLinearDepthConsts;
    float4          m_vViewportToScreenUVs;
    float4          m_vViewportToClip;
    float4          m_vScreenToView;
    float4          m_vCameraPos;
    uint            m_iLLLWidth;
    uint            m_iLLLHeight;
    uint            m_iLLLMaxAlloc; 
    float           m_fLightPushScale;
};

//-------------------------------------------------------------------------
cbuffer ShadowDataCB B_REGISTER( CB_SHADOW_DATA )
{
    float4x4        m_mShadow;
    float4          m_vCascadeOffset[8];
    float4          m_vCascadeScale[8];
    int             m_nCascadeLevels; // Number of Cascades
    int             m_iVisualizeCascades; // 1 is to visualize the cascades in different colors. 0 is to just draw the scene
    int             m_iPCFBlurForLoopStart; // For loop begin value. For a 5x5 kernel this would be -2.
    int             m_iPCFBlurForLoopEnd; // For loop end value. For a 5x5 kernel this would be 3.

    // For Map based selection scheme, this keeps the pixels inside of the the valid range.
    // When there is no boarder, these values are 0 and 1 respectively.
    float           m_fMinBorderPadding;     
    float           m_fMaxBorderPadding;
    float           m_fShadowBiasFromGUI;  // A shadow map offset to deal with self shadow artifacts.  
                                           //These artifacts are aggravated by PCF.
    float           m_fShadowPartitionSize; 

    float           m_fCascadeBlendArea; // Amount to overlap when blending between cascades.
    float           m_fTexelSize; 
    float           m_fNativeTexelSizeInX;
    float           m_fPaddingForCB3; // Padding variables exist because CBs must be a multiple of 16 bytes. 

    float3          m_vLightDir;
    float           m_fPaddingCB4;
};

//-------------------------------------------------------------------------
struct LightInstance
{
  float4x4          m_WorldViewProj;
  float4x4          m_Reserved;
  float             m_LightIndex;
  float             m_Radius;
  float             m_PadA;
  float             m_PadB;
}; 

//-------------------------------------------------------------------------
// Must match LightInstance
cbuffer SimpleCB B_REGISTER( CB_SIMPLE )
{
  float4x4          g_SimpleWorldViewProj;
  float4x4          g_SimpleWorld;
  float             g_SimpleLightIndex;
  float             g_SimpleRadius;
  float             g_SimplePadA;
  float             g_SimplePadB;
};

cbuffer LightInstancesCB B_REGISTER( CB_LIGHT_INSTANCES )
{
  LightInstance     m_LightInstances[MAX_LLL_LIGHTS];
}; 