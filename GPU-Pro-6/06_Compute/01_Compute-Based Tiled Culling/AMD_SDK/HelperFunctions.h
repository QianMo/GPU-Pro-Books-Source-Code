//
// Copyright 2014 ADVANCED MICRO DEVICES, INC.  All Rights Reserved.
//
// AMD is granting you permission to use this software and documentation (if
// any) (collectively, the “Materials”) pursuant to the terms and conditions
// of the Software License Agreement included with the Materials.  If you do
// not have a copy of the Software License Agreement, contact your AMD
// representative for a copy.
// You agree that you will not reverse engineer or decompile the Materials,
// in whole or in part, except as allowed by applicable law.
//
// WARRANTY DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND.  AMD DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, THAT THE SOFTWARE
// WILL RUN UNINTERRUPTED OR ERROR-FREE OR WARRANTIES ARISING FROM CUSTOM OF
// TRADE OR COURSE OF USAGE.  THE ENTIRE RISK ASSOCIATED WITH THE USE OF THE
// SOFTWARE IS ASSUMED BY YOU.
// Some jurisdictions do not allow the exclusion of implied warranties, so
// the above exclusion may not apply to You. 
// 
// LIMITATION OF LIABILITY AND INDEMNIFICATION:  AMD AND ITS LICENSORS WILL
// NOT, UNDER ANY CIRCUMSTANCES BE LIABLE TO YOU FOR ANY PUNITIVE, DIRECT,
// INCIDENTAL, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF
// THE SOFTWARE OR THIS AGREEMENT EVEN IF AMD AND ITS LICENSORS HAVE BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  
// In no event shall AMD's total liability to You for all damages, losses,
// and causes of action (whether in contract, tort (including negligence) or
// otherwise) exceed the amount of $100 USD.  You agree to defend, indemnify
// and hold harmless AMD and its licensors, and any of their directors,
// officers, employees, affiliates or agents from and against any and all
// loss, damage, liability and other expenses (including reasonable attorneys'
// fees), resulting from Your use of the Software or violation of the terms and
// conditions of this Agreement.  
//
// U.S. GOVERNMENT RESTRICTED RIGHTS: The Materials are provided with "RESTRICTED
// RIGHTS." Use, duplication, or disclosure by the Government is subject to the
// restrictions as set forth in FAR 52.227-14 and DFAR252.227-7013, et seq., or
// its successor.  Use of the Materials by the Government constitutes
// acknowledgement of AMD's proprietary rights in them.
// 
// EXPORT RESTRICTIONS: The Materials may be subject to export restrictions as
// stated in the Software License Agreement.
//

//--------------------------------------------------------------------------------------
// File: HelperFunctions.h
//
// Various helper functions
//--------------------------------------------------------------------------------------
#ifndef __AMD_HELPER_FUNCTIONS_H__
#define __AMD_HELPER_FUNCTIONS_H__


namespace AMD
{

#define ARRAY_LENGTH( arry ) sizeof( arry ) / sizeof( arry[ 0 ] )

// Cmd line params structure
typedef struct _CmdLineParams
{
	D3D_DRIVER_TYPE DriverType;
	unsigned int uWidth;
	unsigned int uHeight;
	bool bCapture;
	WCHAR strCaptureFilename[256];
	int iExitFrame;
	bool bRenderHUD;
}CmdLineParams;


//--------------------------------------------------------------------------------------
// Utility function that can optionally create the following objects:
// ID3D11Texture2D (only does this if the pointer is NULL)
// ID3D11ShaderResourceView
// ID3D11RenderTargetView
// ID3D11UnorderedAccessView
//--------------------------------------------------------------------------------------
HRESULT CreateSurface( ID3D11Texture2D** ppTexture, ID3D11ShaderResourceView** ppTextureSRV,
					   ID3D11RenderTargetView** ppTextureRTV, ID3D11UnorderedAccessView** ppTextureUAV, 
					   DXGI_FORMAT Format, unsigned int uWidth, unsigned int uHeight, unsigned int uSampleCount );


//--------------------------------------------------------------------------------------
// Creates a depth stencil surface and optionally creates the following objects:
// ID3D11ShaderResourceView
// ID3D11DepthStencilView
//--------------------------------------------------------------------------------------
HRESULT CreateDepthStencilSurface( ID3D11Texture2D** ppDepthStencilTexture, ID3D11ShaderResourceView** ppDepthStencilSRV,
                                   ID3D11DepthStencilView** ppDepthStencilView, DXGI_FORMAT DSFormat, DXGI_FORMAT SRVFormat, 
                                   unsigned int uWidth, unsigned int uHeight, unsigned int uSampleCount );


//--------------------------------------------------------------------------------------
// Capture a frame and dump it to disk 
//--------------------------------------------------------------------------------------
void CaptureFrame( ID3D11Texture2D* pCaptureTexture, WCHAR* pszCaptureFileName );


//--------------------------------------------------------------------------------------
// Allows the app to render individual meshes of an sdkmesh
// and override the primitive topology (useful for tessellated rendering of SDK meshes )
//--------------------------------------------------------------------------------------
void RenderMesh( CDXUTSDKMesh* pDXUTMesh, UINT uMesh, 
                D3D11_PRIMITIVE_TOPOLOGY PrimType = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED, 
				UINT uDiffuseSlot = INVALID_SAMPLER_SLOT, UINT uNormalSlot = INVALID_SAMPLER_SLOT, 
                UINT uSpecularSlot = INVALID_SAMPLER_SLOT );


//--------------------------------------------------------------------------------------
// Debug function which copies a GPU buffer to a CPU readable buffer
//--------------------------------------------------------------------------------------
ID3D11Buffer* CreateAndCopyToDebugBuf( ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, 
                                      ID3D11Buffer* pBuffer );


//--------------------------------------------------------------------------------------
// Helper function to compile an hlsl shader from file, 
// its binary compiled code is returned
//--------------------------------------------------------------------------------------
HRESULT CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, 
							   LPCSTR szShaderModel, ID3DBlob** ppBlobOut, D3D10_SHADER_MACRO* pDefines );


//--------------------------------------------------------------------------------------
// Helper function for command line retrieval
//--------------------------------------------------------------------------------------
bool IsNextArg( WCHAR*& strCmdLine, WCHAR* strArg );


//--------------------------------------------------------------------------------------
// Helper function for command line retrieval.  Updates strCmdLine and strFlag 
//      Example: if strCmdLine=="-width:1024 -forceref"
// then after: strCmdLine==" -forceref" and strFlag=="1024"
//--------------------------------------------------------------------------------------
bool GetCmdParam( WCHAR*& strCmdLine, WCHAR* strFlag );


//--------------------------------------------------------------------------------------
// Helper function to parse the command line
//--------------------------------------------------------------------------------------
void ParseCommandLine( CmdLineParams* pCmdLineParams );


//--------------------------------------------------------------------------------------
// Check for file existance
//--------------------------------------------------------------------------------------
bool FileExists( WCHAR* pFileName );


//--------------------------------------------------------------------------------------
// Creates a cube with vertex format of 
// 32x3		Position
// 32x3		Normal
// 16x2		UV
// 28 byte stride
//--------------------------------------------------------------------------------------
void CreateCube( float fSize, ID3D11Buffer** ppVertexBuffer, ID3D11Buffer** ppIndexBuffer );


//--------------------------------------------------------------------------------------
// Convert single-precision float to half-precision float, 
// returned as a 16-bit unsigned value
//--------------------------------------------------------------------------------------
unsigned short ConvertF32ToF16(float fValueToConvert);


} // namespace AMD


#endif
//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
