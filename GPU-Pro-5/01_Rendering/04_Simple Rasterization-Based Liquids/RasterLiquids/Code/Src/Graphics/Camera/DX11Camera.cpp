
#include <Graphics/Camera/Camera.hpp>
#include <Graphics/Camera/DX11Camera.hpp>
#include <Input/Keyboard.hpp>

#include <Math/Math.hpp>
#include <Math/Vector/Vector.hpp>

#include <..\\External\\AntTweakBar\\include\\AntTweakBar.h>

///<
void Dx11Camera::Create(ID3D11Device* _pDevice)
{
	{
		D3D11_BUFFER_DESC bd;
		memset(&bd, 0, sizeof(bd));

		bd.Usage = D3D11_USAGE_DEFAULT;
		bd.ByteWidth = sizeof(CameraShaderData);
		bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		bd.CPUAccessFlags = 0;
		HRESULT hr = _pDevice->CreateBuffer(&bd, NULL, &m_pConstants);
		ASSERT(hr==S_OK, "Constants Failed!");	
	}

}

///<
void Dx11Camera::SetForLight(ID3D11DeviceContext* _pImmediateContext, const int32 _ViewProjReg)
{
	if (m_pConstants)
	{
		m_CamShaderData._View=Camera::Get().LightView();
		m_CamShaderData._LightView=Camera::Get().LightView();
		m_CamShaderData._ViewInv=AffineInverse(Camera::Get().View());
		m_CamShaderData._Proj=Camera::Get().Projection();
		
		Vector3f xPos = Camera::Get().X();
		m_CamShaderData._CamPos = Vector4f(m_CamShaderData._View(3,0),m_CamShaderData._View(3,1),m_CamShaderData._View(3,2),1);
		m_CamShaderData._ScreenDimensions=Vector4f((float32)Camera::Get().ScreenDimensions().x(),(float32)Camera::Get().ScreenDimensions().y(),0.0f,1.0f);

		_pImmediateContext->UpdateSubresource(m_pConstants, 0, NULL, &m_CamShaderData, 0, 0 );	
		_pImmediateContext->VSSetConstantBuffers(_ViewProjReg, 1, &m_pConstants);
		_pImmediateContext->GSSetConstantBuffers(_ViewProjReg, 1, &m_pConstants);
		_pImmediateContext->PSSetConstantBuffers(_ViewProjReg, 1, &m_pConstants);
	}
	
}

///<
void Dx11Camera::UpdateConstants(ID3D11DeviceContext* _pImmediateContext)
{	
	m_CamShaderData._View=Camera::Get().View();
	m_CamShaderData._LightView=Camera::Get().LightView();
	m_CamShaderData._ViewInv=AffineInverse(Camera::Get().View());
	m_CamShaderData._Proj=Camera::Get().Projection();
	
	Vector3f xPos = Camera::Get().X();
	m_CamShaderData._CamPos = Vector4f(xPos.x(),xPos.y(),xPos.z(),1);
	m_CamShaderData._ScreenDimensions=Vector4f(M::SCast<float32>(Camera::Get().ScreenDimensions().x()), M::SCast<float32>(Camera::Get().ScreenDimensions().y()),0.0f,1.0f);
	_pImmediateContext->UpdateSubresource(m_pConstants, 0, NULL, &m_CamShaderData, 0, 0 );	
}

///<
void Dx11Camera::SetParams(ID3D11DeviceContext* _pImmediateContext, const int32 _ViewProjReg)
{
	UpdateConstants(_pImmediateContext);

	_pImmediateContext->VSSetConstantBuffers(_ViewProjReg, 1, &m_pConstants);
	_pImmediateContext->GSSetConstantBuffers(_ViewProjReg, 1, &m_pConstants);
	_pImmediateContext->PSSetConstantBuffers(_ViewProjReg, 1, &m_pConstants);
}



