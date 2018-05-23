

#ifndef __DX11_CAMERA_HPP__
#define __DX11_CAMERA_HPP__

#include <d3dx11.h>
#include <d3d11.h>

#include <Common/Common.hpp>
#include <Common/Incopiable.hpp>
#include <Math\Matrix\Matrix.hpp>

struct CameraShaderData
{
	Vector4f _ScreenDimensions;
	Vector4f _CamPos;
	Matrix4f _View;
	Matrix4f _Proj;
	Matrix4f _ViewInv;

	Matrix4f _LightView;
};

///<
class Dx11Camera  : public Incopiable
{
	ID3D11Buffer*		m_pConstants;

	CameraShaderData	m_CamShaderData;

public:

	Dx11Camera() : m_pConstants(NULL){}
	///<
	~Dx11Camera(){ M::Release(&m_pConstants); }

	///<
	void Create										(ID3D11Device* _pDevice);
	void UpdateConstants							(ID3D11DeviceContext* _pImmediateContext);

	void SetParams									(ID3D11DeviceContext* _pImmediateContext, const int32 _ViewProjReg);
	void SetForLight								(ID3D11DeviceContext* _pImmediateContext, const int32 _ViewProjReg);

};


#endif