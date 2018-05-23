
#include <Input/MenuManager.hpp>

#include <d3dx11.h>
#include <d3d11.h>

MenuManager MenuManager::m_inst;

///<
bool MenuManager::Create(ID3D11Device* _pDevice, const Vector2ui _uiWindowSize)
{
	m_uiWindowsSize = _uiWindowSize;

	m_iNumMenus=1;

	int32 iAntTweak = TwInit(TW_DIRECT3D11, _pDevice);

	ASSERT(iAntTweak!=0, "Failed AntTweak!");

	if (iAntTweak!=0)
	{
		TwBar *bar = TwNewBar("Info");
	
		TwDefine(" Info size='250 100' position='0 0' color='255 100 200' refresh='0.0333' "); 
		
		TwWindowSize(m_uiWindowsSize.x(), m_uiWindowsSize.y());

		TwAddVarRO(bar, "Screen Width", TW_TYPE_UINT32, &m_uiWindowsSize[0], " label='Frame Buffer Width' ");
		TwAddVarRO(bar, "Screen Height", TW_TYPE_UINT32, &m_uiWindowsSize[1], " label='Frame Buffer Height' ");

		return true;
	}

	return false;
}

///<
TwBar* MenuManager::AddBar(const char* _csName, const uint32 _iNumElements)
{
	TwBar* pBarPrevious = TwGetBarByIndex(m_iNumMenus);

	Vector2i iMenuSize, iMenuPosition;
	TwGetParam(pBarPrevious, NULL, "size", TW_PARAM_INT32, 2, iMenuSize.Begin());
	TwGetParam(pBarPrevious, NULL, "position", TW_PARAM_INT32, 2, iMenuPosition.Begin());
		
	TwBar* pBar = TwGetBarByName(_csName);
	if (pBar==NULL)
	{
		pBar = TwNewBar(_csName);
		Vector2i newPos = iMenuPosition + Vector2i(0, iMenuSize.y());
		TwSetParam(pBar, NULL, "position", TW_PARAM_INT32, 2, newPos.Begin());	
		iMenuSize[1]=28*_iNumElements;
		TwSetParam(pBar, NULL, "size", TW_PARAM_INT32, 2, iMenuSize.Begin());	

		m_iNumMenus++;
	}

	return pBar;
}

