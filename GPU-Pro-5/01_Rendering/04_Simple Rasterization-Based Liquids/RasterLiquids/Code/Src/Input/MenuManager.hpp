

#ifndef __MENU_MANAGER_HPP__
#define __MENU_MANAGER_HPP__

#include <Common/Incopiable.hpp>
#include <Common/Common.hpp>

#include <Math/Matrix/Matrix.hpp>

#include <d3dx11.h>
#include <d3d11.h>

#include <..\\External\\AntTweakBar\\include\\AntTweakBar.h>

///<
class MenuManager : public Incopiable
{

	int32		m_iNumMenus;
	
	Vector2ui	m_uiWindowsSize;

	static MenuManager m_inst;
	

public:

	///<
	static MenuManager& Get() { return m_inst; }
	///<
	MenuManager(){ memset(this,0,sizeof(MenuManager)); }

	///<
	bool	Create	(ID3D11Device* _pDevice, const Vector2ui _uiWindowSize);
	TwBar*	AddBar	(const char* _csName, const uint32 _iNumElements);

	///<
	void Update();
	///<
	void Release();

};


#endif