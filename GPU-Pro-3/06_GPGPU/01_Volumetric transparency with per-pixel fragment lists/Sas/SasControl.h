#pragma once

#include "DXUTgui.h"
#include "d3dx11effect.h"
#include <map>

namespace SasControl
{

	class Base
	{
	protected:
		const char* label;
		ID3DX11EffectVariable* variable;
	public:
		Base(ID3DX11EffectVariable* variable):variable(variable)
		{	
			label = "<unlabeled>";
			if(variable)
				variable->GetAnnotationByName("SasUiLabel")->AsString()->GetString(&label);
		}
		virtual ~Base(){}
		virtual void apply()=0;
	};

}