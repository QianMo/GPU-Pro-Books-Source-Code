#pragma once
#include "sascontrol.h"

namespace SasControl
{

	class Checkbox :
		public SasControl::Base
	{
		CDXUTCheckBox* checkbox;
	public:
		Checkbox(ID3DX11EffectVariable* variable, CDXUTDialog& ui, int& nextControlPos, int& nextControlId, std::map<int, SasControl::Base*>& idMap);
		~Checkbox(void);

		void apply();

	};

}