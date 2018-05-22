#pragma once
#include "sascontrol.h"

namespace SasControl
{
	class Slider :
		public SasControl::Base
	{
		CDXUTSlider* slider;
		CDXUTStatic* stat;
	public:
		Slider(ID3DX11EffectVariable* variable, CDXUTDialog& ui, int& nextControlPos, int& nextControlId, std::map<int, SasControl::Base*>& idMap);
		~Slider(void);

		void apply();
	};

}