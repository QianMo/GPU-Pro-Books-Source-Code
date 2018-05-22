#include "DXUT.h"
#include "SasCheckbox.h"
#include <sstream>

SasControl::Checkbox::Checkbox(ID3DX11EffectVariable* variable, CDXUTDialog& ui, int& nextControlPos, int& nextControlId, std::map<int, SasControl::Base*>& idMap)
	:Base(variable)
{
	std::wstringstream wss;
	float value;
	variable->AsScalar()->GetFloat(&value);
	wss << label;
	idMap[nextControlId] = this;
	ui.AddCheckBox( nextControlId++, wss.str().c_str(), 5, nextControlPos, 100, 22, value, 0, false, &checkbox);
	nextControlPos += 24;
}

SasControl::Checkbox::~Checkbox(void)
{
}

void SasControl::Checkbox::apply()
{

	variable->AsScalar()->SetBool( checkbox->GetChecked() );

}
