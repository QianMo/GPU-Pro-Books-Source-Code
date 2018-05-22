#include "DXUT.h"
#include "SasSlider.h"
#include <sstream>

SasControl::Slider::Slider(ID3DX11EffectVariable* variable, CDXUTDialog& ui, int& nextControlPos, int& nextControlId, std::map<int, SasControl::Base*>& idMap)
	:Base(variable)
{
	std::wstringstream wss;
	float value;
	variable->AsScalar()->GetFloat(&value);
	wss << label << L": " << value;
	idMap[nextControlId] = this;
	ui.AddStatic( nextControlId++, wss.str().c_str(), 5, nextControlPos, 125, 22, false, &stat );
	nextControlPos += 24;
	float vmin, vmax;
	int nSteps;
	variable->GetAnnotationByName("SasUiMin")->AsScalar()->GetFloat(&vmin);
	variable->GetAnnotationByName("SasUiMax")->AsScalar()->GetFloat(&vmax);
	variable->GetAnnotationByName("SasUiSteps")->AsScalar()->GetInt(&nSteps);
	idMap[nextControlId] = this;
	ui.AddSlider( nextControlId++, 5, nextControlPos, 100, 22, 0, nSteps, nSteps * (value - vmin) / (vmax - vmin), false, &slider );
	nextControlPos += 24;
}

SasControl::Slider::~Slider(void)
{
}

void SasControl::Slider::apply()
{
	float vmin, vmax;
	int nSteps;
	variable->GetAnnotationByName("SasUiMin")->AsScalar()->GetFloat(&vmin);
	variable->GetAnnotationByName("SasUiMax")->AsScalar()->GetFloat(&vmax);
	variable->GetAnnotationByName("SasUiSteps")->AsScalar()->GetInt(&nSteps);

	variable->AsScalar()->SetFloat( slider->GetValue() / (float)nSteps * (vmax - vmin) + vmin);

	const char* desc = "<unlabeled>";
	variable->GetAnnotationByName("SasUiLabel")->AsString()->GetString(&desc);
	std::wstringstream wss;
	float value;
	variable->AsScalar()->GetFloat(&value);
	wss << desc << L": " << value;
	stat->SetText(wss.str().c_str());

}