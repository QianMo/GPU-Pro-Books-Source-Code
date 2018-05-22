#include "DXUT.h"
#include "SasButton.h"

SasControl::Button::Button(CDXUTButton* button, Functor* functor)
	:Base(NULL)
{
	this->button = button;
	this->functor = functor;
}

SasControl::Button::~Button(void)
{
	delete functor;
}

void SasControl::Button::apply()
{
	if(variable == NULL)
		(*functor)();
}