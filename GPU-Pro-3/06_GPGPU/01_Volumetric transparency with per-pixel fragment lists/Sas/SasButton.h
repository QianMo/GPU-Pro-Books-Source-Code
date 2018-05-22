#pragma once
#include "sascontrol.h"

namespace SasControl
{

	class Button :
		public SasControl::Base
	{
	public:
		class Functor
		{
		public:
			virtual void operator()()=0;
		};
	protected:
		CDXUTButton* button;
		Functor* functor;

	public:
		Button(CDXUTButton* button, Functor* functor);
		~Button(void);

		void apply();
	};

}