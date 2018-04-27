#ifndef COMMON_XIELEMARRAY_H_INCLUDED
#define COMMON_XIELEMARRAY_H_INCLUDED

#include "XIArrayConvFunctors.h"

namespace Mod
{
	template	<	
				typename T,							// contained type
				typename ConvF	= XIAttribConv<T>,	// conversion functor (TiXmlElem* -> T)
				typename Cont	= std::vector<T>	// container to derive from
				>
	class XIElemArray : public Cont
	{
		// construction/ destruction
	public:

		XIElemArray( const Cont& def ):
		Cont(def)
		{

		}

		XIElemArray( const XMLElemPtr& parentElem, const String& elemName, const String& subElemName, const ConvF& convF = ConvF())
		{
			MD_FERROR_ON_FALSE( Build(parentElem, elemName, subElemName, convF ) );
		}

		XIElemArray( const XMLElemPtr& parentElem, const String& subElemName, const ConvF& convF = ConvF())
		{
			BuildFromElem(parentElem, subElemName, convF ); 
		}

		XIElemArray( const XMLElemPtr& parentElem, const String& elemName, const String& subElemName, const Cont& def, const ConvF& convF = ConvF())
		{
			if(!Build(parentElem, elemName, subElemName, convF))
				Cont::operator = (def);
		}

		XIElemArray( const XMLElemPtr& parentElem, const ConvF& convF = ConvF())
		{
			MD_FERROR_ON_FALSE( Build(parentElem, convF) );
		}

		// helpers
	private:
		bool Build(const XMLElemPtr& parentElem, const String& elemName, const String& subElemName, const ConvF& convF )
		{
			const XMLElemPtr& elem = parentElem->GetChild(elemName);

			if(!elem)
				return false;		
			BuildFromElem(elem, subElemName, convF);

			return true;
		}

		bool Build(const XMLElemPtr& parentElem, const ConvF& convF )
		{

			if(!parentElem)
				return false;		
			BuildFromElem(parentElem, convF);

			return true;
		}

		void BuildFromElem(const XMLElemPtr& elem, const String& subElemName, const ConvF& convF )
		{
			INT32 idx = 0;
			while(const XMLElemPtr& subElem = elem->GetChild( subElemName, idx++ ))
			{
				insert(end(), convF(subElem));
			}
		}

		void BuildFromElem(const XMLElemPtr& elem, const ConvF& convF )
		{
			typedef XMLElem::Children Elems;
			const Elems& children = elem->GetChildren();

			for( Elems::const_iterator i = children.begin(), e = children.end(); i != e; ++i )
			{
				insert( end(), convF( *i ) );
			}
		}

	};

	typedef XIElemArray< String, XIStringConvertibleElemConv<String> > XIStrElemArray;

	template	<	
				typename T,							// contained type
				typename ConvF	= XIAttribConv<T>,	// conversion functor (TiXmlElem* -> T)
				typename Cont	= std::vector<T>	// container to derive from
				>
	class XIElemAttribArray : public XIElemArray<T>
	{
		// construction/ destruction
	public:
		XIElemAttribArray ( const XMLElemPtr& parentElem, const String& elemName, const String& subElemName, const String& attName ):
		XIElemArray( parentElem, elemName, subElemName, XIAttribConv<T>(attName))
		{}

		XIElemAttribArray ( const XMLElemPtr& parentElem, const String& elemName, const String& subElemName, const String& attName, const Cont& def ):
		XIElemArray( parentElem, elemName, subElemName, def, XIAttribConv<T>(attName))
		{}

		XIElemAttribArray ( const XMLElemPtr& elem, const String& subElemName, const String& attName ):
		XIElemArray( elem, subElemName, XIAttribConv<T>(attName))
		{}

	};

	typedef XIElemAttribArray<String> XIStrArray;

}

#endif