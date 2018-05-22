#ifndef __TINY_UTIL__H__
#define __TINY_UTIL__H__

#include "../Util/Color.h"
#include "../Util/Vector3.h"

#include <string>

class TiXmlElement;

class TinyUtil
{
private:
	template< typename T >
	struct AddElementImpl
	{
		static void do_add_element(TiXmlElement* element, const char* name, T param);
	};

	template< typename T >
	struct GetElementImpl
	{
		static T do_get_element(TiXmlElement* element);
	};

	template< typename T >
	static std::string ConvertToString(T _value);

public:
	template< typename T >
	static void AddElement(TiXmlElement* element, const char* name, T param);

	template< typename T >
	static T GetElement(TiXmlElement* element);
};

#endif

