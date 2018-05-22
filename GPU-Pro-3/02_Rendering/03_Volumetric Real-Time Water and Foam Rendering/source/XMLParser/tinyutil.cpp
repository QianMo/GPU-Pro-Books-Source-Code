
#include "../XMLParser/tinyutil.h"
#include "../XMLParser/tinyxml.h"


// -----------------------------------------------------------------------------
// ---------------------------- TinyUtil::AddElement ---------------------------
// -----------------------------------------------------------------------------
template< typename T >
void TinyUtil::AddElement(TiXmlElement* element, const char* name, T param)
{
	return AddElementImpl<T>::do_add_element(element, name, param);
}

// -----------------------------------------------------------------------------
// ------------------------ TinyUtil::ConvertFromString ------------------------
// -----------------------------------------------------------------------------
template< typename T >
T TinyUtil::GetElement(TiXmlElement* element)
{
	return GetElementImpl<T>::do_get_element(element);
}

// -----------------------------------------------------------------------------
// ---------------- TinyUtil::AddElementImpl<T>::do_add_element ----------------
// -----------------------------------------------------------------------------
template< typename T >
void TinyUtil::AddElementImpl<T>::do_add_element(TiXmlElement* element, const char* name, T param)
{
	TiXmlElement* e = new TiXmlElement(name);
	element->LinkEndChild(e);

	TiXmlText* text = new TiXmlText(ConvertToString<T>(param));
	e->LinkEndChild(text);
}

// -----------------------------------------------------------------------------
// --------- TinyUtil::ConvertFromStringImpl<T>::do_convert_from_string --------
// -----------------------------------------------------------------------------
template< typename T >
T TinyUtil::GetElementImpl<T>::do_get_element(TiXmlElement* element)
{
	assert(false);
	return;
}

// -----------------------------------------------------------------------------
// --------------- TinyUtil::AddElementImpl<int>::do_add_element ---------------
// -----------------------------------------------------------------------------
template<>
void TinyUtil::AddElementImpl<int>::do_add_element(TiXmlElement* element, const char* name, int param)
{
	TiXmlElement* e = new TiXmlElement(name);
	element->LinkEndChild(e);
	e->SetAttribute("type", "int");

	TiXmlText* text = new TiXmlText(ConvertToString<int>(param));
	e->LinkEndChild(text);
}

// -----------------------------------------------------------------------------
// --------------- TinyUtil::AddElementImpl<int>::do_add_element ---------------
// -----------------------------------------------------------------------------
template<>
void TinyUtil::AddElementImpl<unsigned int>::do_add_element(TiXmlElement* element, const char* name, unsigned int param)
{
	TiXmlElement* e = new TiXmlElement(name);
	element->LinkEndChild(e);
	e->SetAttribute("type", "unsigned int");

	TiXmlText* text = new TiXmlText(ConvertToString<unsigned int>(param));
	e->LinkEndChild(text);
}

// -----------------------------------------------------------------------------
// -------------- TinyUtil::AddElementImpl<float>::do_add_element --------------
// -----------------------------------------------------------------------------
template<>
void TinyUtil::AddElementImpl<float>::do_add_element(TiXmlElement* element, const char* name, float param)
{
	TiXmlElement* e = new TiXmlElement(name);
	element->LinkEndChild(e);
	e->SetAttribute("type", "float");

	TiXmlText* text = new TiXmlText(ConvertToString<float>(param));
	e->LinkEndChild(text);
}

// -----------------------------------------------------------------------------
// -------------- TinyUtil::AddElementImpl<Color>::do_add_element --------------
// -----------------------------------------------------------------------------
template<>
void TinyUtil::AddElementImpl<Color>::do_add_element(TiXmlElement* element, const char* name, Color param)
{
	TiXmlElement* e = new TiXmlElement(name);
	element->LinkEndChild(e);
	e->SetAttribute("type", "Color");

	TiXmlElement* elementRed = new TiXmlElement("R");
	e->LinkEndChild(elementRed);
	TiXmlText* textRed = new TiXmlText(ConvertToString<float>(param.r));
	elementRed->LinkEndChild(textRed);

	TiXmlElement* elementGreen = new TiXmlElement("G");
	e->LinkEndChild(elementGreen);
	TiXmlText* textGreen = new TiXmlText(ConvertToString<float>(param.g));
	elementGreen->LinkEndChild(textGreen);

	TiXmlElement* elementBlue = new TiXmlElement("B");
	e->LinkEndChild(elementBlue);
	TiXmlText* textBlue = new TiXmlText(ConvertToString<float>(param.b));
	elementBlue->LinkEndChild(textBlue);

	TiXmlElement* elementAlpha = new TiXmlElement("A");
	e->LinkEndChild(elementAlpha);
	TiXmlText* textAlpha = new TiXmlText(ConvertToString<float>(param.a));
	elementAlpha->LinkEndChild(textAlpha);
}

// -----------------------------------------------------------------------------
// ------------- TinyUtil::AddElementImpl<Vector3>::do_add_element -------------
// -----------------------------------------------------------------------------
template<>
void TinyUtil::AddElementImpl<Vector3>::do_add_element(TiXmlElement* element, const char* name, Vector3 param)
{
	TiXmlElement* e = new TiXmlElement(name);
	element->LinkEndChild(e);
	e->SetAttribute("type", "Vector3");

	TiXmlElement* elementX = new TiXmlElement("X");
	e->LinkEndChild(elementX);
	TiXmlText* textX = new TiXmlText(ConvertToString<float>(param.x));
	elementX->LinkEndChild(textX);

	TiXmlElement* elementY = new TiXmlElement("Y");
	e->LinkEndChild(elementY);
	TiXmlText* textY = new TiXmlText(ConvertToString<float>(param.y));
	elementY->LinkEndChild(textY);

	TiXmlElement* elementZ = new TiXmlElement("Z");
	e->LinkEndChild(elementZ);
	TiXmlText* textZ = new TiXmlText(ConvertToString<float>(param.z));
	elementZ->LinkEndChild(textZ);
}

template void TinyUtil::AddElement<int>(TiXmlElement*, const char*, int);
template void TinyUtil::AddElement<unsigned int>(TiXmlElement*, const char*, unsigned int);
template void TinyUtil::AddElement<float>(TiXmlElement*, const char*, float);
template void TinyUtil::AddElement<bool>(TiXmlElement*, const char*, bool);
template void TinyUtil::AddElement<Color>(TiXmlElement*, const char*, Color);
template void TinyUtil::AddElement<Vector3>(TiXmlElement*, const char*, Vector3);

// -----------------------------------------------------------------------------
// --------------- TinyUtil::GetElementImpl<int>::do_get_element ---------------
// -----------------------------------------------------------------------------
template<>
int TinyUtil::GetElementImpl<int>::do_get_element(TiXmlElement* element)
{
	std::istringstream i(element->GetText());
	int x;
	if (!(i >> x))
		assert(false);

	return x;
}

// -----------------------------------------------------------------------------
// ----------- TinyUtil::GetElementImpl<unsigned int>::do_get_element ----------
// -----------------------------------------------------------------------------
template<>
unsigned int TinyUtil::GetElementImpl<unsigned int>::do_get_element(TiXmlElement* element)
{
	std::istringstream i(element->GetText());
	unsigned int x;
	if (!(i >> x))
		assert(false);

	return x;
}

// -----------------------------------------------------------------------------
// -------------- TinyUtil::GetElementImpl<float>::do_get_element --------------
// -----------------------------------------------------------------------------
template<>
float TinyUtil::GetElementImpl<float>::do_get_element(TiXmlElement* element)
{
	std::istringstream i(element->GetText());
	float x;
	if (!(i >> x))
		assert(false);

	return x;
}

// -----------------------------------------------------------------------------
// --------------- TinyUtil::GetElementImpl<bool>::do_get_element --------------
// -----------------------------------------------------------------------------
template<>
bool TinyUtil::GetElementImpl<bool>::do_get_element(TiXmlElement* element)
{
	std::istringstream i(element->GetText());
	bool x;
	if (!(i >> x))
		assert(false);

	return x;
}

// -----------------------------------------------------------------------------
// -------------- TinyUtil::GetElementImpl<Color>::do_get_element --------------
// -----------------------------------------------------------------------------
template<>
Color TinyUtil::GetElementImpl<Color>::do_get_element(TiXmlElement* element)
{
	Color color;
	float x;

	{
		std::istringstream i(element->FirstChild("R")->ToElement()->GetText());
		if (!(i >> x))
			assert(false);
		color.r = x;
	}
	{
		std::istringstream i(element->FirstChild("G")->ToElement()->GetText());
		if (!(i >> x))
			assert(false);
		color.g = x;
	}
	{
		std::istringstream i(element->FirstChild("B")->ToElement()->GetText());
		if (!(i >> x))
			assert(false);
		color.b = x;
	}
	{
		std::istringstream i(element->FirstChild("A")->ToElement()->GetText());
		if (!(i >> x))
			assert(false);
		color.a = x;
	}

	return color;
}

// -----------------------------------------------------------------------------
// ------------- TinyUtil::GetElementImpl<Vector3>::do_get_element -------------
// -----------------------------------------------------------------------------
template<>
Vector3 TinyUtil::GetElementImpl<Vector3>::do_get_element(TiXmlElement* element)
{
	Vector3 vector3;
	float x;

	{
		std::istringstream i(element->FirstChild("X")->ToElement()->GetText());
		if (!(i >> x))
			assert(false);
		vector3.x = x;
	}
	{
		std::istringstream i(element->FirstChild("Y")->ToElement()->GetText());
		if (!(i >> x))
			assert(false);
		vector3.y = x;
	}
	{
		std::istringstream i(element->FirstChild("Z")->ToElement()->GetText());
		if (!(i >> x))
			assert(false);
		vector3.z = x;
	}

	return vector3;
}

template int TinyUtil::GetElement<int>(TiXmlElement* element);
template unsigned int TinyUtil::GetElement<unsigned int>(TiXmlElement* element);
template float TinyUtil::GetElement<float>(TiXmlElement* element);
template bool TinyUtil::GetElement<bool>(TiXmlElement* element);
template Color TinyUtil::GetElement<Color>(TiXmlElement* element);
template Vector3 TinyUtil::GetElement<Vector3>(TiXmlElement* element);

// -----------------------------------------------------------------------------
// ------------------- std::string TinyUtil::ConvertToString -------------------
// -----------------------------------------------------------------------------
template < typename T >
std::string TinyUtil::ConvertToString(T _value)
{
	std::ostringstream o;
	if (!(o << _value))
		assert(false);
	
	return o.str();
}
