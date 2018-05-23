/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beParameters.h"
#include "beCore/beParameterSet.h"

namespace beCore
{

// Constructor.
Parameters::Parameters()
{
}

// Constructor.
Parameters::Parameters(const Parameters &right)
	: m_parameters( right.m_parameters )
{
}

// Destructor.
Parameters::~Parameters()
{
}

// Adds a parameter of the given name, returning its parameter ID.
uint4 Parameters::Add(const utf8_ntri &name)
{
	uint4 parameterID = GetID(name);

	if (parameterID == InvalidID)
	{
		parameterID = static_cast<uint4>(m_parameters.size());
		m_parameters.push_back( Parameter(name) );
	}

	return parameterID;
}

// Gets the name of the parameter identified by the given ID.
utf8_ntr Parameters::GetName(uint4 parameterID) const
{
	return (parameterID < m_parameters.size())
		? utf8_ntr(m_parameters[parameterID].name)
		: utf8_ntr("");
}

// Gets the current number of parameters.
uint4 Parameters::GetCount() const
{
	return static_cast<uint4>(m_parameters.size());
}

// Gets the parameter identified by the given name.
uint4 Parameters::GetID(const utf8_ntri &name) const
{
	for (parameter_vector::const_iterator itParameter = m_parameters.begin();
		itParameter != m_parameters.end(); ++itParameter)
		if (itParameter->name == name)
			return static_cast<uint4>(itParameter - m_parameters.begin());

	return InvalidID;
}

// Assigns the given value to the parameter identified by the given ID.
void Parameters::SetAnyValue(uint4 parameterID, const lean::any *pValue)
{
	if (parameterID < m_parameters.size())
		// Clones the given value
		m_parameters[parameterID].value = pValue;
}

// Gets the value of the parameter identified by the given ID.
const lean::any* Parameters::GetAnyValue(uint4 parameterID) const
{
	return (parameterID < m_parameters.size())
		? m_parameters[parameterID].value.getptr()
		: nullptr;
}


// Constructor.
ParameterLayout::ParameterLayout()
{
}

// Constructor.
ParameterLayout::ParameterLayout(const ParameterLayout &right)
	: m_parameters( right.m_parameters )
{
}

// Destructor.
ParameterLayout::~ParameterLayout()
{
}

// Adds a parameter of the given name, returning its parameter ID.
uint4 ParameterLayout::Add(const utf8_ntri &name)
{
	uint4 parameterID = GetID(name);

	if (parameterID == InvalidID)
	{
		parameterID = static_cast<uint4>( m_parameters.size() );
		m_parameters.push_back( name.to<utf8_string>() );
	}

	return parameterID;
}

// Gets the name of the parameter identified by the given ID.
utf8_ntr ParameterLayout::GetName(uint4 parameterID) const
{
	return (parameterID < m_parameters.size())
		? utf8_ntr(m_parameters[parameterID])
		: utf8_ntr("");
}

// Gets the current number of parameters.
uint4 ParameterLayout::GetCount() const
{
	return static_cast<uint4>( m_parameters.size() );
}

// Gets the parameter identified by the given name.
uint4 ParameterLayout::GetID(const utf8_ntri &name) const
{
	for (parameter_vector::const_iterator itParameter = m_parameters.begin();
		itParameter != m_parameters.end(); ++itParameter)
		if (*itParameter == name)
			return static_cast<uint4>( itParameter - m_parameters.begin() );

	return InvalidID;
}


// Constructor.
ParameterSet::ParameterSet(const ParameterLayout *pLayout)
	: m_pLayout( LEAN_ASSERT_NOT_NULL(pLayout) )
{
}

// Constructor.
ParameterSet::ParameterSet(const ParameterSet &right)
	: m_pLayout( right.m_pLayout ),
	m_parameters( right.m_parameters )
{
}

// Destructor.
ParameterSet::~ParameterSet()
{
}

// Assigns the given value to the parameter identified by the given ID.
void ParameterSet::SetAnyValue(uint4 parameterID, const lean::any *pValue)
{
	const uint4 parameterCount = m_pLayout->GetCount();

	if (parameterID < parameterCount)
	{
		// Parameter layout may have changed
		if (m_parameters.size() < parameterCount)
			m_parameters.resize(parameterCount, nullptr);

		// Clones the given value
		m_parameters[parameterID] = pValue;
	}
}

// Gets the value of the parameter identified by the given ID.
const lean::any* ParameterSet::GetAnyValue(uint4 parameterID) const
{
	return (parameterID < m_parameters.size())
		? m_parameters[parameterID].getptr()
		: nullptr;
}

} // namespace
