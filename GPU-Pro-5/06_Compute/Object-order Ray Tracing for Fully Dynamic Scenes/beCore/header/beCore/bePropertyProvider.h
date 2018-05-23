/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_PROPERTY_PROVIDER
#define BE_CORE_PROPERTY_PROVIDER

#include "beCore.h"
#include "beComponent.h"
#include "beValueType.h"
#include "beExchangeContainers.h"
#include <lean/properties/property_type.h>

#include <lean/smart/scoped_ptr.h>

namespace beCore
{

/// Widget enumeration.
struct Widget
{
	/// Enumeration.
	enum T
	{
		None,			///< No widget.
		Raw,			///< Raw value widget.
		Slider,			///< Slider widget.
		Color,			///< Color widget.
		Angle,			///< Angle widget.
		Orientation,	///< Orientation widget.

		End
	};
	LEAN_MAKE_ENUM_STRUCT(Widget)
};

struct ValueTypeDesc;

/// Property description.
struct PropertyDesc
{
	const ValueTypeDesc *TypeDesc;	///< Value (component) type.
	uint4 Count;					///< Value (component) count.
	int2 Widget;					///< UI widget.

	/// Default Constructor.
	PropertyDesc()
		: TypeDesc(nullptr),
		Count(0),
		Widget(Widget::None) { }
	/// Constructor. Truncates type ID to 2 bytes.
	PropertyDesc(const ValueTypeDesc &typeDesc, uint4 count, int2 widget)
		: TypeDesc(&typeDesc),
		Count(count),
		Widget(widget) { }
};

LEAN_INLINE bool operator ==(const PropertyDesc &desc1, const PropertyDesc &desc2)
{
	return desc1.TypeDesc == desc2.TypeDesc && desc1.Count == desc2.Count && desc1.Widget == desc2.Widget;
}

LEAN_INLINE bool operator !=(const PropertyDesc &desc1, const PropertyDesc &desc2)
{
	return !(desc1 == desc2);
}

class PropertyVisitor;
class ComponentObserver;
class ReflectedComponent;

struct PropertyVisitFlags
{
	enum T
	{
		None = 0x0,				///< Default access.
		PartialWrite = 0x1,		///< Not all data overwritten, keep untouched data.

		PersistentOnly = 0x10	///< Fail for non-persistent properties.
	};
};

/// Generic property provider base class.
class LEAN_INTERFACE PropertyProvider : public Component
{
	LEAN_INTERFACE_BEHAVIOR(PropertyProvider)

public:
	/// Invalid property ID.
	static const uint4 InvalidID = static_cast<uint4>(-1);

	/// Gets the number of properties.
	virtual uint4 GetPropertyCount() const = 0;
	/// Gets the ID of the given property.
	virtual uint4 GetPropertyID(const utf8_ntri &name) const = 0;
	/// Gets the name of the given property.
	virtual utf8_ntr GetPropertyName(uint4 id) const = 0;
	/// Gets the type of the given property.
	virtual PropertyDesc GetPropertyDesc(uint4 id) const = 0;

	/// Sets the given (raw) values.
	virtual bool SetProperty(uint4 id, const std::type_info &type, const void *values, size_t count) = 0;
	/// Gets the given number of (raw) values.
	virtual bool GetProperty(uint4 id, const std::type_info &type, void *values, size_t count) const = 0;

	/// Visits a property for modification.
	virtual bool WriteProperty(uint4 id, PropertyVisitor &visitor, uint4 flags = PropertyVisitFlags::None) = 0;
	/// Visits a property for reading.
	virtual bool ReadProperty(uint4 id, PropertyVisitor &visitor, uint4 flags = PropertyVisitFlags::None) const = 0;

	/// Sets the given value.
	template <class Value>
	LEAN_INLINE bool SetProperty(uint4 id, const Value &value) { return SetProperty(id, typeid(Value), lean::addressof(value), 1); }
	/// Sets the given values.
	template <class Value>
	LEAN_INLINE bool SetProperty(uint4 id, const Value *values, size_t count) { return SetProperty(id, typeid(Value), values, count); }
	/// Gets a value.
	template <class Value>
	LEAN_INLINE bool GetProperty(uint4 id, Value &value) const { return GetProperty(id, typeid(Value), lean::addressof(value), 1); }
	/// Gets the given number of values.
	template <class Value>
	LEAN_INLINE bool GetProperty(uint4 id, Value *values, size_t count) const { return GetProperty(id, typeid(Value), values, count); }

	/// Adds a property listener.
	virtual void AddObserver(ComponentObserver *listener) = 0;
	/// Removes a property listener.
	virtual void RemoveObserver(ComponentObserver *pListener) = 0;

	/// Hints at externally imposed changes, such as changes via an editor UI.
	BE_CORE_API virtual void ForcedChangeHint() { }
};

/// Simple property listener callback implementation.
class ComponentObserverCollection
{
public:
	struct ListenerEntry;
	typedef lean::scoped_ptr<ListenerEntry> listeners_t;

private:
	listeners_t m_listeners;

public:
	/// Constructor.
	BE_CORE_API ComponentObserverCollection();
	// Copies the given collection.
	BE_CORE_API ComponentObserverCollection(const ComponentObserverCollection &right);
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Takes over the given collection.
	BE_CORE_API ComponentObserverCollection(ComponentObserverCollection &&right);
#endif
	/// Destructor.
	BE_CORE_API ~ComponentObserverCollection();

	/// Copies the given collection.
	ComponentObserverCollection& operator =(ComponentObserverCollection right)
	{
		swap(right);
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Copies the given collection.
	LEAN_INLINE ComponentObserverCollection& operator =(ComponentObserverCollection &&right)
	{
		swap(right);
		return *this;
	}
#endif

	/// Adds a property listener.
	BE_CORE_API void AddObserver(ComponentObserver *listener);
	/// Removes a property listener.
	BE_CORE_API void RemoveObserver(ComponentObserver *pListener);
	/// Calls all listeners.
	BE_CORE_API void EmitPropertyChanged(const PropertyProvider &provider) const;
	/// Calls all listeners.
	BE_CORE_API void EmitChildChanged(const ReflectedComponent &provider) const;
	/// Calls all listeners.
	BE_CORE_API void EmitStructureChanged(const Component &provider) const;
	/// Calls all listeners.
	BE_CORE_API void EmitComponentReplaced(const Component &previous) const;

	/// Checks if any listeners have been registered.
	LEAN_INLINE bool HasObservers() const { return m_listeners != nullptr; }

	/// Checks if any property listeners have been registered before making the call.
	LEAN_INLINE void RarelyEmitPropertyChanged(const PropertyProvider &provider) const
	{
		if (m_listeners != nullptr)
			EmitPropertyChanged(provider);
	}
	/// Checks if any property listeners have been registered before making the call.
	LEAN_INLINE void RarelyEmitChildChanged(const ReflectedComponent &provider) const
	{
		if (m_listeners != nullptr)
			EmitChildChanged(provider);
	}
	/// Checks if any property listeners have been registered before making the call.
	LEAN_INLINE void RarelyEmitStructureChanged(const Component &provider) const
	{
		if (m_listeners != nullptr)
			EmitStructureChanged(provider);
	}
	/// Checks if any property listeners have been registered before making the call.
	LEAN_INLINE void RarelyEmitComponentReplaced(const Component &provider) const
	{
		if (m_listeners != nullptr)
			EmitComponentReplaced(provider);
	}

	/// Swaps the listeners of this collection with those of the given collection.
	LEAN_INLINE void swap(ComponentObserverCollection &right)
	{
		m_listeners.swap(right.m_listeners);
	}
};

/// Swaps the listeners of the given collections.
LEAN_INLINE void swap(ComponentObserverCollection &left, ComponentObserverCollection &right)
{
	left.swap(right);
}

/// Simple property listener callback implementation.
template <class Interface = PropertyProvider>
class LEAN_INTERFACE PropertyFeedbackProvider : public Interface
{
	LEAN_BASE_BEHAVIOR(PropertyFeedbackProvider)

private:
	ComponentObserverCollection m_listenerCollection;

protected:
	LEAN_BASE_DELEGATE(PropertyFeedbackProvider, Interface)

public:
	/// Adds a property listener.
	void AddObserver(ComponentObserver *listener) { m_listenerCollection.AddObserver(listener); }
	/// Removes a property listener.
	void RemoveObserver(ComponentObserver *pListener) { m_listenerCollection.RemoveObserver(pListener); }
	
	/// Checks, if any property listeners have been registered, before making the call.
	LEAN_INLINE void EmitPropertyChanged() const { m_listenerCollection.RarelyEmitPropertyChanged(*this); }
	/// Checks, if any property listeners have been registered, before making the call.
	LEAN_INLINE void EmitChildChanged() const { m_listenerCollection.RarelyEmitChildChanged(*this); }
	/// Checks, if any property listeners have been registered, before making the call.
	LEAN_INLINE void EmitStructureChanged() const { m_listenerCollection.RarelyEmitStructureChanged(*this); }
	/// Checks, if any property listeners have been registered, before making the call.
	LEAN_INLINE void EmitComponentReplaced() const { m_listenerCollection.RarelyEmitComponentReplaced(*this); }
};

/// No listener support implementation.
template <class Interface = PropertyProvider>
class LEAN_INTERFACE NoPropertyFeedbackProvider : public Interface
{
	LEAN_BASE_BEHAVIOR(NoPropertyFeedbackProvider)

protected:
	LEAN_BASE_DELEGATE(NoPropertyFeedbackProvider, Interface)
	
public:
	/// Does nothing.
	void AddObserver(ComponentObserver *listener) { }
	/// Does nothing.
	void RemoveObserver(ComponentObserver *pListener) { }
};

/// Generic property provider base class.
template <class Interface = PropertyProvider>
class LEAN_INTERFACE OptionalPropertyProvider : public Interface
{
	LEAN_BASE_BEHAVIOR(OptionalPropertyProvider)

protected:
	LEAN_BASE_DELEGATE(OptionalPropertyProvider, Interface)

public:
	/// Gets the number of properties.
	virtual uint4 GetPropertyCount() const { return 0; }
	/// Gets the ID of the given property.
	virtual uint4 GetPropertyID(const utf8_ntri &name) const { return InvalidID; }
	/// Gets the name of the given property.
	virtual utf8_ntr GetPropertyName(uint4 id) const { return utf8_ntr(""); }
	/// Gets the type of the given property.
	virtual PropertyDesc GetPropertyDesc(uint4 id) const { return PropertyDesc(); }

	/// Sets the given (raw) values.
	virtual bool SetProperty(uint4 id, const std::type_info &type, const void *values, size_t count) { return false; }
	/// Gets the given number of (raw) values.
	virtual bool GetProperty(uint4 id, const std::type_info &type, void *values, size_t count) const { return false; }

	/// Visits a property for modification.
	virtual bool WriteProperty(uint4 id, PropertyVisitor &visitor, uint4 flags = PropertyVisitFlags::None) { return false; }
	/// Visits a property for reading.
	virtual bool ReadProperty(uint4 id, PropertyVisitor &visitor, uint4 flags = PropertyVisitFlags::None) const { return false; }
};

/// Enhanced generic property provider base class.
class LEAN_INTERFACE EnhancedPropertyProvider : public PropertyProvider
{
	LEAN_INTERFACE_BEHAVIOR(EnhancedPropertyProvider)

public:
	/// Resets the given property to its default value.
	virtual bool ResetProperty(size_t id) = 0;

	/// Gets a default value provider.
	virtual const PropertyProvider* GetPropertyDefaults() const = 0;
	/// Gets a range minimum provider.
	virtual const PropertyProvider* GetLowerPropertyLimits() const = 0;
	/// Gets a range maximum provider.
	virtual const PropertyProvider* GetUpperPropertyLimits() const = 0;
};

/// Transfers all from the given source property provider to the given destination property provider.
BE_CORE_API void TransferProperties(PropertyProvider &dest, const PropertyProvider &source);

} // namespace

#endif