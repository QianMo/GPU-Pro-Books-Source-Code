#pragma once
#include "RenderContext.h"
#include "ControlContext.h"

class Entity;

/// Scene graph node interface. To be implemented by scene graph branches (e.g. NodeGroup) and leaves (e.g. Entity).
class Node
{
protected:
	/// True if the part of the scene graph that belongs to the node has to be rendered.
	bool visible;
	/// True if the part of the scene graph that belongs to the node has to be animated.
	bool animated;
public:
	/// Constructor.
	Node();
	/// Virtual destructor.
	virtual ~Node(){};
	/// Renders scene graph from this node down. (Invokes render for every Entity.)
	virtual void render(const RenderContext& context)=0;
	/// Animates scene graph from this node down. (Invokes animate for every Entity.)
	virtual void animate(double dt)=0;
	/// Controls scene graph from this node down. (Invokes control for every Entity.)
	virtual void control(const ControlContext& context)=0;
	/// Performs interactions between the target entity and entities in the scene graph from this node down.
	virtual void interact(Entity* target)=0;

	/// Sets animation state.
	void setAnimated(bool a){animated = a;}
	/// Returns animation state.
	bool getAnimated() {return animated;}

	/// Sets visibility state.
	void setVisible(bool a){visible = a;}
	/// Returns visibility state.
	bool getVisible() {return visible;}
};
