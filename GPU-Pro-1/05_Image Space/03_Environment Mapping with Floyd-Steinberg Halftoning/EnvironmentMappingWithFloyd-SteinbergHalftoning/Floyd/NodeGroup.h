#pragma once
#include <vector>
#include "Node.h"

/// Scene graph branching node.
class NodeGroup : public Node
{
	/// Dynamic container for references to children nodes in the scene graph.
	std::vector<Node*> subnodes;
public:
	/// Destructor. Releases contained subnodes.
	~NodeGroup(void);

	/// Inserts a subnode.
	void add(Node* e);

	/// Invokes render for all subnodes.
	virtual void render(const RenderContext& context);
	/// Invokes animate for all subnodes.
	virtual void animate(double dt);
	/// Invokes control for all subnodes.
	virtual void control(const ControlContext& context);
	/// Invokes interact for all subnodes.
	virtual void interact(Entity* target);
};
