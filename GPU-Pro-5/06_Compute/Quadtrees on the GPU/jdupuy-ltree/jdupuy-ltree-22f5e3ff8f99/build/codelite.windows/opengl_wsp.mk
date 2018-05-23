.PHONY: clean All

All:
	@echo "----------Building project:[ quadtree_simple - debug ]----------"
	@$(MAKE) -f  "quadtree_simple.mk"
clean:
	@echo "----------Cleaning project:[ quadtree_simple - debug ]----------"
	@$(MAKE) -f  "quadtree_simple.mk" clean
