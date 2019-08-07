.PHONY: clean

clean:
	@echo "Removing .pyc files..."
	find . -name "*.pyc" -type f -delete
	@echo "Removing __pycache__ directories..."
	find . -name "*__pycache__" -type d -delete
	@echo "Removing extra files created by parser..."
	rm -f planning/plan.txt planning/problem.pddl planning/problem.txt
