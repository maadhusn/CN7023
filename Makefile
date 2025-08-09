.PHONY: smoke train eval clean install lint format

# Default Python interpreter
PYTHON := python3

# Install dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

# Lint and format code
lint:
	$(PYTHON) -m ruff check src/
	$(PYTHON) -m black --check src/

format:
	$(PYTHON) -m ruff check --fix src/
	$(PYTHON) -m black src/

# Synthetic smoke test (no dataset required)
smoke: install
	@echo "Running synthetic end-to-end smoke test..."
	$(PYTHON) src/demo_synth.py
	@echo "Training CNN model on synthetic data..."
	$(PYTHON) src/train_cnn.py
	@echo "Evaluating CNN model with GradCAM..."
	$(PYTHON) src/eval_cnn.py --gradcam 2
	@echo "Training ANN model on handcrafted features..."
	$(PYTHON) src/train_ann.py
	@echo "Fusing CNN and ANN predictions..."
	$(PYTHON) src/fuse.py
	@echo "Smoke test completed successfully!"

# Train CNN model
train: install
	$(PYTHON) src/train_cnn.py

# Evaluate CNN model with GradCAM
eval: install
	$(PYTHON) src/eval_cnn.py
	$(PYTHON) src/gradcam.py

# Train ANN model
train-ann: install
	$(PYTHON) src/features.py
	$(PYTHON) src/train_ann.py

# Fusion of CNN and ANN
fuse: install
	$(PYTHON) src/fuse.py

# Clean generated files
clean:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/models/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete

# Help
help:
	@echo "Available targets:"
	@echo "  smoke     - Run synthetic end-to-end test (no dataset required)"
	@echo "  train     - Train CNN model"
	@echo "  eval      - Evaluate CNN model with GradCAM"
	@echo "  train-ann - Train ANN model with HOG features"
	@echo "  fuse      - Fusion of CNN and ANN predictions"
	@echo "  install   - Install dependencies"
	@echo "  lint      - Check code style"
	@echo "  format    - Format code"
	@echo "  clean     - Clean generated files"
	@echo "  help      - Show this help message"
