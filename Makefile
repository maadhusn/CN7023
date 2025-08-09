.PHONY: train visualize clean

train:
	python train.py

visualize:
	python visualize_results.py

clean:
	rm -rf results/
