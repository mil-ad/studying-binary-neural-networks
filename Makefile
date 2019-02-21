.PHONY: help
help:
	@echo "Make targets are:"
	@echo "  make help - shows this message"
	@echo "  make download-cifar10 - Downloads pylearn2 GCA Whitened CIFAR10 dataset."
	@echo "  make clean-last-run - removes last run's intermediate files."
	@echo "  make clean - removes intermediate files."

.PHONY: download-cifar10
download-cifar10:
	@echo Downloading ...
	@(curl -o archive.zip -L 'https://www.dropbox.com/sh/ygbx8ckcjwk0jlj/AAAtNQojrcx0A5Fc8FLZ3iTia?dl=1')
	@echo Exracting ...
	@(unzip archive.zip -d datasets/cifar_10/pylearn2_gca_whitened -x / && rm archive.zip)
	@echo Checking the MD5 sums ...
	@(cd datasets/cifar_10 && md5sum --check pylearn2_gca_whitened.md5 && cd - > /dev/null)

.PHONY: create-conda-env
create-conda-env:
	@(conda env create --file environment.yml)

.PHONY: clean-last-run
clean-last-run:
	@(ls -d -t -1 ./_sessions/** | head -n 1 | xargs rm -rf)
	@(rm ./_sessions/latest)

.PHONY: clean
clean:
	@(rm -rf _sessions)
	@(rm -rf _MNIST_DATA)
	@(find . -type d -name __pycache__ | xargs rm -rf)
	@(find . -type d -name .ipynb_checkpoints | xargs rm -rf)
	@(rm -f tmux*)
