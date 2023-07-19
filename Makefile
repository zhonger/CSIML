PHONY: init build

init:
	pip3 install -r requirements.txt
build:
	poetry build
install:
	pip3 install dist/*.whl
uninstall:
	pip3 uninstall pictl -y
clean:
	rm -rf dist && py3clean .
