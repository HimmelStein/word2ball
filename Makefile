init:
	pip3 install -r requirements.txt
	pip3 install .
test:
	nosetests word2ball/tests
