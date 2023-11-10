SRC='src'

view:
	python3 ${SRC}/gan_mnist_cnn_view_data.py $(record)

train:
	python3 ${SRC}/gan_mnist_cnn_train.py

test:
	python3 ${SRC}/gan_mnist_cnn_test.py
