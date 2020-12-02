## What is missing
What is missing is the dataset. It has to be present in order to run the code.

## How to run it
This code runs better on a GPU.
In order to use a specific GPU (for example, with id 0) to run the code, use:

	CUDA_VISIBLE_DEVICES=0 python main.py

if no GPU is available, the code can run on CPU too:
			
	python main.py
