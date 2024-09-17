env-cpu-create:
	mamba env create -f environment-cpu-from_history.yml
	# pip install keras-core
	# ToDo move keras-core/keras to mamba when/if available on conda-forge

env-cpu-create-from-lock:
	mamba env create -f environment-cpu.yml

env-cpu-export:
	mamba env export > environment-cpu.yml
	mamba env export --from-history > environment-cpu-from_history.yml

env-gpu-create:
	mamba env create -f environment-Gpu-from_history.yml
    # See https://keras.io/keras_core/#installation and https://colab.research.google.com/drive/13cpd3wCwEHpsmypY9o6XB6rXgBm5oSxu?usp=sharing
    # pip install --upgrade -I keras-core "jax[cuda12_pip]==0.4.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
	# ToDo move keras-core/keras/jax to mamba when/if available on conda-forge

env-gpu-create-from-lock:
	mamba env create -f environment-gpu.yml

env-gpu-export:
	mamba env export > environment-gpu.yml
	mamba env export --from-history > environment-gpu-from_history.yml

env-update:
	mamba update --all