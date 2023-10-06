env-cpu-create:
	mamba env create -f environment-cpu-from_history.yml
	# pip install keras-core
	# ToDo move keras-core/keras to mamba when/if available on conda-forge

env-cpu-create-from-lock:
	mamba env create -f environment-cpu.yml

env-cpu-export:
	mamba env export > environment-cpu.yml
	mamba env export --from-history > environment-cpu-from_history.yml

env-update:
	mamba update --all