# Install pyg

# echo "install pyg env"
# conda create -n pyg python=3.7
# conda install pyg -c pyg -c conda-forge
#pip install torch einops pytorch_lightning h5py matplotlib obspy pandas tensorflow

# ipython kernel install --user --name=pyg
# conda deactivate

# pip install torch-scatter
# pip install torch-sparse
# pip install torch-geometric

# pip install torch-cluster
# pip install torch-spline-conv

# Add project path
#export PYTHONPATH=$PYTHONPATH:/path/you/want/to/add
export PYTHONPATH=$PYTHONPATH:/home2/saeed/gMLP_phase


git clone --recurse-submodules https://github.com/seisbench/seisbench
# git clone --recurse-submodules https://github.com/SCEDC/pystp.git
# cd pystp
# pip install .
