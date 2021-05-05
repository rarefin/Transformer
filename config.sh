
sudo apt-get update && sudo apt-get install -y \
	vim screen htop wget gcc libsndfile1 g++
sudo apt-get install -y git-core bash-completion

# Python packages
pip install tqdm jupyter jupyterlab matplotlib 
pip install numpy Cython
pip install -r requirements.txt