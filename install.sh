tar -xvf run/static_data/encodings.tar.gz
mv encodings/ run/static_data/

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/julien-piet/cpp-hash.git
cd cpp-hash
python3 setup.py install
cd ..

pip install -r requirements.txt
pip install evaluate