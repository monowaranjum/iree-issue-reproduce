set -e
rm -rf ./saved-model-unsigned
rm -rf ./saved-model-signed

mkdir ./saved-model-unsigned
mkdir ./saved-model-signed

python3 fine-tune.py $1
python3 signer.py

rm -rf ./numpy-images
mkdir ./numpy-images

python3 input_prep.py

python3 ground-truth-generator.py

rm -rf ./mlir-artifacts
rm -rf ./executables

mkdir ./mlir-artifacts
mkdir ./executables

cp -r ./numpy-images ./executables/
cp ./iree-run-module ./executables