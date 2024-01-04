tf-mlir-translate --savedmodel-objectgraph-to-mlir --tf-savedmodel-exported-names=custom_predict ./saved-model-signed -o ./mlir-artifacts/starting_dialect.mlir

tf-opt --tf-executor-graph-pruning --tf-executor-to-functional-conversion --canonicalize --tf-lower-to-mlprogram-and-hlo ./mlir-artifacts/starting_dialect.mlir -o ./mlir-artifacts/mlprog_hlo.mlir

iree-opt --iree-stablehlo-to-stablehlo-preprocessing --iree-stablehlo-canonicalize --iree-stablehlo-to-linalg ./mlir-artifacts/mlprog_hlo.mlir -o ./mlir-artifacts/final.mlir

iree-compile --iree-hal-target-backends=vulkan-spirv ./mlir-artifacts/final.mlir -o ./executables/tf_cats_vs_dogs_vulkan.vmfb

cp ./iree-run-module ./executables
cp ./android-runner.sh ./executables