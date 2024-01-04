chmod +x ./iree-run-module

echo "cat.npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/cat.npy"


echo "cat0.npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/cat0.npy"


echo "cat1.npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/cat1.npy"


echo "cat2.npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/cat2.npy"


echo "confusing_dog.npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/confusing_dog.npy"

echo "dog0.npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/dog0.npy"


echo "human_with_cat.npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/human_with_cat.npy"


echo "leopard.npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/leopard.npy"

echo "lion,npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/lion.npy"

echo "scooby_doo.npy"
./iree-run-module --device=vulkan --function=custom_predict --module=./tf_cats_vs_dogs_vulkan.vmfb --vulkan_debug_verbosity=4 --input="1x224x224x3xf32=@numpy-images/scooby_doo.npy"