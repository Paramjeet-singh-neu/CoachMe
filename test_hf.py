from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub("Voxel51/Safe_and_Unsafe_Behaviours", max_samples=5)
print(dataset)
print(dataset.first())


dataset = load_from_hub("Voxel51/qualcomm-exercise-video-dataset-benchmark", max_samples=5)
print(dataset)