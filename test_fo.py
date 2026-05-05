# save this as test_fo.py and run it
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart-video")
session = fo.launch_app(dataset)

input("Press Enter to exit...")