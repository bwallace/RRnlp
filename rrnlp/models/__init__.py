import os 
import rrnlp 
import json
import time
import sys

weights_path = os.path.join(os.path.dirname(rrnlp.__file__), "models", "weights")
device = "cpu"


with open(os.path.join(weights_path, "weights_manifest.json"), 'r') as f:
	files_needed = json.load(f)



#####
# On import, check if word vectors are where we expect; if not, fetch them.
####
def reporthook(count, block_size, total_size):
    # shamelessly stolen: https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

for model_name, model_data in files_needed.items():

	for f in model_data['files']:

		url = f"https://zenodo.org/record/{model_data['zenodo']}/files/" + f

		f_path = os.path.join(weights_path, f"{model_data['zenodo']}_{f}")

		if not os.path.exists(f_path):
		    import urllib.request     
		    print(f"Attempting to fetch weights from Zenodo {url}")
		    # TODO this is slow so should probably add a progress bar;
		    # at present it just kinda sits there for a long time.
		    
		    urllib.request.urlretrieve(url, f_path, reporthook)
		    if os.path.exists(f_path):
		        print("success!")
		    else:
		        raise Exception(f"Sorry; unable to download data needed for the {model_name} model ({f}) - you will be unable to use this model.")
