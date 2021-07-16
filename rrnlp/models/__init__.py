import os 
import rrnlp 

weights_path = os.path.join(os.path.dirname(rrnlp.__file__), "models", "weights")
device = "cpu"

files_needed = ['RoB_encoder_custom.pt',
 				'evidence_identification_encoder_custom.pt',
				'bert_ptyp_LR.pck',
				 'evidence_identification_clf.pt',
				 'RCT_overall_abs_clf.pt',
				 'RCT_encoder_custom.pt',
				 'inference_clf.pt',
				 'interventions_encoder_custom.pt',
				 'outcomes_encoder_custom.pt',
				 'outcomes_clf.pt',
				 'bert_LR.pck',
				 'sample_size_weights.pt',
				 'interventions_clf.pt',
				 'inference_encoder_custom.pt',
				 'RoB_overall_abs_clf.pt',
				 'population_encoder_custom.pt',
				 'population_clf.pt',
				 'bias_prob_clf.pck']

weights_path = rrnlp.models.weights_path

for f in files_needed:
	url = "https://zenodo.org/record/5110032/files/" + f
	f_path = os.path.join(weights_path, f)

	if not os.path.exists(f_path):
	    import urllib     
	    print(f"Attempting to fetch weights from {url}")
	    # TODO this is slow so should probably add a progress bar;
	    # at present it just kinda sits there for a long time.
	    
	    urllib.request.urlretrieve(url, f_path, reporthook)
	    if os.path.exists(f_path):
	        print("success!")
	    else:
	        raise Exception("Sorry; unable to download static word vectors, and will not be able to use this model.")
