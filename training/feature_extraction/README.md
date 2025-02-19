

audio_supress.py -- for dyadic/group interactions, supresses audio referring to speakers other than specific user (needs to be adjusted). Takes wav file and pyannote speaker diarization.

ast_feats.py -- extract features from audio to be used in [AST model](https://arxiv.org/pdf/2104.01778). Takes in audio directory and directory of labels and extracts features as a torch.stack (tensor). Matches feature frequency to dataset sample frequency. 
Features dimensions are 1024x128, which could be flattened for processing.

fine_tuning_ast.py -- trains AST in dataset, calculates features too

(folder) ast -- AST code repo.
IMPORTANT: need to modify ast--src--models--ast_models.py to include own directory (line 122 and beyond)



