## Brief Description of files

### Data
In [this box folder](https://cornell.app.box.com/folder/311062381534).

### Audio Processing

_with v_env honda310_

audio_supress.py -- for dyadic/group interactions, supresses audio referring to speakers other than specific user (needs to be adjusted). Takes wav file and pyannote speaker diarization.

ast_feats.py -- extract features from audio to be used in [AST model](https://arxiv.org/pdf/2104.01778). Takes in audio directory and directory of labels and extracts features as a torch.stack (tensor). Matches feature frequency to dataset sample frequency. 
Features dimensions are 1024x128, which could be flattened for processing.

fine_tuning_ast.py -- trains AST in dataset, calculates features too

(folder) ast -- AST code repo.
**IMPORTANT**: need to modify ast--src--models--ast_models.py to include own directory (line 122 and beyond)

### Text Embeddings

_with v_env honda310_

get_text_embeddings.py - gets text embeddings from BERT (or other specified model) into csv file 

Arguments:
* 'vtt_folder', help='Path to the VTT folder')
* 'label_csv', help='Path to the label csv file')
* 'output_prefix', help='Prefix for output files')
* '--model', default='bert-base-uncased', 

 ` python3 get_text_embeddings.py ../../data/transcripts/ ../../data/all_participants.csv ../../data/text_embeddings/ `

 text embeddings are matched to the frequency of the labels. Using pca.py, the text_embeddings csv can be reduced to a smaller feature vector.

The pca embeddings can be treated like any other feature modality, and for single modality training we can use the 2 datasets (test PCA and original, dimension 768)

