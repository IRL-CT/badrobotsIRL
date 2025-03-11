import webvtt
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from datetime import datetime
import numpy as np

def parse_vtt_timestamp(timestamp):
    """Convert VTT timestamp to seconds"""
    time_obj = datetime.strptime(timestamp, '%H:%M:%S.%f')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond/1000000

class VTTEmbeddingProcessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
    
    def process_vtt_file(self, vtt_file_path):
        """
        Process a VTT file and return timestamps with BERT embeddings
        """
        # Read VTT file
        captions = webvtt.read(vtt_file_path)
        
        # Store results
        results = []
        
        # Process each caption
        for caption in captions:
            # Get start and end times in seconds
            start_time = parse_vtt_timestamp(caption.start)
            end_time = parse_vtt_timestamp(caption.end)

            print('caption text:', caption.text)
            #remove SPEAKER XX: from the text
            caption.text = caption.text.split(':')[-1]
            print('caption text:', caption.text)
            
            # Get BERT embedding for the text
            with torch.no_grad():
                inputs = self.tokenizer(caption.text, 
                                      return_tensors="pt",
                                      padding=True,
                                      truncation=True,
                                      max_length=512)
                outputs = self.model(**inputs)
                # Use [CLS] token embedding as sentence embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]

                print('shape embedding:', embedding.shape)
            
            results.append({
                'participant':vtt_file_path.split('/')[-1].split('.')[0],
                'start_time': start_time,
                'end_time': end_time,
                'text': caption.text,
                'embedding': embedding
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        return df
    
    def save_embeddings(self, df, label_path, output_path):
        """
        Save the embeddings and metadata to files
        """
        # Save embeddings as numpy array
        #embeddings = np.stack(df['embedding'].values)
        #np.save(f"{output_path}_embeddings.npy", embeddings)
        
        # Save metadata (timestamps and text) as CSV
        #metadata = df[['start_time', 'end_time', 'text']]
        #metadata.to_csv(f"{output_path}_metadata.csv", index=False)

        new_df = pd.DataFrame()
        features_df = pd.DataFrame()

        #open label file as dataframe
        label_df = pd.read_csv(label_path)
        embeds_shape = df['embedding'].values[0].shape
        #print(label_df['participant'].unique())
        #get timestamps, for each timestamp match with the df start_time and end_time
        for participant in label_df['participant'].unique():
            #get the label df for the participant
            label_df_participant = label_df[label_df['participant'] == participant]
            #get the df for the participant
            df_participant = df[df['participant'] == participant]
            timestamps = label_df_participant['frame'].values/30
            print('timestamps:', timestamps)
            print(label_df_participant.shape)
            print(df_participant.shape)
            print(df['participant'].unique())
            for ind, row in label_df_participant.iterrows():
                timestamp = row['frame']/30
                #get the df with the closest start_time and before the end_time
                df_filtered = df_participant[(df_participant['end_time'] >= timestamp) & (df_participant['start_time'] <= timestamp)]
                
                if len(df_filtered) == 0:
                    #print('no caption found for timestamp:', timestamp)
                    #continue
                    #add a series of zeros, in the same shape as the embeddings
                    #add each embedding as a column in fearues_df
                    features_df = pd.concat([features_df, pd.DataFrame(np.zeros(embeds_shape).reshape(1,-1))], ignore_index=True)
                    new_df = pd.concat([new_df, pd.DataFrame({'participant': row['participant'], 'frame': row['frame']}, index=[0])], ignore_index=True)
                else:
                    print(timestamp)
                    #print(df_participant['end_time'].values[-1])
                    print(df_filtered)
                    #get the first caption found
                    df_filtered = df_filtered.iloc[-1]
                    #add each embedding as a column in fearues_df
                    features_df = pd.concat([features_df, pd.DataFrame(df_filtered['embedding'].reshape(1,-1))], ignore_index=True)
                    #get timestamp and embedding as a row and append to new_df
                    new_df = pd.concat([new_df, pd.DataFrame({'participant': row['participant'], 'frame': row['frame']}, index=[0])], ignore_index=True)
                #check if we added a nan
               
                #if nan, print the timestamp
                if features_df.isnull().values.any():
                    print('nan values found')
                    print(timestamp)
                   
                    #break

        #add the embeddings as columns to the new_df
        new_df = pd.concat([new_df, features_df], axis=1)
        print(new_df.head())
        #check if it's got nan values
        print('checking for nan values')
        print(new_df.isnull().sum())
        print(new_df.shape)
        #remove rows with nan values
        new_df = new_df.dropna()
        print(new_df.shape)



        #save new_df as a csv file
        new_df.to_csv(f"{output_path}text_embeddings.csv", index=False)


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Process VTT file to BERT embeddings')
    parser.add_argument('vtt_folder', help='Path to the VTT folder')
    parser.add_argument('label_csv', help='Path to the label csv file')
    parser.add_argument('output_prefix', help='Prefix for output files')
    parser.add_argument('--model', default='bert-base-uncased', 
                        help='BERT model to use (default: bert-base-uncased)')
    
    args = parser.parse_args()

    big_df = pd.DataFrame()

    for vtt_file in os.listdir(args.vtt_folder):
        if vtt_file.endswith(".vtt"):
            print(f"Processing {vtt_file}")
            # Initialize processor
            processor = VTTEmbeddingProcessor(model_name=args.model)

            # Process VTT file
            df = processor.process_vtt_file(os.path.join(args.vtt_folder, vtt_file))
            print(df.head())

            big_df = pd.concat([big_df, df], ignore_index=True)

    # Save results
    processor.save_embeddings(big_df, args.label_csv, args.output_prefix)
    print(f"Processed {len(df)} captions")
    #print(f"Saved embeddings to {os.path.join(args.output_prefix, vtt_file[:-4])}_embeddings.npy")
    #print(f"Saved metadata to {os.path.join(args.output_prefix, vtt_file[:-4])}_metadata.csv")



if __name__ == "__main__":
    main()

    #default command: python3 get_text_embeddings.py ../../data/transcripts/ ../../data/all_participants.csv ../../data/text_embeddings/ 