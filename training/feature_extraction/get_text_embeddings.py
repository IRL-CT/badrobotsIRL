import webvtt
import torch
from transformers import CLIPTokenizer, CLIPTextModel
import pandas as pd
from datetime import datetime
import numpy as np

def parse_vtt_timestamp(timestamp):
    """Convert VTT timestamp to seconds"""
    time_obj = datetime.strptime(timestamp, '%H:%M:%S.%f')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond/1000000

class VTTEmbeddingProcessor:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()

    def process_vtt_file(self, vtt_file_path):
        """
        Process a VTT file and return timestamps with CLIP text embeddings
        """
        captions = webvtt.read(vtt_file_path)
        results = []

        for caption in captions:
            start_time = parse_vtt_timestamp(caption.start)
            end_time = parse_vtt_timestamp(caption.end)

            print('caption text:', caption.text)
            # Remove SPEAKER XX: prefix
            caption.text = caption.text.split(':')[-1]
            print('caption text:', caption.text)

            # Get CLIP text embedding
            with torch.no_grad():
                inputs = self.tokenizer(caption.text,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=77)  # CLIP max token length
                outputs = self.model(**inputs)
                # Use pooler_output (EOS token representation) as sentence embedding
                embedding = outputs.pooler_output[0].numpy()

                print('shape embedding:', embedding.shape)

            results.append({
                'participant': vtt_file_path.split('/')[-1].split('.')[0],
                'start_time': start_time,
                'end_time': end_time,
                'text': caption.text,
                'embedding': embedding
            })

        return pd.DataFrame(results)

    def save_embeddings(self, df, label_path, output_path, fps=100):
        """
        For each row in label_path CSV, find the matching VTT caption by timestamp
        and append its CLIP embedding.
        Output columns: first 4 columns of label CSV (frame, participant,
        binary_label, multiclass_label) followed by embedding dimensions.
        """
        label_df = pd.read_csv(label_path)
        # Columns 0-3: frame, participant, binary_label, multiclass_label
        meta_cols = label_df.columns[:4].tolist()

        embeds_shape = df['embedding'].values[0].shape
        embedding_rows = []
        meta_rows = []

        for participant in label_df['participant'].unique():
            label_df_part = label_df[label_df['participant'] == participant]
            df_part = df[df['participant'] == participant]
            print(f"participant: {participant} | label rows: {len(label_df_part)} | captions: {len(df_part)}")

            for _, row in label_df_part.iterrows():
                timestamp = row['frame'] / fps
                df_filtered = df_part[
                    (df_part['end_time'] >= timestamp) &
                    (df_part['start_time'] <= timestamp)
                ]

                if len(df_filtered) == 0:
                    embedding_rows.append(np.zeros(embeds_shape))
                else:
                    embedding_rows.append(df_filtered.iloc[-1]['embedding'])

                meta_rows.append(row[meta_cols].values)

        meta_df = pd.DataFrame(meta_rows, columns=meta_cols)
        embed_df = pd.DataFrame(np.stack(embedding_rows))
        result_df = pd.concat([meta_df, embed_df], axis=1)

        print(result_df.head())
        print('nan values:', result_df.isnull().sum().sum())
        print('shape:', result_df.shape)

        result_df = result_df.dropna()
        print('shape after dropna:', result_df.shape)

        result_df.to_csv(f"{output_path}text_embeddings.csv", index=False)


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Process VTT files to CLIP text embeddings')
    parser.add_argument('vtt_folder', help='Path to the VTT folder')
    parser.add_argument('output_prefix', help='Prefix for output files')
    parser.add_argument('--label_csv',
                        default='../../preprocessing/curated_features/allparticipants_100fps.csv',
                        help='Path to the label CSV (default: allparticipants_100fps.csv)')
    parser.add_argument('--model', default='openai/clip-vit-base-patch32',
                        help='CLIP model to use (default: openai/clip-vit-base-patch32)')
    parser.add_argument('--fps', type=int, default=100,
                        help='Frames per second of the label CSV (default: 100)')

    args = parser.parse_args()

    processor = VTTEmbeddingProcessor(model_name=args.model)
    big_df = pd.DataFrame()

    for vtt_file in os.listdir(args.vtt_folder):
        if vtt_file.endswith(".vtt"):
            print(f"Processing {vtt_file}")
            df = processor.process_vtt_file(os.path.join(args.vtt_folder, vtt_file))
            print(df.head())
            big_df = pd.concat([big_df, df], ignore_index=True)

    processor.save_embeddings(big_df, args.label_csv, args.output_prefix, fps=args.fps)
    print(f"Processed {len(big_df)} captions")


if __name__ == "__main__":
    main()

    # default command:
    # python3 get_text_embeddings.py ../../data/transcripts/ ../../data/text_embeddings/
