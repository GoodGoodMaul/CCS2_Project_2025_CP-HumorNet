import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import copy
from tqdm import tqdm
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# --- File Path Configuration (Please ensure these paths are correct in your environment) ---
# Google Drive mount path (example)
DRIVE_MOUNT_PATH = "/content/gdrive/MyDrive/"
# Base path where feature files are located (example)
BASE_PROJECT_PATH = os.path.join(DRIVE_MOUNT_PATH, "Project_CCS2-main/sdk_features/")

# Path to the dataset split file
data_folds_path = os.path.join(BASE_PROJECT_PATH, "data_folds.pkl")
# Path to the OpenFace feature file
openface_file = os.path.join(BASE_PROJECT_PATH, "openface_features_sdk.pkl")
# Path to the COVAREP feature file
covarep_file = os.path.join(BASE_PROJECT_PATH, "covarep_features_sdk.pkl")
# Path to the language feature file
language_file = os.path.join(BASE_PROJECT_PATH, "language_sdk.pkl")
# Path to the humor label file
humor_label_file = os.path.join(BASE_PROJECT_PATH, "humor_label_sdk.pkl")

# Audio word-level feature dimension constant
_AUDIO_WORD_DIM_CONST = 81
# Video word-level feature dimension constant
_VIDEO_WORD_DIM_CONST = 371
# Hidden dimension of sentence-level LSTM in Hierarchical LSTM (Modified to align with Script_B's configuration idea)
SENTENCE_LSTM_HIDDEN_DIM_CONFIG = 256
# Hidden dimension of sample-level LSTM in Hierarchical LSTM (also its output dimension, projector layer input dimension) (Modified to align with Script_B)
SAMPLE_LSTM_HIDDEN_DIM_CONFIG = 512


# Helper function to load pickle files
def load_pickle(pickle_file):
    try:
        # Open file in binary read mode
        with open(pickle_file, 'rb') as f:
            # Load pickle data
            return pickle.load(f)
    # Handle possible UnicodeDecodeError
    except UnicodeDecodeError:
        # If UnicodeDecodeError occurs, try opening with latin1 encoding
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    # Handle other possible exceptions
    except Exception as e:
        print(f'Cannot load data {pickle_file}: {e}')
        # Raise exception
        raise

# Helper function to safely prepare feature data for np.array()
def _prepare_feature_for_numpy(feature_data):
    # If input data is None, return an empty list
    if feature_data is None: return []
    # If input data is a numpy array
    if isinstance(feature_data, np.ndarray):
        # If it's an empty numpy array, return an empty list
        if feature_data.size == 0: return []
        # Return non-empty numpy array
        return feature_data
    # If input data is a list
    if isinstance(feature_data, list):
        # If it's an empty list, return an empty list
        if not feature_data: return []
        # Return non-empty list
        return feature_data
    # Other unexpected types, return an empty list (can add a warning)
    return []

# Function to extract features and labels
def extract_features_and_labels(id_list, language_sdk, covarep_sdk, openface_sdk, humor_label_sdk):
    # Initialize lists to store various features and labels
    ps_list, cs_list, cvp_p_list, cvp_c_list, of_p_list, of_c_list = [], [], [], [], [], []
    labels_list = []
    # Iterate through the ID list
    for hid in id_list:
        # Add punchline text
        ps_list.append(language_sdk[hid]['punchline_sentence'])
        # Add context text list
        cs_list.append(language_sdk[hid]['context_sentences'])

        # COVAREP (audio) feature processing
        # Prepare COVAREP features for the punchline
        prepared_punchline_cvp = _prepare_feature_for_numpy(covarep_sdk[hid]['punchline_features'])
        # Convert the prepared punchline audio features to a float32 numpy array and add
        cvp_p_list.append(np.array(prepared_punchline_cvp, dtype=np.float32))
        # Process context COVAREP features (one feature array per sentence)
        processed_sents_cvp = []
        for sent_feat in covarep_sdk[hid]['context_features']:
            prepared_sent_cvp = _prepare_feature_for_numpy(sent_feat)
            processed_sents_cvp.append(np.array(prepared_sent_cvp, dtype=np.float32))
        # Add the list of processed context audio features
        cvp_c_list.append(processed_sents_cvp)

        # OpenFace (video) feature processing
        # Prepare OpenFace features for the punchline
        prepared_punchline_of = _prepare_feature_for_numpy(openface_sdk[hid]['punchline_features'])
        # Convert the prepared punchline video features to a float32 numpy array and add
        of_p_list.append(np.array(prepared_punchline_of, dtype=np.float32))
        # Process context OpenFace features
        processed_sents_of = []
        for sent_feat in openface_sdk[hid]['context_features']:
            prepared_sent_of = _prepare_feature_for_numpy(sent_feat)
            processed_sents_of.append(np.array(prepared_sent_of, dtype=np.float32))
        # Add the list of processed context video features
        of_c_list.append(processed_sents_of)

        # Add labels
        labels_list.append(humor_label_sdk[hid])

    # Return all extracted features and labels, specifying the dtype for numpy arrays
    return (
        np.array(ps_list, dtype=object), np.array(cs_list, dtype=object),
        np.array(cvp_p_list, dtype=object), np.array(cvp_c_list, dtype=object),
        np.array(of_p_list, dtype=object), np.array(of_c_list, dtype=object),
        np.array(labels_list, dtype=np.float32)
    )

# Prepare data for the new dataset structure: output a list of samples, each sample is a dictionary containing all sentence features/texts
# Among them, the features/text of the punchline will be the last item in the corresponding modality list
def concatenate_multimodal_data_for_dataset(cvp_c, of_c, cs, cvp_p, of_p, ps):
    # Get the number of samples (based on the number of context sentences)
    num_samples = len(cs)
    # List to store all sample data
    all_samples_data = []
    # Iterate through each sample
    for i in range(num_samples):
        # Data dictionary for a single sample, containing 'audio', 'video', 'text' keys
        sample_data = {'audio': [], 'video': [], 'text': []}

        # Audio data processing
        # Extract context audio features, ensuring they are valid numpy arrays (word count > 0, correct dimension)
        current_sample_audio = [s for s in list(cvp_c[i]) if isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[0] > 0 and s.shape[1] == _AUDIO_WORD_DIM_CONST]
        # Get punchline audio features
        punchline_audio = cvp_p[i]
        # Append punchline audio features to the end of the list, if valid
        if isinstance(punchline_audio, np.ndarray) and punchline_audio.ndim == 2 and punchline_audio.shape[0] > 0 and punchline_audio.shape[1] == _AUDIO_WORD_DIM_CONST:
            current_sample_audio.append(punchline_audio)
        # If the current audio list is empty (both context and punchline are invalid or missing), add a placeholder for the punchline (single sample, correct dimension)
        elif not current_sample_audio:
            current_sample_audio.append(np.zeros((1, _AUDIO_WORD_DIM_CONST), dtype=np.float32))
        # Store the processed audio feature list into the sample data dictionary
        sample_data['audio'] = current_sample_audio

        # Video data processing (logic same as audio)
        current_sample_video = [s for s in list(of_c[i]) if isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[0] > 0 and s.shape[1] == _VIDEO_WORD_DIM_CONST]
        punchline_video = of_p[i]
        if isinstance(punchline_video, np.ndarray) and punchline_video.ndim == 2 and punchline_video.shape[0] > 0 and punchline_video.shape[1] == _VIDEO_WORD_DIM_CONST:
            current_sample_video.append(punchline_video)
        elif not current_sample_video:
            current_sample_video.append(np.zeros((1, _VIDEO_WORD_DIM_CONST), dtype=np.float32))
        sample_data['video'] = current_sample_video

        # Text data processing
        # Extract context sentence text list
        current_sample_text = [s for s in list(cs[i]) if isinstance(s, str)]
        # Get punchline text
        punchline_text_str = ps[i]
        # If the punchline text is a string, append it
        if isinstance(punchline_text_str, str):
            current_sample_text.append(punchline_text_str)
        # If the current text list is empty (both context and punchline are invalid or missing), add an empty string as a punchline placeholder
        elif not current_sample_text:
            current_sample_text.append("")
        sample_data['text'] = current_sample_text

        # Add the current sample's data dictionary to the total list
        all_samples_data.append(sample_data)
    # Return the list containing all sample data
    return all_samples_data


# --- Dataset Class: Modified for Context/Punchline Splitting ---
class CustomFeatureDatasetContextPunchline(Dataset):
    # Initialization function
    def __init__(self, list_of_sample_data_dicts, list_of_labels,
                 bert_tokenizer, max_bert_len_for_part=512,
                 audio_word_dim=_AUDIO_WORD_DIM_CONST, video_word_dim=_VIDEO_WORD_DIM_CONST):

        # List of sample data dictionaries (each element is a sample, containing 'audio', 'video', 'text' keys)
        self.list_of_sample_data_dicts = list_of_sample_data_dicts
        # List of labels, converted to torch.long type
        self.list_of_labels = torch.tensor(list_of_labels, dtype=torch.long)
        # BERT tokenizer
        self.tokenizer = bert_tokenizer
        # Maximum BERT length for each part (context/punchline)
        self.max_bert_len_for_part = max_bert_len_for_part
        # Audio word feature dimension
        self.audio_word_dim = audio_word_dim
        # Video word feature dimension
        self.video_word_dim = video_word_dim

    # Return the length of the dataset
    def __len__(self):
        return len(self.list_of_labels)

    # Helper function to tokenize the text part
    def _tokenize_text_part(self, text_sentences_list):
        # If the text list is empty
        if not text_sentences_list:
            # If the tokenizer has a pad token, use it, otherwise an empty string might be tokenized into special tokens
            processed_text = self.tokenizer.pad_token if self.tokenizer.pad_token is not None else ""
        else:
            # Join all sentences in the sentence list with spaces
            processed_text = " ".join(text_sentences_list)
            # If it's only whitespace or empty after joining
            if not processed_text.strip():
                processed_text = self.tokenizer.pad_token if self.tokenizer.pad_token is not None else ""

        # Call the tokenizer to tokenize
        bert_inputs = self.tokenizer(
            processed_text, add_special_tokens=True, return_attention_mask=True, # Add special tokens, return attention_mask
            max_length=self.max_bert_len_for_part, padding='max_length', truncation=True, # Max length, pad to max length, truncate
            return_tensors="pt", # Return PyTorch tensors
        )
        # Return input_ids and attention_mask, and remove the batch dimension (because this is single sample processing)
        return bert_inputs["input_ids"].squeeze(0), bert_inputs["attention_mask"].squeeze(0)

    # Helper function to process audio/video parts
    # all_sentences_features_for_sample: List of all sentence features for the entire sample (list of numpy arrays)
    # part_sentences_indices: Indices in the total sentence list that the current part (context or punchline) should contain
    # word_dim: Word feature dimension for audio or video
    def _process_av_part(self, all_sentences_features_for_sample, part_sentences_indices, word_dim):
        # List to store feature tensors of all sentences in this part
        part_features_list = []
        # If the sample itself does not have any sentence features (e.g., the entire sample is empty)
        if not all_sentences_features_for_sample:
            # Add a placeholder tensor (1 word, specified dimension)
            part_features_list.append(torch.zeros((1, word_dim), dtype=torch.float32))
            return part_features_list

        # Iterate through the sentence indices of the specified part
        for sent_idx in part_sentences_indices:
            # Ensure the index is within the valid range
            if 0 <= sent_idx < len(all_sentences_features_for_sample):
                # Get features of a single sentence (numpy array)
                sent_feat = all_sentences_features_for_sample[sent_idx]
                # Validate feature validity: is a numpy array, 2D, word count > 0, correct dimension
                if isinstance(sent_feat, np.ndarray) and sent_feat.ndim == 2 and sent_feat.shape[0] > 0 and sent_feat.shape[1] == word_dim:
                    # Convert to PyTorch tensor and add to the list
                    part_features_list.append(torch.as_tensor(sent_feat, dtype=torch.float32))

        # If this part is empty after processing (e.g., all sentences are invalid or indices are out of range, or the specified index list is empty)
        if not part_features_list:
            # Add a placeholder tensor for this part
            part_features_list.append(torch.zeros((1, word_dim), dtype=torch.float32))
        return part_features_list


    # Method to get single sample data
    def __getitem__(self, index):
        # Get the sample data dictionary for the current index
        sample_data = self.list_of_sample_data_dicts[index]
        # Audio: list of numpy arrays (sentence features)
        audio_all_sents_raw = sample_data['audio']
        # Video: list of numpy arrays (sentence features)
        video_all_sents_raw = sample_data['video']
        # Text: list of sentence strings
        text_all_sents_str = sample_data['text']
        # Get label
        label = self.list_of_labels[index]

        # Determine the total number of sentences based on the number of text sentences
        n_total_sents = len(text_all_sents_str)

        # Prepare placeholder input_ids and attention_mask for empty text parts
        empty_ids, empty_mask = self._tokenize_text_part([])

        # Case 1: If the sample has no sentences at all (n_total_sents == 0)
        if n_total_sents == 0:
            # Context part is empty/placeholder
            ctx_audio_part = self._process_av_part([], [], self.audio_word_dim) # Passing an empty list will result in a placeholder
            ctx_video_part = self._process_av_part([], [], self.video_word_dim)
            ctx_input_ids, ctx_attention_mask = empty_ids, empty_mask
            # Punchline part is empty/placeholder
            pl_audio_part = self._process_av_part([], [], self.audio_word_dim)
            pl_video_part = self._process_av_part([], [], self.video_word_dim)
            pl_input_ids, pl_attention_mask = empty_ids, empty_mask

        # Case 2: If there is only one sentence, treat it as only punchline, context is empty
        elif n_total_sents == 1:
            # Context part is empty/placeholder
            ctx_audio_part = self._process_av_part([], [], self.audio_word_dim) # Passing an empty index list will result in a placeholder
            ctx_video_part = self._process_av_part([], [], self.video_word_dim)
            ctx_input_ids, ctx_attention_mask = empty_ids, empty_mask
            # Punchline part is this one sentence (index 0)
            pl_audio_part = self._process_av_part(audio_all_sents_raw, [0], self.audio_word_dim)
            pl_video_part = self._process_av_part(video_all_sents_raw, [0], self.video_word_dim)
            pl_input_ids, pl_attention_mask = self._tokenize_text_part([text_all_sents_str[0]])

        # Case 3: If there are multiple sentences, split into context and punchline
        else:
            # Context sentence indices: from 0 to the second to last
            ctx_indices = list(range(n_total_sents - 1))
            # Punchline sentence index: only the last one
            pl_indices = [n_total_sents - 1]

            # Process context part
            ctx_audio_part = self._process_av_part(audio_all_sents_raw, ctx_indices, self.audio_word_dim)
            ctx_video_part = self._process_av_part(video_all_sents_raw, ctx_indices, self.video_word_dim)
            ctx_input_ids, ctx_attention_mask = self._tokenize_text_part([text_all_sents_str[i] for i in ctx_indices])

            # Process punchline part
            pl_audio_part = self._process_av_part(audio_all_sents_raw, pl_indices, self.audio_word_dim)
            pl_video_part = self._process_av_part(video_all_sents_raw, pl_indices, self.video_word_dim)
            pl_input_ids, pl_attention_mask = self._tokenize_text_part([text_all_sents_str[i] for i in pl_indices])

        # Return a tuple of context data, punchline data, and label
        return (ctx_audio_part, ctx_video_part, ctx_input_ids, ctx_attention_mask,
                pl_audio_part, pl_video_part, pl_input_ids, pl_attention_mask,
                label)

# --- Custom Collate Function for Context/Punchline Data ---
def custom_collate_fn_context_punchline(batch):
    # batch is a list where each element is the tuple returned by __getitem__
    # Unpack batch data into respective lists
    (ctx_audio_list, ctx_video_list, ctx_ids_list, ctx_mask_list,
     pl_audio_list, pl_video_list, pl_ids_list, pl_mask_list,
     labels_list) = zip(*batch)

    # Directly stack text IDs, masks, and labels (they are already fixed-size tensors)
    batched_ctx_ids = torch.stack(ctx_ids_list)
    batched_ctx_masks = torch.stack(ctx_mask_list)
    batched_pl_ids = torch.stack(pl_ids_list)
    batched_pl_masks = torch.stack(pl_mask_list)
    batched_labels = torch.stack(labels_list)

    # Helper function to process a list of audio/video data for a part (e.g., context audio)
    # part_data_list: A list of samples, where each sample is a list of sentence tensors
    # word_dim_const: Word feature dimension of this modality
    def _collate_av_part(part_data_list, word_dim_const):
        # Get the number of sentences in each sample
        sample_lengths = [len(sample) for sample in part_data_list]
        # Maximum number of sentences in the batch, 0 if empty
        max_sents = max(sample_lengths) if sample_lengths else 0

        # Get the word count of each sentence and find the maximum word count
        sentence_word_counts_flat = []
        for sample in part_data_list: # Iterate through each sample
            for sentence_tensor in sample: # Iterate through each sentence tensor in the sample
                sentence_word_counts_flat.append(sentence_tensor.shape[0]) # Add the word count of this sentence
        # Maximum number of words in the batch, 0 if empty
        max_words = max(sentence_word_counts_flat) if sentence_word_counts_flat else 0

        # Ensure max_words and max_sents are at least 1 to avoid zero dimensions in tensors
        max_words = max(1, max_words)
        max_sents = max(1, max_sents)

        # Create padded feature tensor and length tensor
        # padded_features: (batch_size, max_sentences, max_words, feature_dimension)
        # sentence_lengths_tensor: (batch_size, max_sentences) - records the actual word count of each sentence
        padded_features = torch.zeros(len(part_data_list), max_sents, max_words, word_dim_const)
        sentence_lengths_tensor = torch.zeros(len(part_data_list), max_sents, dtype=torch.long)

        # Iterate through each sample in the batch
        for i, sample in enumerate(part_data_list):
            # Iterate through each sentence tensor in the sample
            for j, sentence_tensor in enumerate(sample):
                # Word count of the current sentence
                num_words = sentence_tensor.shape[0]
                # Pad only if there are words
                if num_words > 0:
                    # Pad features into the padded_features tensor
                    padded_features[i, j, :num_words, :] = sentence_tensor
                    # Record the actual word count into the sentence_lengths_tensor tensor
                    sentence_lengths_tensor[i, j] = num_words
        # Return padded features, list of sentence counts per sample (as tensor), and word counts per sentence tensor
        return padded_features, torch.tensor(sample_lengths, dtype=torch.long), sentence_lengths_tensor

    # Process audio and video data for context and punchline separately
    # ctx_padded_audio: (B, S_ctx_max, W_ctx_max, D_audio)
    # ctx_audio_sl: (B,) - Actual number of sentences per sample for context
    # ctx_audio_ssl: (B, S_ctx_max) - Actual word count of each sentence per sample for context
    ctx_padded_audio, ctx_audio_sl, ctx_audio_ssl = _collate_av_part(ctx_audio_list, _AUDIO_WORD_DIM_CONST)
    ctx_padded_video, ctx_video_sl, ctx_video_ssl = _collate_av_part(ctx_video_list, _VIDEO_WORD_DIM_CONST)
    pl_padded_audio, pl_audio_sl, pl_audio_ssl = _collate_av_part(pl_audio_list, _AUDIO_WORD_DIM_CONST)
    pl_padded_video, pl_video_sl, pl_video_ssl = _collate_av_part(pl_video_list, _VIDEO_WORD_DIM_CONST)

    # Return all processed batch data
    return (ctx_padded_audio, ctx_audio_sl, ctx_audio_ssl, # Context audio (features, sample sentence count, words per sentence)
            ctx_padded_video, ctx_video_sl, ctx_video_ssl, # Context video
            batched_ctx_ids, batched_ctx_masks,             # Context text
            pl_padded_audio, pl_audio_sl, pl_audio_ssl,     # Punchline audio
            pl_padded_video, pl_video_sl, pl_video_ssl,     # Punchline video
            batched_pl_ids, batched_pl_masks,               # Punchline text
            batched_labels)                                 # Labels


# --- Hierarchical LSTM Aggregator ---
class HierarchicalLSTMAggregator(nn.Module):
    # Initialization function
    def __init__(self, word_dim, sentence_lstm_hidden_dim, sample_lstm_hidden_dim,
                 sentence_lstm_layers=1, sample_lstm_layers=1, dropout_rate=0.3):
        super().__init__()
        # Hidden dimension of sentence-level LSTM
        self.sentence_lstm_hidden_dim = sentence_lstm_hidden_dim
        # Hidden dimension of sample-level LSTM
        self.sample_lstm_hidden_dim = sample_lstm_hidden_dim

        # Sentence-level LSTM: input word embeddings, output sentence representation
        self.sentence_lstm = nn.LSTM(word_dim, sentence_lstm_hidden_dim,
                                     num_layers=sentence_lstm_layers, batch_first=True,
                                     bidirectional=False) # Can be set to True if needed, output dimension will become 2*hidden_dim

        # If sentence LSTM is bidirectional, the input dimension of sample LSTM needs to be multiplied by 2
        sample_lstm_input_dim = sentence_lstm_hidden_dim * (2 if self.sentence_lstm.bidirectional else 1)

        # Sample-level LSTM: input sentence representations, output sample representation
        self.sample_lstm = nn.LSTM(sample_lstm_input_dim, sample_lstm_hidden_dim,
                                   num_layers=sample_lstm_layers, batch_first=True,
                                   bidirectional=False) # Can be set to True if needed
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    # Forward propagation function
    def forward(self, features, sample_lengths, sentence_lengths):
        # features: (batch_size, max_sentences, max_words, word_dimension)
        # sample_lengths: (batch_size) - actual number of sentences per sample
        # sentence_lengths: (batch_size, max_sentences) - actual word count per sentence

        # Get the shape of the feature tensor
        batch_size, max_sents, max_words, _ = features.shape
        # Final output dimension of sample LSTM (considering bidirectional case)
        final_output_dim_sample = self.sample_lstm_hidden_dim * (2 if self.sample_lstm.bidirectional else 1)

        # Handle the extreme case where all inputs in the batch are empty
        # If max_sentences or max_words is 0, or batch_size is 0, or all sample_lengths are 0
        if max_sents == 0 or max_words == 0 or batch_size == 0 or torch.all(sample_lengths == 0):
            # Return a zero tensor with shape (batch_size, final_output_dim_sample)
            return torch.zeros(batch_size, final_output_dim_sample, device=features.device)

        # 1. Process sentence level
        # Merge batch and sentence dimensions to pass through LSTM at once
        # (B, S, W, D) -> (B*S, W, D)
        sents_features = features.view(batch_size * max_sents, max_words, -1)
        # (B, S) -> (B*S)
        sents_word_lengths = sentence_lengths.view(batch_size * max_sents)

        # Filter out sentences with length 0 to avoid pack_padded_sequence error
        valid_sents_indices = sents_word_lengths > 0
        # If all sentences are empty (all lengths are 0)
        if not torch.any(valid_sents_indices):
            # Return a zero tensor matching the shape of the sample LSTM output
            return torch.zeros(batch_size, final_output_dim_sample, device=features.device)

        # Get valid sentence features and corresponding lengths
        sents_features_packed_data = sents_features[valid_sents_indices]
        sents_word_lengths_packed_data = sents_word_lengths[valid_sents_indices]

        # Pack padded sequence (length tensor needs to be moved to CPU for packing)
        packed_sents_input = pack_padded_sequence(sents_features_packed_data, sents_word_lengths_packed_data.cpu(),
                                                batch_first=True, enforce_sorted=False)
        # Pass through sentence LSTM
        # h_n_sent: (num_layers*num_directions, B*S_valid, sentence_hidden_dim)
        _, (h_n_sent, _) = self.sentence_lstm(packed_sents_input)

        # Get the actual output dimension of sentence LSTM (considering bidirectional)
        sent_hidden_dim_actual = self.sentence_lstm_hidden_dim * (2 if self.sentence_lstm.bidirectional else 1)
        # Get the hidden state of the last time step (for unidirectional LSTM, take the last layer; for bidirectional, concatenate the last time steps of the last two layers)
        # Output shape: (B*S_valid, sentence_hidden_dim)
        if self.sentence_lstm.bidirectional:
            # Concatenate the forward and backward hidden states of the last time step of the bidirectional LSTM
            sentence_embeddings_valid = torch.cat((h_n_sent[-2,:,:], h_n_sent[-1,:,:]), dim=1)
        else:
            # Unidirectional LSTM, take the hidden state of the last time step of the last layer
            sentence_embeddings_valid = h_n_sent[-1,:,:]
        # Apply dropout to sentence embeddings
        sentence_embeddings_valid = self.dropout(sentence_embeddings_valid)

        # Put valid sentence embeddings back to their original positions, use zero vectors for empty sentences
        # Create a zero tensor with shape (B*S, actual_sentence_hidden_dim)
        all_sentence_embeddings = torch.zeros(batch_size * max_sents, sent_hidden_dim_actual, device=features.device)
        # Fill valid sentence embeddings into corresponding positions
        all_sentence_embeddings[valid_sents_indices] = sentence_embeddings_valid

        # (B*S, H_sent) -> (B, S, H_sent), reshape to sample LSTM input format
        sample_features_for_sample_lstm = all_sentence_embeddings.view(batch_size, max_sents, sent_hidden_dim_actual)

        # 2. Process sample level
        # Pack padded sequence (based on actual number of sentences per sample, sample_lengths)
        # Filter out samples with length 0 (i.e., samples with actual sentence count of 0)
        valid_sample_indices = sample_lengths > 0
        # If all samples are empty (actual sentence counts are all 0)
        if not torch.any(valid_sample_indices):
            # Return a zero tensor matching the shape of the sample LSTM output
            return torch.zeros(batch_size, final_output_dim_sample, device=features.device)

        # Get valid sample features and corresponding lengths
        sample_features_packed_input_data = sample_features_for_sample_lstm[valid_sample_indices]
        sample_lengths_packed_data = sample_lengths[valid_sample_indices]

        # Pack padded sequence
        packed_sample_input = pack_padded_sequence(sample_features_packed_input_data, sample_lengths_packed_data.cpu(),
                                                  batch_first=True, enforce_sorted=False)

        # Pass through sample LSTM
        # h_n_sample: (num_layers*num_directions, B_valid, sample_hidden_dim)
        _, (h_n_sample, _) = self.sample_lstm(packed_sample_input)

        # Get the hidden state of the last time step
        # Output shape: (B_valid, sample_hidden_dim)
        if self.sample_lstm.bidirectional:
            # Concatenate the forward and backward hidden states of the last time step of the bidirectional LSTM
            sample_embeddings_valid = torch.cat((h_n_sample[-2,:,:], h_n_sample[-1,:,:]), dim=1)
        else:
            # Unidirectional LSTM, take the hidden state of the last time step of the last layer
            sample_embeddings_valid = h_n_sample[-1,:,:]
        # Apply dropout to sample embeddings
        sample_embeddings_valid = self.dropout(sample_embeddings_valid)

        # Put valid sample embeddings back to their original positions, use zero vectors for empty samples
        # Create a zero tensor with shape (B, final_output_dim_sample_lstm)
        final_sample_embeddings = torch.zeros(batch_size, final_output_dim_sample, device=features.device)
        # Fill valid sample embeddings into corresponding positions
        final_sample_embeddings[valid_sample_indices] = sample_embeddings_valid
        # Return final sample embeddings
        return final_sample_embeddings


# --- GLU Linear Layer ---
class GLULinear(nn.Module):
    # Initialization function, input dimension and output dimension
    def __init__(self, input_dim, output_dim):
        super(GLULinear, self).__init__()
        # The first linear layer is followed by a GELU activation function
        self.layer1 = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU())
        # The second linear layer
        self.layer2 = nn.Linear(input_dim, output_dim)
    # Forward propagation function
    def forward(self, x):
        # Element-wise multiplication of the outputs of the two linear layers
        return self.layer1(x) * self.layer2(x)

# --- Advanced Cross-Attention/Self-Attention Module ---
class MultiHeadAttentionModule(nn.Module):
    # Initialization function
    # dim: feature dimension, num_heads: number of attention heads
    def __init__(self, dim, num_heads=1):
        super(MultiHeadAttentionModule, self).__init__()
        # Feature dimension
        self.dim = dim
        # Number of attention heads
        self.num_heads = num_heads
        # Dimension of each head
        self.head_dim = dim // num_heads
        # Ensure dimension is divisible by the number of heads
        if self.head_dim * num_heads != self.dim:
            raise ValueError("dim must be divisible by num_heads")

        # Linear layer to generate Key
        self.K_layer = nn.Linear(dim, dim, bias=False)
        # Linear layer to generate Value
        self.V_layer = nn.Linear(dim, dim, bias=False)
        # Linear layer to generate Query
        self.Q_layer = nn.Linear(dim, dim, bias=False)
        # Softmax layer, used to calculate attention weights
        self.attend = nn.Softmax(dim = -1)
        # Fully connected layer before output
        self.fc_out = nn.Linear(dim, dim)

    # Forward propagation function
    # feat1_query is Query, feat2_key_value is Key and Value
    # mask: optional attention mask
    def forward(self, feat1_query, feat2_key_value, mask=None):
        # Query shape: (batch_size, Query_sequence_length, Query_dimension)
        B_q, N_q, C_q = feat1_query.shape
        # Key/Value shape: (batch_size, Key/Value_sequence_length, Key/Value_dimension)
        B_kv, N_kv, C_kv = feat2_key_value.shape

        # Check if batch sizes of Query and Key/Value match
        if B_q != B_kv: raise ValueError(f"Batch sizes do not match: Query is {B_q}, Key/Value is {B_kv}")

        # Generate Q, K, V and adjust shape for multi-head: (batch, num_heads, sequence_length, head_dimension)
        # Q: (B, N_q, C_q) -> (B, N_q, num_heads, head_dim) -> (B, num_heads, N_q, head_dim)
        Q = self.Q_layer(feat1_query).reshape(B_q, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # K: (B, N_kv, C_kv) -> (B, N_kv, num_heads, head_dim) -> (B, num_heads, N_kv, head_dim)
        K = self.K_layer(feat2_key_value).reshape(B_kv, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # V: (B, N_kv, C_kv) -> (B, N_kv, num_heads, head_dim) -> (B, num_heads, N_kv, head_dim)
        V = self.V_layer(feat2_key_value).reshape(B_kv, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate dot product of Q and K_transpose and scale ( scaled_dot_product = (Q @ K.T) / sqrt(head_dim) )
        # dots shape: (B, num_heads, N_q, N_kv)
        dots = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        # If a mask is provided (usually a padding mask for K,V sequence, shape B, N_kv)
        if mask is not None:
            # unsqueeze expands the mask to (B, 1, 1, N_kv) to match the shape of dots (B, nH, N_q, N_kv) for broadcasting
            # Fill positions in dots where mask is 0 (i.e., padding positions) with a very small value, so their weight approaches 0 after softmax
            dots = dots.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        # Calculate attention weights (attn_weights shape: B, num_heads, N_q, N_kv)
        attn_weights = self.attend(dots)
        # Attention weights weighted V (out shape: B, num_heads, N_q, head_dim)
        out = torch.matmul(attn_weights, V)
        # Permute and merge multi-head results: (B, num_heads, N_q, head_dim) -> (B, N_q, num_heads, head_dim) -> (B, N_q, dim)
        out = out.permute(0, 2, 1, 3).reshape(B_q, N_q, self.dim)
        # Pass through output fully connected layer
        out = self.fc_out(out)
        # Return final output
        return out

# --- Adapted Single Stream Processor ---
class AdaptedSingleStreamProcessor(nn.Module):
    # Initialization function
    # audio_video_input_dim: Input dimension after audio/video projection
    # bert_hidden_size: BERT's hidden layer size
    # max_bert_len_for_lstm: Maximum input sequence length expected by the internal text LSTM
    # lstm_hidden_size: Hidden size of the internal text LSTM
    # attention_token_dim: Dimension of attention tokens
    # num_attention_tokens_per_modal: Number of tokens output after processing each modality
    # active_modalities: Tuple of active modalities, e.g., ('audio', 'video', 'text')
    # num_ca_sa_heads: Number of heads for cross-attention and self-attention modules
    # dropout_rate: Dropout rate for the text FC part
    def __init__(self, audio_video_input_dim, bert_hidden_size, max_bert_len_for_lstm,
                 lstm_hidden_size, attention_token_dim, num_attention_tokens_per_modal,
                 active_modalities=('audio', 'video', 'text'), num_ca_sa_heads=1, dropout_rate=0.5):
        super().__init__()
        # Number of tokens output after processing each modality
        self.n_tokens_per_modal = num_attention_tokens_per_modal
        # Dimension of attention tokens
        self.attention_token_dim = attention_token_dim
        # Maximum input sequence length expected by the internal text LSTM
        self.max_bert_len_for_lstm_input = max_bert_len_for_lstm
        # Active modalities
        self.active_modalities = active_modalities
        # Flattened feature dimension output by each modality processor (N * C)
        self.expected_feature_dim_after_mod_proc = self.n_tokens_per_modal * self.attention_token_dim

        # Audio feature processor: receives projected features, maps to NxC token representation
        self.audio_feat_processor_to_tokens = nn.Sequential(
            GLULinear(audio_video_input_dim, 1024),
            GLULinear(1024, self.expected_feature_dim_after_mod_proc),
            nn.LayerNorm(self.expected_feature_dim_after_mod_proc) # Layer normalization
        )
        # Video feature processor: logic same as audio
        self.vision_feat_processor_to_tokens = nn.Sequential(
            GLULinear(audio_video_input_dim, 1024),
            GLULinear(1024, self.expected_feature_dim_after_mod_proc),
            nn.LayerNorm(self.expected_feature_dim_after_mod_proc)
        )
        # Text processing: BERT hidden state -> LSTM -> Fully connected layer -> NxC token representation
        # Text LSTM processor
        self.text_lstm_processor = nn.LSTM(bert_hidden_size, lstm_hidden_size, batch_first=True)
        # Text FC processor, maps LSTM output to token representation
        self.text_fc_processor_to_tokens = nn.Sequential(
            nn.Dropout(dropout_rate), # Dropout layer
            # LSTM output is (B, S_lstm, H_lstm), after reshape it's (B, S_lstm * H_lstm)
            GLULinear(lstm_hidden_size * self.max_bert_len_for_lstm_input, 1024),
            GLULinear(1024, self.expected_feature_dim_after_mod_proc),
            nn.LayerNorm(self.expected_feature_dim_after_mod_proc)
        )

        # Attention module instantiation
        # ZA: Audio cross-attention (query is concatenation of all modalities, key/value are audio tokens)
        self.ZA = MultiHeadAttentionModule(dim=attention_token_dim, num_heads=num_ca_sa_heads)
        # ZV: Video cross-attention
        self.ZV = MultiHeadAttentionModule(dim=attention_token_dim, num_heads=num_ca_sa_heads)
        # ZT: Text cross-attention
        self.ZT = MultiHeadAttentionModule(dim=attention_token_dim, num_heads=num_ca_sa_heads)
        # SA_stream: Intra-stream self-attention
        self.SA_stream = MultiHeadAttentionModule(dim=attention_token_dim, num_heads=num_ca_sa_heads)
        # Final output dimension of this stream processor (after averaging SA output, or, dimension of a single token)
        self.output_final_dim = attention_token_dim

    # Forward propagation function
    # audio_input_proj, vision_input_proj from Hierarchical LSTM + Projector layer (B, D_projector)
    # text_sequence_input_bert is BERT's hidden state (B, S_bert, D_bert)
    def forward(self, audio_input_proj, vision_input_proj, text_sequence_input_bert):
        # Dynamically determine batch size
        b = 0
        if audio_input_proj is not None and audio_input_proj.nelement() > 0: b = audio_input_proj.shape[0]
        elif vision_input_proj is not None and vision_input_proj.nelement() > 0: b = vision_input_proj.shape[0]
        elif text_sequence_input_bert is not None and text_sequence_input_bert.nelement() > 0: b = text_sequence_input_bert.shape[0]

        # Handle empty batch (all inputs are empty or None)
        if b == 0:
            dev = torch.device("cpu") # Default device
            # Try to get device from valid input
            if audio_input_proj is not None and audio_input_proj.nelement() > 0: dev = audio_input_proj.device
            elif vision_input_proj is not None and vision_input_proj.nelement() > 0: dev = vision_input_proj.device
            elif text_sequence_input_bert is not None and text_sequence_input_bert.nelement() > 0: dev = text_sequence_input_bert.device

            # Create empty flat features and stream output
            empty_flat = torch.zeros(0, self.expected_feature_dim_after_mod_proc, device=dev)
            empty_stream_out = torch.zeros(0, 1, self.output_final_dim, device=dev)
            # Return empty flat features for contrastive loss and empty stream output
            return empty_flat, empty_flat, empty_flat, empty_stream_out

        # Get current device (ensure at least one valid input to determine device)
        device = audio_input_proj.device if audio_input_proj is not None and audio_input_proj.nelement() > 0 else \
                 (vision_input_proj.device if vision_input_proj is not None and vision_input_proj.nelement() > 0 else \
                  text_sequence_input_bert.device)

        # Initialize flat features for contrastive loss (audio_f_flat) and token features for attention (audio_f_tokens)
        # Audio processing
        audio_f_flat = torch.zeros(b, self.expected_feature_dim_after_mod_proc, device=device)
        audio_f_tokens = torch.zeros(b, self.n_tokens_per_modal, self.attention_token_dim, device=device)
        # If audio modality is active, input is not empty, and input is not all zeros (indicates actual content)
        if 'audio' in self.active_modalities and audio_input_proj is not None and audio_input_proj.nelement() > 0 and audio_input_proj.abs().sum() > 1e-9 :
            audio_f_flat = self.audio_feat_processor_to_tokens(audio_input_proj) # (B, N*C)
            audio_f_tokens = audio_f_flat.view(b, self.n_tokens_per_modal, self.attention_token_dim) # (B, N, C)

        # Video processing (logic same as audio)
        vis_f_flat = torch.zeros(b, self.expected_feature_dim_after_mod_proc, device=device)
        vis_f_tokens = torch.zeros(b, self.n_tokens_per_modal, self.attention_token_dim, device=device)
        if 'video' in self.active_modalities and vision_input_proj is not None and vision_input_proj.nelement() > 0 and vision_input_proj.abs().sum() > 1e-9:
            vis_f_flat = self.vision_feat_processor_to_tokens(vision_input_proj)
            vis_f_tokens = vis_f_flat.view(b, self.n_tokens_per_modal, self.attention_token_dim)

        # Text processing
        text_f_flat = torch.zeros(b, self.expected_feature_dim_after_mod_proc, device=device)
        text_f_tokens = torch.zeros(b, self.n_tokens_per_modal, self.attention_token_dim, device=device)
        if 'text' in self.active_modalities and text_sequence_input_bert is not None and text_sequence_input_bert.nelement() > 0 and text_sequence_input_bert.abs().sum() > 1e-9:
            # Get current BERT output sequence length
            current_bert_seq_len = text_sequence_input_bert.shape[1]
            text_sequence_input_bert_adjusted = text_sequence_input_bert
            # Adjust BERT output sequence length to match LSTM expected input
            if current_bert_seq_len != self.max_bert_len_for_lstm_input:
                if current_bert_seq_len > self.max_bert_len_for_lstm_input: # Truncate if too long
                    text_sequence_input_bert_adjusted = text_sequence_input_bert[:, :self.max_bert_len_for_lstm_input, :]
                else: # Pad with zeros if too short
                    padding_needed = self.max_bert_len_for_lstm_input - current_bert_seq_len
                    # Create padding tensor (B, padding_needed, D_bert)
                    padding_tensor = torch.zeros(b, padding_needed, text_sequence_input_bert.shape[2], device=device)
                    # Concatenate original BERT output and padding tensor
                    text_sequence_input_bert_adjusted = torch.cat([text_sequence_input_bert, padding_tensor], dim=1)

            # Pass through text LSTM
            lstm_output, _ = self.text_lstm_processor(text_sequence_input_bert_adjusted) # (B, S_lstm, H_lstm)
            # Flatten LSTM output: (B, S_lstm * H_lstm)
            text_f_flat_from_lstm = lstm_output.reshape(b, -1)
            # Process flattened LSTM output through FC layer
            text_f_flat = self.text_fc_processor_to_tokens(text_f_flat_from_lstm) # (B, N*C)
            # Reshape to token form
            text_f_tokens = text_f_flat.view(b, self.n_tokens_per_modal, self.attention_token_dim) # (B, N, C)

        # Collect tokens from active modalities with content
        active_mod_token_lists = []
        if 'audio' in self.active_modalities and audio_f_tokens.abs().sum() > 1e-9: active_mod_token_lists.append(audio_f_tokens)
        if 'video' in self.active_modalities and vis_f_tokens.abs().sum() > 1e-9: active_mod_token_lists.append(vis_f_tokens)
        if 'text'  in self.active_modalities and text_f_tokens.abs().sum() > 1e-9: active_mod_token_lists.append(text_f_tokens)

        # If there are no active modalities with content
        if not active_mod_token_lists:
            # Return flat features and zero stream output (because there's no content for attention calculation)
            return audio_f_flat, vis_f_flat, text_f_flat, torch.zeros(b, 1, self.output_final_dim, device=device)

        # Concatenate tokens of active modalities as Query for cross-attention
        # query_for_modality_ca shape: (B, num_active_modalities * N, C_token)
        query_for_modality_ca = torch.cat(active_mod_token_lists, dim=1)

        # Perform inter-modality cross-attention
        # Initialize result tensor
        res_za, res_zv, res_zt = torch.zeros_like(query_for_modality_ca), torch.zeros_like(query_for_modality_ca), torch.zeros_like(query_for_modality_ca)
        # If audio is active and has content
        if 'audio' in self.active_modalities and audio_f_tokens.abs().sum() > 1e-9:
            # query_for_modality_ca as Query, audio_f_tokens as Key and Value
            res_za = self.ZA(query_for_modality_ca, audio_f_tokens)
        # If video is active and has content
        if 'video' in self.active_modalities and vis_f_tokens.abs().sum() > 1e-9:
            res_zv = self.ZV(query_for_modality_ca, vis_f_tokens)
        # If text is active and has content
        if 'text' in self.active_modalities and text_f_tokens.abs().sum() > 1e-9:
            res_zt = self.ZT(query_for_modality_ca, text_f_tokens)

        # Merge cross-attention results (element-wise addition)
        feat_after_mod_ca = res_za + res_zv + res_zt
        # Intra-stream self-attention, with residual connection
        # feat_after_mod_ca as Query, Key, and Value
        feat_after_sa_stream = self.SA_stream(feat_after_mod_ca, feat_after_mod_ca) + feat_after_mod_ca
        # Average the features after self-attention along the sequence dimension to get the final stream representation
        stream_output_representation = torch.mean(feat_after_sa_stream, dim=1) # (B, C_token)

        # Return flat features for contrastive loss, and the final stream output representation (add a dimension to match the expected (B, 1, C_token) shape)
        return audio_f_flat, vis_f_flat, text_f_flat, stream_output_representation.unsqueeze(1)


# --- Main Model: ContextPunchlineHumorModelNew ---
class ContextPunchlineHumorModelNew(nn.Module):
    # Initialization function
    def __init__(self,
                 bert_model_name_or_path,
                 audio_word_dim, video_word_dim,
                 sentence_lstm_hidden_dim, sample_lstm_hidden_dim, hier_lstm_dropout,
                 projector_output_dim,
                 bert_hidden_size_actual, max_bert_len_for_lstm,
                 text_lstm_hidden_size_in_stream,
                 attention_token_dim, num_attention_tokens_per_modal,
                 stream_ca_sa_heads, stream_dropout_rate,
                 final_cross_attention_heads, # MODIFIED: This will now be used for the new final fusion heads
                 mlp_hidden_dim, num_classes,
                 ):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_name_or_path)

        self.ctx_audio_hier_lstm = HierarchicalLSTMAggregator(audio_word_dim, sentence_lstm_hidden_dim, sample_lstm_hidden_dim, dropout_rate=hier_lstm_dropout)
        self.ctx_video_hier_lstm = HierarchicalLSTMAggregator(video_word_dim, sentence_lstm_hidden_dim, sample_lstm_hidden_dim, dropout_rate=hier_lstm_dropout)
        self.ctx_audio_projector = nn.Linear(sample_lstm_hidden_dim, projector_output_dim)
        self.ctx_video_projector = nn.Linear(sample_lstm_hidden_dim, projector_output_dim)
        self.context_processor = AdaptedSingleStreamProcessor(
            audio_video_input_dim=projector_output_dim,
            bert_hidden_size=bert_hidden_size_actual,
            max_bert_len_for_lstm=max_bert_len_for_lstm,
            lstm_hidden_size=text_lstm_hidden_size_in_stream,
            attention_token_dim=attention_token_dim,
            num_attention_tokens_per_modal=num_attention_tokens_per_modal,
            active_modalities=('audio', 'video', 'text'),
            num_ca_sa_heads=stream_ca_sa_heads,
            dropout_rate=stream_dropout_rate
        )

        self.pl_audio_hier_lstm = HierarchicalLSTMAggregator(audio_word_dim, sentence_lstm_hidden_dim, sample_lstm_hidden_dim, dropout_rate=hier_lstm_dropout)
        self.pl_video_hier_lstm = HierarchicalLSTMAggregator(video_word_dim, sentence_lstm_hidden_dim, sample_lstm_hidden_dim, dropout_rate=hier_lstm_dropout)
        self.pl_audio_projector = nn.Linear(sample_lstm_hidden_dim, projector_output_dim)
        self.pl_video_projector = nn.Linear(sample_lstm_hidden_dim, projector_output_dim)
        self.punchline_processor = AdaptedSingleStreamProcessor(
            audio_video_input_dim=projector_output_dim,
            bert_hidden_size=bert_hidden_size_actual,
            max_bert_len_for_lstm=max_bert_len_for_lstm,
            lstm_hidden_size=text_lstm_hidden_size_in_stream,
            attention_token_dim=attention_token_dim,
            num_attention_tokens_per_modal=num_attention_tokens_per_modal,
            active_modalities=('audio', 'video', 'text'),
            num_ca_sa_heads=stream_ca_sa_heads,
            dropout_rate=stream_dropout_rate
        )

        # --- MODIFIED: Final Fusion ---
        self.final_fusion_input_dim = attention_token_dim

        self.final_ca_query_streams_on_ctx_kv = MultiHeadAttentionModule(
            dim=self.final_fusion_input_dim,
            num_heads=final_cross_attention_heads
        )
        self.final_ca_query_streams_on_pl_kv = MultiHeadAttentionModule(
            dim=self.final_fusion_input_dim,
            num_heads=final_cross_attention_heads
        )
        self.final_fusion_sa_after_ca_sum = MultiHeadAttentionModule(
            dim=self.final_fusion_input_dim,
            num_heads=final_cross_attention_heads
        )
        # REMOVED: self.cross_attention_final

        self.mlp = nn.Sequential(
            nn.Linear(self.final_fusion_input_dim, mlp_hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2), nn.ReLU(), nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(mlp_hidden_dim // 2, num_classes)

    def forward(self,
                ctx_padded_audio, ctx_audio_sl, ctx_audio_ssl,
                ctx_padded_video, ctx_video_sl, ctx_video_ssl,
                ctx_input_ids, ctx_attention_mask,
                pl_padded_audio, pl_audio_sl, pl_audio_ssl,
                pl_padded_video, pl_video_sl, pl_video_ssl,
                pl_input_ids, pl_attention_mask,
                current_modality_config=None, tokenizer_for_padding=None
                ):

        actual_hier_lstm_output_dim_ctx_a = self.ctx_audio_hier_lstm.sample_lstm_hidden_dim * (2 if self.ctx_audio_hier_lstm.sample_lstm.bidirectional else 1)
        ctx_a_vec = torch.zeros(ctx_padded_audio.shape[0], actual_hier_lstm_output_dim_ctx_a, device=ctx_padded_audio.device)
        if current_modality_config is None or current_modality_config.get('audio', True):
            if torch.any(ctx_audio_sl > 0):
                ctx_a_vec = self.ctx_audio_hier_lstm(ctx_padded_audio, ctx_audio_sl, ctx_audio_ssl)

        actual_hier_lstm_output_dim_ctx_v = self.ctx_video_hier_lstm.sample_lstm_hidden_dim * (2 if self.ctx_video_hier_lstm.sample_lstm.bidirectional else 1)
        ctx_v_vec = torch.zeros(ctx_padded_video.shape[0], actual_hier_lstm_output_dim_ctx_v, device=ctx_padded_video.device)
        if current_modality_config is None or current_modality_config.get('video', True):
            if torch.any(ctx_video_sl > 0):
                ctx_v_vec = self.ctx_video_hier_lstm(ctx_padded_video, ctx_video_sl, ctx_video_ssl)

        ctx_a_proj = self.ctx_audio_projector(ctx_a_vec)
        ctx_v_proj = self.ctx_video_projector(ctx_v_vec)

        ctx_bert_hs = torch.zeros(ctx_input_ids.shape[0], ctx_input_ids.shape[1], self.bert_model.config.hidden_size, device=ctx_input_ids.device)
        if current_modality_config is None or current_modality_config.get('text', True):
            if torch.any(ctx_attention_mask.sum(dim=1) > 0):
                ctx_bert_outputs = self.bert_model(input_ids=ctx_input_ids, attention_mask=ctx_attention_mask)
                ctx_bert_hs = ctx_bert_outputs.last_hidden_state.to(torch.float32)
        ctx_audio_f_flat, ctx_vis_f_flat, ctx_text_f_flat, ctx_stream_repr = self.context_processor(ctx_a_proj, ctx_v_proj, ctx_bert_hs)

        actual_hier_lstm_output_dim_pl_a = self.pl_audio_hier_lstm.sample_lstm_hidden_dim * (2 if self.pl_audio_hier_lstm.sample_lstm.bidirectional else 1)
        pl_a_vec = torch.zeros(pl_padded_audio.shape[0], actual_hier_lstm_output_dim_pl_a, device=pl_padded_audio.device)
        if current_modality_config is None or current_modality_config.get('audio', True):
            if torch.any(pl_audio_sl > 0):
                pl_a_vec = self.pl_audio_hier_lstm(pl_padded_audio, pl_audio_sl, pl_audio_ssl)

        actual_hier_lstm_output_dim_pl_v = self.pl_video_hier_lstm.sample_lstm_hidden_dim * (2 if self.pl_video_hier_lstm.sample_lstm.bidirectional else 1)
        pl_v_vec = torch.zeros(pl_padded_video.shape[0], actual_hier_lstm_output_dim_pl_v, device=pl_padded_video.device)
        if current_modality_config is None or current_modality_config.get('video', True):
            if torch.any(pl_video_sl > 0):
                pl_v_vec = self.pl_video_hier_lstm(pl_padded_video, pl_video_sl, pl_video_ssl)

        pl_a_proj = self.pl_audio_projector(pl_a_vec)
        pl_v_proj = self.pl_video_projector(pl_v_vec)

        pl_bert_hs = torch.zeros(pl_input_ids.shape[0], pl_input_ids.shape[1], self.bert_model.config.hidden_size, device=pl_input_ids.device)
        if current_modality_config is None or current_modality_config.get('text', True):
            if torch.any(pl_attention_mask.sum(dim=1) > 0):
                pl_bert_outputs = self.bert_model(input_ids=pl_input_ids, attention_mask=pl_attention_mask)
                pl_bert_hs = pl_bert_outputs.last_hidden_state.to(torch.float32)
        pl_audio_f_flat, pl_vis_f_flat, pl_text_f_flat, pl_stream_repr = self.punchline_processor(pl_a_proj, pl_v_proj, pl_bert_hs)

        # --- MODIFIED: New final fusion logic ---
        streams_query = torch.cat((ctx_stream_repr, pl_stream_repr), dim=1)
        res_ca_ctx = self.final_ca_query_streams_on_ctx_kv(streams_query, ctx_stream_repr)
        res_ca_pl = self.final_ca_query_streams_on_pl_kv(streams_query, pl_stream_repr)
        fused_after_ca = res_ca_ctx + res_ca_pl
        fused_after_sa = self.final_fusion_sa_after_ca_sum(fused_after_ca, fused_after_ca)
        fused_after_sa = fused_after_sa + fused_after_ca # Residual connection for the self-attention on fused representations
        fused_representation = torch.mean(fused_after_sa, dim=1)

        mlp_out = self.mlp(fused_representation)
        logits = self.classifier(mlp_out)

        contrastive_features = {
            'ctx_audio': ctx_audio_f_flat, 'ctx_video': ctx_vis_f_flat, 'ctx_text': ctx_text_f_flat,
            'pl_audio': pl_audio_f_flat, 'pl_video': pl_vis_f_flat, 'pl_text': pl_text_f_flat
        }
        return logits, contrastive_features


# --- Contrastive Loss Function ---
class ContrastiveLossELI5(nn.Module):
    # Initialization function, temperature coefficient
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        # Use CrossEntropyLoss to calculate loss (SimCLR style)
        self.criterion = nn.CrossEntropyLoss()

    # Forward propagation function
    # emb_i, emb_j are embeddings from different modalities or views (B, D)
    def forward(self, emb_i, emb_j):
        # Get batch size
        batch_size = emb_i.shape[0]
        # Contrastive loss requires at least 2 samples to compute, otherwise return 0 loss
        if batch_size <= 1:
            return torch.tensor(0.0, device=emb_i.device, requires_grad=True)

        # Check if embeddings are all zeros, if so, loss is 0 (to avoid NaN)
        # If the sum of absolute values of all elements in either embedding tensor is less than a very small value, it is considered empty or all zeros
        if emb_i.abs().sum() < 1e-9 or emb_j.abs().sum() < 1e-9:
            return torch.tensor(0.0, device=emb_i.device, requires_grad=True)

        # L2 normalize embedding vectors
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # Concatenate the two groups of normalized embeddings along the batch dimension: (2*B, D)
        representations = torch.cat([z_i, z_j], dim=0)
        # Calculate similarity matrix (cosine similarity between all sample pairs, then divide by temperature)
        # (2*B, D) @ (D, 2*B) -> (2*B, 2*B)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Create labels: for each sample in z_i, its positive sample is the corresponding sample in z_j
        # For example, row similarity_matrix[0] is the similarity of z_i[0] with all representations
        # Its positive sample z_j[0] has index batch_size + 0 in representations
        labels_i_to_j = torch.arange(batch_size, device=emb_i.device) + batch_size
        # For each sample in z_j, its positive sample is the corresponding sample in z_i
        # For example, row similarity_matrix[batch_size+0] is the similarity of z_j[0] with all representations
        # Its positive sample z_i[0] has index 0 in representations
        labels_j_to_i = torch.arange(batch_size, device=emb_i.device)

        # Calculate loss, separately for z_i querying z_j and z_j querying z_i
        # loss_i: z_i as anchor, corresponding sample in z_j as positive
        # similarity_matrix[:batch_size] is the similarity of z_i with all representations (B, 2*B)
        loss_i = self.criterion(similarity_matrix[:batch_size], labels_i_to_j)
        # loss_j: z_j as anchor, corresponding sample in z_i as positive
        # similarity_matrix[batch_size:] is the similarity of z_j with all representations (B, 2*B)
        loss_j = self.criterion(similarity_matrix[batch_size:], labels_j_to_i)
        # Return average loss
        return (loss_i + loss_j) / 2.0


# --- Modified Training Function (Only contrastive loss calculation method is changed) ---
def train_new_model(model, data_loader, optimizer, scheduler,
                    bce_criterion, contrastive_loss_fn, device, epoch, num_epochs,
                    contrastive_loss_weight, current_modality_config, tokenizer_for_padding):
    # Set model to training mode
    model.train()
    # The BERT part of the model is already globally frozen externally, no explicit model.bert_model.eval() needed here

    # Initialize total BCE loss, total contrastive loss, total loss
    total_bce_loss = 0
    total_simclr_loss = 0 # Used to accumulate final_simclr_loss_for_batch for each batch
    total_loss = 0
    # Create tqdm progress bar to display training progress
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train {current_modality_config['name']}]", leave=False)

    # Iterate through each batch in the data loader
    for batch_idx, batch in enumerate(progress_bar):
        # Unpack data from collate_fn
        (ctx_a_feat, ctx_a_sl, ctx_a_ssl, ctx_v_feat, ctx_v_sl, ctx_v_ssl, ctx_ids, ctx_mask,
         pl_a_feat, pl_a_sl, pl_a_ssl, pl_v_feat, pl_v_sl, pl_v_ssl, pl_ids, pl_mask,
         labels) = batch

        # Get current batch size
        current_batch_size = ctx_a_feat.shape[0]
        # If batch is empty, skip
        if current_batch_size == 0: continue

        # Move data to the specified device (excluding the last label)
        batch_data_on_device = []
        for tensor_item in batch[:-1]:
            batch_data_on_device.append(tensor_item.to(device))
        # Move labels to device and convert to long type
        labels = labels.to(device).long()

        # Clear optimizer gradients
        optimizer.zero_grad()

        # Model forward pass, get classification logits and contrastive features
        logits, contrastive_feats = model(*batch_data_on_device, current_modality_config=current_modality_config, tokenizer_for_padding=tokenizer_for_padding)

        # Calculate BCE classification loss
        bce_loss = bce_criterion(logits, labels)

        # --- Contrastive Loss Calculation (modified to averaging method) ---
        final_simclr_loss_for_batch = torch.tensor(0.0, device=device) # Initialize contrastive loss for this batch
        if current_batch_size > 1 and contrastive_loss_weight > 0:
            accumulated_contrastive_loss_components = []

            # Contrastive loss for the context stream
            ctx_individual_losses = []
            if current_modality_config.get('audio', True) and current_modality_config.get('video', True):
                if contrastive_feats['ctx_audio'].nelement() > 0 and contrastive_feats['ctx_video'].nelement() > 0:
                    ctx_individual_losses.append(contrastive_loss_fn(contrastive_feats['ctx_audio'], contrastive_feats['ctx_video']))
            if current_modality_config.get('audio', True) and current_modality_config.get('text', True):
                if contrastive_feats['ctx_audio'].nelement() > 0 and contrastive_feats['ctx_text'].nelement() > 0:
                    ctx_individual_losses.append(contrastive_loss_fn(contrastive_feats['ctx_audio'], contrastive_feats['ctx_text']))
            if current_modality_config.get('video', True) and current_modality_config.get('text', True):
                if contrastive_feats['ctx_video'].nelement() > 0 and contrastive_feats['ctx_text'].nelement() > 0:
                    ctx_individual_losses.append(contrastive_loss_fn(contrastive_feats['ctx_video'], contrastive_feats['ctx_text']))

            if ctx_individual_losses: # Calculate average only if the list is not empty
                accumulated_contrastive_loss_components.append(torch.mean(torch.stack(ctx_individual_losses)))

            # Contrastive loss for the punchline stream
            pl_individual_losses = []
            if current_modality_config.get('audio', True) and current_modality_config.get('video', True):
                if contrastive_feats['pl_audio'].nelement() > 0 and contrastive_feats['pl_video'].nelement() > 0:
                    pl_individual_losses.append(contrastive_loss_fn(contrastive_feats['pl_audio'], contrastive_feats['pl_video']))
            if current_modality_config.get('audio', True) and current_modality_config.get('text', True):
                if contrastive_feats['pl_audio'].nelement() > 0 and contrastive_feats['pl_text'].nelement() > 0:
                    pl_individual_losses.append(contrastive_loss_fn(contrastive_feats['pl_audio'], contrastive_feats['pl_text']))
            if current_modality_config.get('video', True) and current_modality_config.get('text', True):
                if contrastive_feats['pl_video'].nelement() > 0 and contrastive_feats['pl_text'].nelement() > 0:
                    pl_individual_losses.append(contrastive_loss_fn(contrastive_feats['pl_video'], contrastive_feats['pl_text']))

            if pl_individual_losses: # Calculate average only if the list is not empty
                accumulated_contrastive_loss_components.append(torch.mean(torch.stack(pl_individual_losses)))

            # Calculate the final contrastive loss (if multiple components exist, take their average)
            if accumulated_contrastive_loss_components:
                final_simclr_loss_for_batch = torch.mean(torch.stack(accumulated_contrastive_loss_components))
            # else: final_simclr_loss_for_batch remains its initial value of 0.0

        # Total loss = BCE loss + contrastive_loss_weight * calculated batch contrastive loss
        current_loss = bce_loss + contrastive_loss_weight * final_simclr_loss_for_batch

        # Backpropagate to calculate gradients
        current_loss.backward()
        # Update model parameters
        optimizer.step()
        # If a learning rate scheduler is used
        if scheduler is not None:
            # Update learning rate
            scheduler.step()

        # Accumulate loss values (item() gets scalar value)
        total_bce_loss += bce_loss.item()
        total_simclr_loss += final_simclr_loss_for_batch.item() # Accumulate the calculated batch contrastive loss
        total_loss += current_loss.item()
        # Update progress bar display information
        progress_bar.set_postfix(loss=f"{current_loss.item():.4f}", bce=f"{bce_loss.item():.4f}", simclr=f"{final_simclr_loss_for_batch.item():.4f}")

    # If data loader is not empty
    if len(data_loader) > 0:
        # Calculate average loss
        avg_loss = total_loss / len(data_loader)
        avg_bce_loss = total_bce_loss / len(data_loader)
        avg_simclr_loss = total_simclr_loss / len(data_loader)
        # Print average training loss for the current epoch
        print(f"Epoch {epoch+1} ({current_modality_config['name']}) Train Avg Loss: {avg_loss:.4f}, BCE: {avg_bce_loss:.4f}, SimCLR: {avg_simclr_loss:.4f}")


# --- Validation/Test Function (Added F1 Score) ---
def validate_or_test_new_model(model, data_loader, bce_criterion, device, epoch, num_epochs,
                               current_modality_config, tokenizer_for_padding, mode="Val"):
    # Set model to evaluation mode
    model.eval()
    # Initialize total BCE loss
    total_bce_loss = 0
    # List to store all prediction results
    all_preds = []
    # List to store all true labels
    all_labels = []

    # Set progress bar description (corrected logic)
    if mode == "Test" and epoch is None:
        desc = f"Final Test [{current_modality_config['name']}]"
    elif mode == "Test": # and epoch is not None (implicitly for this branch after the first)
        desc = f"Test after Epoch {epoch+1} [{current_modality_config['name']}]"
    elif mode == "Val": # epoch should not be None for validation
        desc = f"Epoch {epoch+1}/{num_epochs} [{mode} {current_modality_config['name']}]"
    else: # Fallback, though ideally all cases are covered
        desc = f"Processing [{mode} {current_modality_config['name']}]"


    # Do not calculate gradients within this block to save memory and computation
    with torch.no_grad():
        # Iterate through each batch in the data loader
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=desc, leave=False)):
            # Unpack data
            (ctx_a_feat, ctx_a_sl, ctx_a_ssl, ctx_v_feat, ctx_v_sl, ctx_v_ssl, ctx_ids, ctx_mask,
             pl_a_feat, pl_a_sl, pl_a_ssl, pl_v_feat, pl_v_sl, pl_v_ssl, pl_ids, pl_mask,
             labels) = batch

            # Get current batch size
            current_batch_size = ctx_a_feat.shape[0]
            # If batch is empty, skip
            if current_batch_size == 0: continue

            # Move data to device
            batch_data_on_device = [t.to(device) for t in batch[:-1]]
            labels = labels.to(device).long()

            # Model forward pass, ignore contrastive features (not needed during validation/testing)
            logits, _ = model(*batch_data_on_device, current_modality_config=current_modality_config, tokenizer_for_padding=tokenizer_for_padding)

            # Calculate BCE loss
            bce_loss = bce_criterion(logits, labels)
            # Accumulate BCE loss
            total_bce_loss += bce_loss.item()
            # Get predicted class (index of the max value in logits)
            preds = torch.argmax(logits, dim=1)
            # Store prediction results (convert to numpy array)
            all_preds.extend(preds.cpu().numpy())
            # Store true labels (convert to numpy array)
            all_labels.extend(labels.cpu().numpy())

    # If data loader is empty or no labels were collected
    if len(data_loader) == 0 or len(all_labels) == 0 :
        print(f"Epoch {epoch+1 if epoch is not None else 'N/A'} ({current_modality_config['name']}) {mode}: DataLoader or collected labels are empty.")
        if mode == "Val": return 0.0, 0.0 # Validation mode returns 0.0 accuracy, 0.0 F1
        return 0.0, 0.0, 0.0 # Test mode returns 0.0 loss, 0.0 accuracy, 0.0 F1

    # Calculate average BCE loss
    avg_bce_loss = total_bce_loss / len(data_loader)
    # Calculate accuracy (if label list is not empty)
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    # Calculate F1 score (if label list is not empty), use 'binary' because it's binary classification, zero_division handles boundary cases
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) if all_labels else 0.0

    # Print evaluation results
    print(f"Epoch {epoch+1 if epoch is not None else 'N/A'} ({current_modality_config['name']}) {mode} Avg BCE: {avg_bce_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    # If validation mode, return accuracy and F1
    if mode == "Val": return accuracy, f1
    # If test mode, return average loss, accuracy, and F1
    return avg_bce_loss, accuracy, f1

# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- Hyperparameter Configuration ---
    BERT_MODEL_NAME_FOR_MAIN = "bert-base-uncased"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SENTENCE_LSTM_HIDDEN_DIM_CONFIG defined globally
    # SAMPLE_LSTM_HIDDEN_DIM_CONFIG defined globally
    HIER_LSTM_DROPOUT = 0.3

    PROJECTOR_OUTPUT_DIM = 1024

    MAX_BERT_LEN_FOR_PART_DATASET = 512
    TEXT_LSTM_HIDDEN_SIZE_IN_STREAM = 256   # This version of the code still uses this parameter
    ATTENTION_TOKEN_DIM = 32
    NUM_ATTENTION_TOKENS_PER_MODAL = 16
    STREAM_CA_SA_HEADS = 1
    STREAM_DROPOUT_RATE = 0.3

    FINAL_CROSS_ATTENTION_HEADS = 1         # Used for all attention modules in the new final fusion structure
    MLP_HIDDEN_DIM = 256
    NUM_CLASSES = 2

    BATCH_SIZE = 16 # Warning: The new final fusion structure is more complex, may need to reduce this value
    LEARNING_RATE = 8e-5
    NUM_EPOCHS = 8 # It is recommended to increase epochs for actual use
    TEMPERATURE_CONTRASTIVE = 0.5
    CONTRASTIVE_LOSS_WEIGHT = 0.03

    print(f"Using device: {DEVICE}")
    print(f"BERT model used: {BERT_MODEL_NAME_FOR_MAIN}")
    print(f"Hierarchical LSTM: Sentence-level hidden dim {SENTENCE_LSTM_HIDDEN_DIM_CONFIG}, Sample-level hidden dim {SAMPLE_LSTM_HIDDEN_DIM_CONFIG}, Dropout {HIER_LSTM_DROPOUT}")
    print(f"Projector output dimension (Stream processor audio/video input): {PROJECTOR_OUTPUT_DIM}")
    print(f"Max BERT length for context/punchline part: {MAX_BERT_LEN_FOR_PART_DATASET}")
    print(f"Stream processor internal text LSTM hidden size: {TEXT_LSTM_HIDDEN_SIZE_IN_STREAM}") # Keep printing as this param is still in model def
    print(f"Attention token dimension: {ATTENTION_TOKEN_DIM}, Tokens per modality: {NUM_ATTENTION_TOKENS_PER_MODAL}")
    print(f"Stream processor attention heads: {STREAM_CA_SA_HEADS}, Stream processor text FC Dropout rate: {STREAM_DROPOUT_RATE}")
    print(f"Final fusion stage attention heads: {FINAL_CROSS_ATTENTION_HEADS}, MLP hidden dimension: {MLP_HIDDEN_DIM}")
    print(f"Training parameters: Batch size {BATCH_SIZE}, Learning rate {LEARNING_RATE}, Epochs {NUM_EPOCHS}")
    print(f"Contrastive loss: Temperature {TEMPERATURE_CONTRASTIVE}, Weight {CONTRASTIVE_LOSS_WEIGHT}")
    print("\n !!! WARNING: The new final fusion structure (mimicking ASP) is more complex than the original single cross-attention and may significantly increase VRAM consumption and computation time. If you encounter OOM, try drastically reducing BATCH_SIZE first. !!! \n")

    # --- Load Raw Data ---
    print("Loading raw data pickle files...")
    # Ensure paths are correct
    # Example: data_folds_path = "path_to_your_gdrive/Project_CCS2-main/sdk_features/data_folds.pkl"
    # Replace with your actual paths
    # To run locally, you might need to download these files or adjust paths
    # For demonstration, we'll assume files might not exist and add checks or placeholders.
    try:
        data_folds = load_pickle(data_folds_path)
        language_sdk = load_pickle(language_file)
        covarep_sdk = load_pickle(covarep_file)
        openface_sdk = load_pickle(openface_file)
        humor_label_sdk = load_pickle(humor_label_file)
        print("Raw data loading complete.")

        train_ids = data_folds['train']
        dev_ids = data_folds['dev']
        test_ids = data_folds['test']

    except FileNotFoundError:
        print("Error: One or more data files not found. Please check paths and ensure files exist.")
        print("Using placeholder data for demonstration.")
        # Placeholder data for demonstration if files are missing
        train_ids, dev_ids, test_ids = ['h1','h2'], ['h3'], ['h4']
        language_sdk = {
            f'h{i}': {'punchline_sentence': f'Punchline {i}', 'context_sentences': [f'Context sent {i}.1', f'Context sent {i}.2']} for i in range(1, 5)
        }
        covarep_sdk = {
            f'h{i}': {
                'punchline_features': np.random.rand(5, _AUDIO_WORD_DIM_CONST).astype(np.float32) if i % 2 == 0 else [], # Some empty
                'context_features': [np.random.rand(np.random.randint(3,7), _AUDIO_WORD_DIM_CONST).astype(np.float32) for _ in range(2)]
            } for i in range(1,5)
        }
        openface_sdk = {
            f'h{i}': {
                'punchline_features': np.random.rand(5, _VIDEO_WORD_DIM_CONST).astype(np.float32),
                'context_features': [np.random.rand(np.random.randint(3,7), _VIDEO_WORD_DIM_CONST).astype(np.float32) for _ in range(2)]
            } for i in range(1,5)
        }
        humor_label_sdk = {f'h{i}': float(i % 2) for i in range(1,5)}


    print("Extracting features and labels...")
    (train_ps, train_cs, train_cvp_p, train_cvp_c, train_of_p, train_of_c, train_labels) = \
        extract_features_and_labels(train_ids, language_sdk, covarep_sdk, openface_sdk, humor_label_sdk)
    (dev_ps, dev_cs, dev_cvp_p, dev_cvp_c, dev_of_p, dev_of_c, dev_labels) = \
        extract_features_and_labels(dev_ids, language_sdk, covarep_sdk, openface_sdk, humor_label_sdk)
    (test_ps, test_cs, test_cvp_p, test_cvp_c, test_of_p, test_of_c, test_labels) = \
        extract_features_and_labels(test_ids, language_sdk, covarep_sdk, openface_sdk, humor_label_sdk)
    print("Feature and label extraction complete.")

    print("Structuring data for new dataset format (context/punchline split)...")
    train_sample_data_dicts = concatenate_multimodal_data_for_dataset(train_cvp_c, train_of_c, train_cs, train_cvp_p, train_of_p, train_ps)
    dev_sample_data_dicts = concatenate_multimodal_data_for_dataset(dev_cvp_c, dev_of_c, dev_cs, dev_cvp_p, dev_of_p, dev_ps)
    test_sample_data_dicts = concatenate_multimodal_data_for_dataset(test_cvp_c, test_of_c, test_cs, test_cvp_p, test_of_p, test_ps)
    print("Data structuring complete.")

    print("Initializing BERT tokenizer...")
    bert_tokenizer_global = AutoTokenizer.from_pretrained(BERT_MODEL_NAME_FOR_MAIN)
    _bert_temp_model = AutoModel.from_pretrained(BERT_MODEL_NAME_FOR_MAIN)
    BERT_HIDDEN_SIZE_ACTUAL = _bert_temp_model.config.hidden_size
    del _bert_temp_model
    print(f"Actual BERT hidden size: {BERT_HIDDEN_SIZE_ACTUAL}")

    print("Creating CustomFeatureDatasetContextPunchline instances...")
    train_dataset = CustomFeatureDatasetContextPunchline(
        train_sample_data_dicts, train_labels, bert_tokenizer_global,
        max_bert_len_for_part=MAX_BERT_LEN_FOR_PART_DATASET
    )
    dev_dataset = CustomFeatureDatasetContextPunchline(
        dev_sample_data_dicts, dev_labels, bert_tokenizer_global,
        max_bert_len_for_part=MAX_BERT_LEN_FOR_PART_DATASET
    )
    test_dataset = CustomFeatureDatasetContextPunchline(
        test_sample_data_dicts, test_labels, bert_tokenizer_global,
        max_bert_len_for_part=MAX_BERT_LEN_FOR_PART_DATASET
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=custom_collate_fn_context_punchline, drop_last=True if BATCH_SIZE > 1 and len(train_dataset) > BATCH_SIZE else False)
    val_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_context_punchline)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_context_punchline)
    print(f"Dataloaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    modality_configurations = [
        {'name': 'AVT_CtxPl_Contra_FinalFusionMimicASP_NoFinalOrigSA', 'audio': True, 'video': True, 'text': True},
    ]
    all_models_results = {}

    for config_idx, model_config_iter in enumerate(modality_configurations):
        config_name = model_config_iter['name']
        print(f"\n--- Starting processing for model config: {config_name} ---")

        model = ContextPunchlineHumorModelNew(
            bert_model_name_or_path=BERT_MODEL_NAME_FOR_MAIN,
            audio_word_dim=_AUDIO_WORD_DIM_CONST, video_word_dim=_VIDEO_WORD_DIM_CONST,
            sentence_lstm_hidden_dim=SENTENCE_LSTM_HIDDEN_DIM_CONFIG,
            sample_lstm_hidden_dim=SAMPLE_LSTM_HIDDEN_DIM_CONFIG,
            hier_lstm_dropout=HIER_LSTM_DROPOUT,
            projector_output_dim=PROJECTOR_OUTPUT_DIM,
            bert_hidden_size_actual=BERT_HIDDEN_SIZE_ACTUAL,
            max_bert_len_for_lstm=MAX_BERT_LEN_FOR_PART_DATASET, # This is max_bert_len_for_part
            text_lstm_hidden_size_in_stream=TEXT_LSTM_HIDDEN_SIZE_IN_STREAM,
            attention_token_dim=ATTENTION_TOKEN_DIM,
            num_attention_tokens_per_modal=NUM_ATTENTION_TOKENS_PER_MODAL,
            stream_ca_sa_heads=STREAM_CA_SA_HEADS,
            stream_dropout_rate=STREAM_DROPOUT_RATE,
            final_cross_attention_heads=FINAL_CROSS_ATTENTION_HEADS,
            mlp_hidden_dim=MLP_HIDDEN_DIM,
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        print("Freezing BERT parameters in the main model...")
        for param in model.bert_model.parameters():
            param.requires_grad = False
        print("BERT parameters frozen.")

        bce_criterion = nn.CrossEntropyLoss().to(DEVICE)
        contrastive_loss_fn = ContrastiveLossELI5(temperature=TEMPERATURE_CONTRASTIVE).to(DEVICE)
        optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = AdamW(optimizer_params, lr=LEARNING_RATE)
        scheduler = None
        if len(train_loader) > 0 and NUM_EPOCHS > 0:
            num_training_steps_per_epoch = len(train_loader)
            total_training_steps = num_training_steps_per_epoch * NUM_EPOCHS
            num_warmup_steps = int(total_training_steps * 0.1)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=total_training_steps)

        print(f"Starting training for {config_name}... Total {NUM_EPOCHS} epochs.")
        best_val_accuracy_for_config = 0.0
        best_val_f1_at_best_acc = 0.0 # Store F1 at the point of best accuracy
        best_model_state_path = f"best_model_{config_name}.pth"

        if len(train_loader) == 0:
            print(f"Training data loader for {config_name} is empty. Skipping training.")
        else:
            for epoch in range(NUM_EPOCHS):
                train_new_model(model, train_loader, optimizer, scheduler, bce_criterion,
                                contrastive_loss_fn, DEVICE, epoch, NUM_EPOCHS,
                                CONTRASTIVE_LOSS_WEIGHT, model_config_iter, bert_tokenizer_global)
                if len(val_loader) > 0:
                    val_accuracy, val_f1 = validate_or_test_new_model(model, val_loader, bce_criterion, DEVICE,
                                                                    epoch, NUM_EPOCHS, model_config_iter,
                                                                    bert_tokenizer_global, mode="Val")
                    if val_accuracy > best_val_accuracy_for_config:
                        best_val_accuracy_for_config = val_accuracy
                        best_val_f1_at_best_acc = val_f1 # Save F1 at this best accuracy point
                        print(f"Epoch {epoch+1} ({config_name}): New best validation accuracy: {best_val_accuracy_for_config:.4f} (F1: {best_val_f1_at_best_acc:.4f}). Saving model...")
                        torch.save(model.state_dict(), best_model_state_path)
                else:
                    print(f"Epoch {epoch+1} ({config_name}): Validation data loader is empty. Skipping validation.")
            print(f"Training for {config_name} complete. Best validation accuracy for this config: {best_val_accuracy_for_config:.4f} (corresponding F1: {best_val_f1_at_best_acc:.4f})")

        print(f"\nStarting test phase for {config_name}...")
        test_accuracy, test_f1, test_loss = 0.0, 0.0, 0.0 # Initialize
        if len(test_loader) == 0:
            print(f"Test data loader for {config_name} is empty. Skipping test.")
            all_models_results[config_name] = {'val_acc': best_val_accuracy_for_config, 'val_f1': best_val_f1_at_best_acc,
                                               'test_acc': 0.0, 'test_f1':0.0, 'test_loss': 0.0}
        else:
            if os.path.exists(best_model_state_path) and best_val_accuracy_for_config > 0: # Check if model was saved
                print(f"Loading best model state from {best_model_state_path} for testing.")
                model.load_state_dict(torch.load(best_model_state_path, map_location=DEVICE))
            elif best_val_accuracy_for_config == 0 and len(train_loader) > 0 : # Was trained, but no improvement or no val
                print(f"No best validation model saved (or validation accuracy was 0), using model from last training epoch for testing.")
            elif len(train_loader) == 0: # Not trained
                print(f"No training was performed for {config_name}. Testing with initialized model (results might be poor).")

            test_loss, test_accuracy, test_f1 = validate_or_test_new_model(
                model, test_loader, bce_criterion, DEVICE, epoch=None, num_epochs=NUM_EPOCHS, # epoch=None for final test
                current_modality_config=model_config_iter, tokenizer_for_padding=bert_tokenizer_global, mode="Test"
            )
            print(f"Final test results for {config_name} -> Avg BCE Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
            all_models_results[config_name] = {
                'val_acc': best_val_accuracy_for_config, 'val_f1': best_val_f1_at_best_acc,
                'test_acc': test_accuracy, 'test_f1': test_f1, 'test_loss': test_loss,
            }

    print("\n\n--- Final Results Summary for All Model Configurations ---")
    for config_name, results in all_models_results.items():
        print(f"Configuration: {config_name}")
        print(f"  Best Validation Accuracy: {results.get('val_acc', 0.0):.4f} (Corresponding Val F1: {results.get('val_f1', 0.0):.4f})")
        print(f"  Test Set Accuracy: {results.get('test_acc', 0.0):.4f}")
        print(f"  Test Set F1 Score: {results.get('test_f1', 0.0):.4f}")
        print(f"  Test Set Loss: {results.get('test_loss', 0.0):.4f}")
        print("-" * 30)
    print("All operations complete.")