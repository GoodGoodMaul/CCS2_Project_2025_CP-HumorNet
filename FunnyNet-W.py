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
DRIVE_MOUNT_PATH = "/content/gdrive/MyDrive/" # Google Drive mount path
BASE_PROJECT_PATH = os.path.join(DRIVE_MOUNT_PATH, "Project_CCS2-main/sdk_features/") # Base path where feature files are located

data_folds_path = os.path.join(BASE_PROJECT_PATH, "data_folds.pkl")
openface_file = os.path.join(BASE_PROJECT_PATH, "openface_features_sdk.pkl")
covarep_file = os.path.join(BASE_PROJECT_PATH, "covarep_features_sdk.pkl")
language_file = os.path.join(BASE_PROJECT_PATH, "language_sdk.pkl")
humor_label_file = os.path.join(BASE_PROJECT_PATH, "humor_label_sdk.pkl")
# word_embedding_list_file = os.path.join(BASE_PROJECT_PATH, "word_embedding_list.pkl") # Not used directly, kept
_AUDIO_WORD_DIM_CONST = 81 # Define constant for audio word-level feature dimension
_VIDEO_WORD_DIM_CONST = 371 # Define constant for video word-level feature dimension
# New: Define hidden dimension for hierarchical LSTM, will be used by subsequent models
SENTENCE_LSTM_HIDDEN_DIM_CONFIG = 256
SAMPLE_LSTM_HIDDEN_DIM_CONFIG = 512


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'Unable to load data {pickle_file}: {e}')
        raise

def _prepare_feature_for_numpy(feature_data):
    """
    Helper function to prepare feature data to be safely passed to np.array().
    If the input is None, an empty list, or an empty NumPy array, it returns an empty list,
    so that np.array() creates an empty numerical array.
    Otherwise, returns the original data (if it's a list or an existing NumPy array).
    """
    if feature_data is None:
        return []  # None -> empty list for np.array
    if isinstance(feature_data, np.ndarray):
        # If it's a NumPy array, check its size
        if feature_data.size == 0:
            return []  # Empty NumPy array -> empty list for np.array
        return feature_data # Non-empty NumPy array, np.array() will copy it
    if isinstance(feature_data, list):
        if not feature_data: # Empty list
            return []
        return feature_data # Non-empty list

    # For other unexpected types, can print a warning and return an empty list
    # print(f"Warning: Unexpected feature type {type(feature_data)}, treating as empty.")
    return []

# Then use it in the extract_features_and_labels function like this:
def extract_features_and_labels(id_list, language_sdk, covarep_sdk, openface_sdk, humor_label_sdk):
    ps_list, cs_list, cvp_p_list, cvp_c_list, of_p_list, of_c_list = [], [], [], [], [], []
    labels_list = []

    for hid in id_list:
        ps_list.append(language_sdk[hid]['punchline_sentence'])
        cs_list.append(language_sdk[hid]['context_sentences'])

        # COVAREP features
        prepared_punchline_cvp = _prepare_feature_for_numpy(covarep_sdk[hid]['punchline_features'])
        cvp_p_list.append(np.array(prepared_punchline_cvp))

        processed_sents_cvp = []
        for sent_feat in covarep_sdk[hid]['context_features']:
            prepared_sent_cvp = _prepare_feature_for_numpy(sent_feat)
            processed_sents_cvp.append(np.array(prepared_sent_cvp))
        cvp_c_list.append(processed_sents_cvp)

        # OpenFace features
        prepared_punchline_of = _prepare_feature_for_numpy(openface_sdk[hid]['punchline_features'])
        of_p_list.append(np.array(prepared_punchline_of))

        processed_sents_of = []
        for sent_feat in openface_sdk[hid]['context_features']:
            prepared_sent_of = _prepare_feature_for_numpy(sent_feat)
            processed_sents_of.append(np.array(prepared_sent_of))
        of_c_list.append(processed_sents_of)

        labels_list.append(humor_label_sdk[hid])

    return (
        np.array(ps_list, dtype=object),
        np.array(cs_list, dtype=object),
        np.array(cvp_p_list, dtype=object),
        np.array(cvp_c_list, dtype=object),
        np.array(of_p_list, dtype=object),
        np.array(of_c_list, dtype=object),
        np.array(labels_list, dtype=np.float32)
    )

def concatenate_multimodal_data(cvp_c, of_c, cs, cvp_p, of_p, ps):
    num_samples = len(cvp_c)
    if not (len(of_c) == num_samples and \
            len(cs) == num_samples and \
            len(cvp_p) == num_samples and \
            len(of_p) == num_samples and \
            len(ps) == num_samples):
        raise ValueError("All input lists must have the same number of samples.")
    concatenated_audio_features = []
    concatenated_video_features = []
    concatenated_text_features = []
    for i in range(num_samples):
        sample_cvp_c_sentence_features = list(cvp_c[i])
        sample_of_c_sentence_features = list(of_c[i])
        sample_cs_sentence_texts = list(cs[i])
        punchline_audio_features = cvp_p[i]
        punchline_video_features = of_p[i]
        punchline_text = ps[i]
        current_sample_audio = sample_cvp_c_sentence_features.copy()
        current_sample_audio.append(punchline_audio_features)
        concatenated_audio_features.append(current_sample_audio)
        current_sample_video = sample_of_c_sentence_features.copy()
        current_sample_video.append(punchline_video_features)
        concatenated_video_features.append(current_sample_video)
        current_sample_text = sample_cs_sentence_texts.copy()
        current_sample_text.append(punchline_text)
        concatenated_text_features.append(current_sample_text)
    return concatenated_audio_features, concatenated_video_features, concatenated_text_features


# --- Modify CustomFeatureDataset ---
# It is now only responsible for providing raw (or near-raw) feature data and text data
class CustomFeatureDataset(Dataset):
    def __init__(self, list_of_audio_sample_data, list_of_video_sample_data,
                 list_of_text_sentence_lists_per_sample, list_of_labels,
                 bert_tokenizer, max_bert_len=512):

        if not (len(list_of_audio_sample_data) == len(list_of_video_sample_data) == \
                len(list_of_text_sentence_lists_per_sample) == len(list_of_labels)):
            raise ValueError("All input data lists must have the same length.")

        self.list_of_audio_sample_data = list_of_audio_sample_data
        self.list_of_video_sample_data = list_of_video_sample_data
        self.list_of_text_sentence_lists_per_sample = list_of_text_sentence_lists_per_sample
        self.list_of_labels = torch.tensor(list_of_labels, dtype=torch.long) # Ensure labels are Long type
        self.tokenizer = bert_tokenizer
        self.max_bert_len = max_bert_len

    def __len__(self):
        return len(self.list_of_labels)

    def __getitem__(self, index):
        # Returns a list of sentences for each sample, where each sentence is word features (num_words, feature_dim)
        # collate_fn will handle this variable-length data
        audio_sample_sentences_raw = [torch.as_tensor(sent_feat, dtype=torch.float32)
                                      for sent_feat in self.list_of_audio_sample_data[index]
                                      if isinstance(sent_feat, np.ndarray) and sent_feat.ndim == 2 and sent_feat.shape[0] > 0]
        # If the processed list is empty (e.g., all sentences are empty or incorrectly formatted), provide a placeholder
        if not audio_sample_sentences_raw:
             audio_sample_sentences_raw = [torch.zeros((1, _AUDIO_WORD_DIM_CONST), dtype=torch.float32)]


        video_sample_sentences_raw = [torch.as_tensor(sent_feat, dtype=torch.float32)
                                      for sent_feat in self.list_of_video_sample_data[index]
                                      if isinstance(sent_feat, np.ndarray) and sent_feat.ndim == 2 and sent_feat.shape[0] > 0]
        if not video_sample_sentences_raw:
            video_sample_sentences_raw = [torch.zeros((1, _VIDEO_WORD_DIM_CONST), dtype=torch.float32)]


        text_sentences_for_this_sample = self.list_of_text_sentence_lists_per_sample[index]
        label = self.list_of_labels[index]

        if not text_sentences_for_this_sample:
            concatenated_text_for_bert = ""
        else:
            if not all(isinstance(s, str) for s in text_sentences_for_this_sample):
                concatenated_text_for_bert = ""
            else:
                concatenated_text_for_bert = " ".join(text_sentences_for_this_sample)

        bert_inputs = self.tokenizer(
            concatenated_text_for_bert, add_special_tokens=True, return_attention_mask=True,
            max_length=self.max_bert_len, padding='max_length', truncation=True, return_tensors="pt",
        )
        input_ids = bert_inputs["input_ids"].squeeze(0)
        attention_mask = bert_inputs["attention_mask"].squeeze(0)

        return audio_sample_sentences_raw, video_sample_sentences_raw, input_ids, attention_mask, label

# --- New: Custom Collate Function ---
def custom_collate_fn(batch):
    audio_data_raw, video_data_raw, text_ids_list, text_masks_list, labels_list = zip(*batch)

    # Process text and labels (they are already tensors)
    batched_text_ids = torch.stack(text_ids_list)
    batched_text_masks = torch.stack(text_masks_list)
    batched_labels = torch.stack(labels_list)

    # Process audio data (list of lists of tensors)
    # Goal: Create a tensor of shape (B, S_max, W_max, D_audio) and length information
    # S_max: max number of sentences in a sample in the batch, W_max: max number of words in a sentence in the batch

    # Audio processing
    audio_sample_lengths = [len(sample) for sample in audio_data_raw]
    max_sents_audio = max(audio_sample_lengths) if audio_sample_lengths else 0

    # Get word counts for each sentence and find the maximum word count
    audio_sentence_word_counts_flat = []
    for sample in audio_data_raw:
        for sentence_tensor in sample:
            audio_sentence_word_counts_flat.append(sentence_tensor.shape[0])
    max_words_audio = max(audio_sentence_word_counts_flat) if audio_sentence_word_counts_flat else 0

    # Create padded audio tensor and length tensors
    # padded_features: (batch_size, max_sents, max_words, feat_dim)
    # sentence_lengths: (batch_size, max_sents) - records actual word count for each sentence
    # sample_lengths: (batch_size) - records actual sentence count for each sample (already obtained as audio_sample_lengths)
    padded_audio_features = torch.zeros(len(audio_data_raw), max_sents_audio, max_words_audio, _AUDIO_WORD_DIM_CONST)
    audio_sentence_lengths = torch.zeros(len(audio_data_raw), max_sents_audio, dtype=torch.long)

    for i, sample in enumerate(audio_data_raw):
        for j, sentence_tensor in enumerate(sample):
            num_words = sentence_tensor.shape[0]
            if num_words > 0: # Only pad if there are words
                padded_audio_features[i, j, :num_words, :] = sentence_tensor
                audio_sentence_lengths[i, j] = num_words

    # Video processing (similar to audio)
    video_sample_lengths = [len(sample) for sample in video_data_raw]
    max_sents_video = max(video_sample_lengths) if video_sample_lengths else 0
    video_sentence_word_counts_flat = []
    for sample in video_data_raw:
        for sentence_tensor in sample:
            video_sentence_word_counts_flat.append(sentence_tensor.shape[0])
    max_words_video = max(video_sentence_word_counts_flat) if video_sentence_word_counts_flat else 0

    padded_video_features = torch.zeros(len(video_data_raw), max_sents_video, max_words_video, _VIDEO_WORD_DIM_CONST)
    video_sentence_lengths = torch.zeros(len(video_data_raw), max_sents_video, dtype=torch.long)

    for i, sample in enumerate(video_data_raw):
        for j, sentence_tensor in enumerate(sample):
            num_words = sentence_tensor.shape[0]
            if num_words > 0:
                padded_video_features[i, j, :num_words, :] = sentence_tensor
                video_sentence_lengths[i, j] = num_words

    return (padded_audio_features, torch.tensor(audio_sample_lengths, dtype=torch.long), audio_sentence_lengths,
            padded_video_features, torch.tensor(video_sample_lengths, dtype=torch.long), video_sentence_lengths,
            batched_text_ids, batched_text_masks, batched_labels)


# --- New: Hierarchical LSTM Aggregator (Trainable) ---
class HierarchicalLSTMAggregator(nn.Module):
    def __init__(self, word_dim, sentence_lstm_hidden_dim, sample_lstm_hidden_dim,
                 sentence_lstm_layers=1, sample_lstm_layers=1, dropout_rate=0.3): # Added dropout
        super().__init__()
        self.sentence_lstm_hidden_dim = sentence_lstm_hidden_dim
        self.sample_lstm_hidden_dim = sample_lstm_hidden_dim

        self.sentence_lstm = nn.LSTM(word_dim, sentence_lstm_hidden_dim,
                                     num_layers=sentence_lstm_layers, batch_first=True,
                                     bidirectional=False) # Can be set to True if needed

        # If sentence_lstm is bidirectional, input dimension for sample_lstm needs to be multiplied by 2
        sample_lstm_input_dim = sentence_lstm_hidden_dim * (2 if self.sentence_lstm.bidirectional else 1)

        self.sample_lstm = nn.LSTM(sample_lstm_input_dim, sample_lstm_hidden_dim,
                                   num_layers=sample_lstm_layers, batch_first=True,
                                   bidirectional=False) # Can be set to True if needed
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, features, sample_lengths, sentence_lengths):
        # features: (batch_size, max_sents, max_words, word_dim)
        # sample_lengths: (batch_size) - actual number of sentences per sample
        # sentence_lengths: (batch_size, max_sents) - actual number of words per sentence

        batch_size, max_sents, max_words, _ = features.shape

        # 1. Process sentence level
        # Merge batch and sents dimensions to pass through LSTM at once
        # (B, S, W, D) -> (B*S, W, D)
        sents_features = features.view(batch_size * max_sents, max_words, -1)
        # (B, S) -> (B*S)
        sents_word_lengths = sentence_lengths.view(batch_size * max_sents)

        # Filter out sentences with length 0 to avoid pack_padded_sequence error
        valid_sents_indices = sents_word_lengths > 0
        if not torch.any(valid_sents_indices): # If all sentences are empty
            # Return a zero tensor with shape matching sample_lstm output
            final_output_dim = self.sample_lstm_hidden_dim * (2 if self.sample_lstm.bidirectional else 1)
            return torch.zeros(batch_size, final_output_dim, device=features.device)

        sents_features_packed = sents_features[valid_sents_indices]
        sents_word_lengths_packed = sents_word_lengths[valid_sents_indices]

        # Pack variable length sequences
        packed_sents_input = pack_padded_sequence(sents_features_packed, sents_word_lengths_packed.cpu(),
                                                  batch_first=True, enforce_sorted=False)

        # Pass through sentence LSTM
        # h_n_sent: (num_layers * num_directions, B*S_valid, sent_hidden_dim)
        _, (h_n_sent, _) = self.sentence_lstm(packed_sents_input)

        # Get hidden state of the last time step (for unidirectional LSTM, take the last layer)
        # (B*S_valid, sent_hidden_dim)
        if self.sentence_lstm.bidirectional:
            # Concatenate hidden states of the last time step from both directions of bidirectional LSTM
            sentence_embeddings_valid = torch.cat((h_n_sent[-2,:,:], h_n_sent[-1,:,:]), dim=1)
        else:
            sentence_embeddings_valid = h_n_sent[-1,:,:]

        sentence_embeddings_valid = self.dropout(sentence_embeddings_valid)

        # Place valid sentence embeddings back to their original positions, use zero vectors for empty sentences
        # Output shape: (B*S, sent_hidden_dim_actual)
        sent_hidden_dim_actual = sentence_embeddings_valid.shape[-1]
        all_sentence_embeddings = torch.zeros(batch_size * max_sents, sent_hidden_dim_actual, device=features.device)
        all_sentence_embeddings[valid_sents_indices] = sentence_embeddings_valid

        # (B*S, H_sent) -> (B, S, H_sent)
        sample_features = all_sentence_embeddings.view(batch_size, max_sents, sent_hidden_dim_actual)

        # 2. Process sample level
        # Pack variable length sequences (based on actual number of sentences per sample, sample_lengths)
        # Filter out samples with length 0
        valid_sample_indices = sample_lengths > 0
        if not torch.any(valid_sample_indices):
            final_output_dim = self.sample_lstm_hidden_dim * (2 if self.sample_lstm.bidirectional else 1)
            return torch.zeros(batch_size, final_output_dim, device=features.device)

        sample_features_packed_input = sample_features[valid_sample_indices]
        sample_lengths_packed = sample_lengths[valid_sample_indices]

        packed_sample_input = pack_padded_sequence(sample_features_packed_input, sample_lengths_packed.cpu(),
                                                   batch_first=True, enforce_sorted=False)

        # Pass through sample LSTM
        # h_n_sample: (num_layers * num_directions, B_valid, sample_hidden_dim)
        _, (h_n_sample, _) = self.sample_lstm(packed_sample_input)

        # Get hidden state of the last time step
        # (B_valid, sample_hidden_dim)
        if self.sample_lstm.bidirectional:
            sample_embeddings_valid = torch.cat((h_n_sample[-2,:,:], h_n_sample[-1,:,:]), dim=1)
        else:
            sample_embeddings_valid = h_n_sample[-1,:,:]

        sample_embeddings_valid = self.dropout(sample_embeddings_valid)

        # Place valid sample embeddings back to their original positions
        final_output_dim = sample_embeddings_valid.shape[-1]
        final_sample_embeddings = torch.zeros(batch_size, final_output_dim, device=features.device)
        final_sample_embeddings[valid_sample_indices] = sample_embeddings_valid

        return final_sample_embeddings


class linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.layer1(x) * self.layer2(x)
        return x

class CA_SA(nn.Module):
    def __init__(self, dim=32):
        super(CA_SA, self).__init__()
        self.dim = dim
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)
        self.Q = nn.Linear(dim, dim, bias=False)
        self.attend = nn.Softmax(dim = -1)
    def forward(self, feat1, feat2):
        K = self.K(feat2)
        V = self.V(feat2)
        Q = self.Q(feat1)
        dots = torch.bmm(Q, K.permute(0, 2, 1))
        attn = self.attend(dots)
        out = torch.bmm(attn, V)
        return out

class MultimodalFusionLSTMHead(nn.Module):
    def __init__(self, projected_audio_video_dim=1024, # This is the dimension from the trainable projection layer
                 bert_hidden_size=768,
                 max_bert_len=512,
                 lstm_hidden_size=256, # Hidden size of text LSTM
                 attention_token_dim=32, num_attention_tokens_per_modal=16, num_classes=2,
                 active_modalities=('audio', 'video', 'text')):
        super().__init__()
        self.n = num_attention_tokens_per_modal
        self.attention_token_dim = attention_token_dim
        self.max_bert_len = max_bert_len
        self.active_modalities = active_modalities
        self.expected_feature_dim_for_attention = self.n * self.attention_token_dim # 512

        # Audio feature processor: input projected_audio_video_dim (e.g., 1024), output expected_feature_dim_for_attention (e.g., 512)
        self.audio_feat_processor = nn.Sequential(
            linear(projected_audio_video_dim, 1024),
            linear(1024, self.expected_feature_dim_for_attention),
            nn.LayerNorm(self.expected_feature_dim_for_attention)
        )
        self.vision_feat_processor = nn.Sequential(
            linear(projected_audio_video_dim, 1024),
            linear(1024, self.expected_feature_dim_for_attention),
            nn.LayerNorm(self.expected_feature_dim_for_attention)
        )
        self.text_lstm_processor = nn.LSTM(bert_hidden_size, lstm_hidden_size, batch_first=True)
        self.text_fc_processor = nn.Sequential(
            nn.Dropout(0.5),
            linear(lstm_hidden_size * self.max_bert_len, 1024), # Flatten LSTM output
            linear(1024, self.expected_feature_dim_for_attention),
            nn.LayerNorm(self.expected_feature_dim_for_attention)
        )
        self.ZA = CA_SA(dim=attention_token_dim)
        self.ZV = CA_SA(dim=attention_token_dim)
        self.ZT = CA_SA(dim=attention_token_dim)
        self.SA = CA_SA(dim=attention_token_dim)
        self.pre = nn.Sequential(
            linear(self.expected_feature_dim_for_attention, num_classes)
        )

    def forward(self, audio_input_projected, vision_input_projected, text_sequence_input_bert):
        b = audio_input_projected.shape[0] if audio_input_projected.nelement() > 0 else (vision_input_projected.shape[0] if vision_input_projected.nelement() > 0 else text_sequence_input_bert.shape[0])
        device = audio_input_projected.device if audio_input_projected.nelement() > 0 else (vision_input_projected.device if vision_input_projected.nelement() > 0 else text_sequence_input_bert.device)


        if 'audio' in self.active_modalities and audio_input_projected.nelement() > 0 :
            audio_f = self.audio_feat_processor(audio_input_projected)
        else:
            audio_f = torch.zeros(b, self.expected_feature_dim_for_attention, device=device)

        if 'video' in self.active_modalities and vision_input_projected.nelement() > 0:
            vis_f = self.vision_feat_processor(vision_input_projected)
        else:
            vis_f = torch.zeros(b, self.expected_feature_dim_for_attention, device=device)

        if 'text' in self.active_modalities and text_sequence_input_bert.nelement() > 0 :
            lstm_output, (h_n, c_n) = self.text_lstm_processor(text_sequence_input_bert)
            if lstm_output.shape[1] != self.max_bert_len:
                if lstm_output.nelement() == 0 and self.max_bert_len > 0 :
                    lstm_output = torch.zeros(b, self.max_bert_len, lstm_output.shape[2] if lstm_output.ndim > 2 else self.text_lstm_processor.hidden_size, device=device)
                elif lstm_output.shape[1] != self.max_bert_len :
                    raise ValueError(f"LSTM output sequence length {lstm_output.shape[1]} does not match expected max_bert_len {self.max_bert_len}")
            text_f_processed = lstm_output.reshape(b, -1)
            text_f = self.text_fc_processor(text_f_processed)
        else:
            text_f = torch.zeros(b, self.expected_feature_dim_for_attention, device=device)

        audio_for_attn = audio_f.view(b, self.n, self.attention_token_dim)
        vis_for_attn = vis_f.view(b, self.n, self.attention_token_dim)
        sub_for_attn = text_f.view(b, self.n, self.attention_token_dim)
        active_processed_features_for_query = []
        if 'audio' in self.active_modalities:
            active_processed_features_for_query.append(F.normalize(audio_f, dim=1))
        if 'video' in self.active_modalities:
            active_processed_features_for_query.append(F.normalize(vis_f, dim=1))
        if 'text' in self.active_modalities:
            active_processed_features_for_query.append(F.normalize(text_f, dim=1))

        if not active_processed_features_for_query:
            final_feat_to_classify = torch.zeros(b, self.expected_feature_dim_for_attention, device=device)
        else:
            z_feat_concatenated = torch.cat(active_processed_features_for_query, dim=1)
            num_active_modalities_for_query = len(active_processed_features_for_query)
            z_feat_for_query = z_feat_concatenated.view(b, num_active_modalities_for_query * self.n, self.attention_token_dim)
            feat_ZA_res = self.ZA(z_feat_for_query, audio_for_attn) if 'audio' in self.active_modalities else torch.zeros_like(z_feat_for_query)
            feat_ZV_res = self.ZV(z_feat_for_query, vis_for_attn)   if 'video' in self.active_modalities else torch.zeros_like(z_feat_for_query)
            feat_ZT_res = self.ZT(z_feat_for_query, sub_for_attn)   if 'text'  in self.active_modalities else torch.zeros_like(z_feat_for_query)
            feat_after_ca = feat_ZA_res + feat_ZV_res + feat_ZT_res
            feat_after_sa = self.SA(feat_after_ca, feat_after_ca) + feat_after_ca
            if num_active_modalities_for_query > 0:
                chunks = feat_after_sa.chunk(num_active_modalities_for_query, dim=1)
                final_feat_to_classify_list = [chunk.reshape(b, -1) for chunk in chunks]
                final_feat_to_classify = torch.stack(final_feat_to_classify_list).sum(dim=0)
                if final_feat_to_classify.shape[1] != self.expected_feature_dim_for_attention:
                        # This case should theoretically not happen, as reshape(-1) will become n * attention_token_dim
                        # If it happens, a linear layer might be needed to adjust dimensions
                        # print(f"Warning: Dimension mismatch before final classifier. Expected {self.expected_feature_dim_for_attention}, got {final_feat_to_classify.shape[1]}")
                        # As a temporary solution, if dimensions don't match, a zero vector could be returned or an error raised, but this indicates preceding logic might need adjustment
                        # raise ValueError("Dimension mismatch")
                        pass # Assume dimensions are correct
            else:
                final_feat_to_classify = torch.zeros(b, self.expected_feature_dim_for_attention, device=device)
        prob_logits = self.pre(final_feat_to_classify)
        return audio_f, vis_f, text_f, prob_logits

class ContrastiveLossELI5(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, emb_i, emb_j):
        batch_size = emb_i.shape[0]
        if batch_size <= 1: # Contrastive loss requires at least 2 samples
            return torch.tensor(0.0, device=emb_i.device, requires_grad=True)
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        labels_i_to_j = torch.arange(batch_size, device=emb_i.device) + batch_size
        labels_j_to_i = torch.arange(batch_size, device=emb_i.device)

        # Mask out self-similarity on the diagonal, avoid calculating similarity of a sample with itself as a positive sample
        # For NT-Xent loss, the diagonal (self vs self) usually doesn't directly participate in loss calculation,
        # because positive samples are corresponding samples from different augmented views.
        # The logits for CrossEntropyLoss are similarity scores, and labels point to the position of positive samples.
        # In similarity_matrix, (i, i+bs) and (i+bs, i) are positive pairs.
        loss_i = self.criterion(similarity_matrix[:batch_size], labels_i_to_j)
        loss_j = self.criterion(similarity_matrix[batch_size:], labels_j_to_i)
        return (loss_i + loss_j) / 2.0

# --- Modify train and validate/test functions to include new hierarchical LSTM modules ---
def train(audio_hier_lstm, video_hier_lstm, # New hierarchical LSTM modules
          audio_projector, video_projector, text_feature_extractor,
          clf_head, data_loader, optimizer, scheduler,bce_criterion, contrastive_loss_fn, device, epoch, num_epochs, contrastive_loss_weight,
          current_modality_config, tokenizer_for_padding):

    audio_hier_lstm.train() # Set to training mode
    video_hier_lstm.train() # Set to training mode
    audio_projector.train()
    video_projector.train()
    text_feature_extractor.train() # Although usually frozen, keep mode consistent
    clf_head.train()

    total_bce_loss = 0
    total_simclr_loss = 0
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train {current_modality_config['name']}]", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Get data from collate_fn
        (padded_audio_features, audio_sample_lengths, audio_sentence_lengths,
         padded_video_features, video_sample_lengths, video_sentence_lengths,
         bert_input_ids, bert_attention_mask, label_data) = batch

        current_batch_size = padded_audio_features.shape[0]
        if current_batch_size == 0:
            continue

        # Process data according to modality configuration (zero out)
        if not current_modality_config.get('audio', False):
            padded_audio_features = torch.zeros_like(padded_audio_features)
            audio_sample_lengths = torch.zeros_like(audio_sample_lengths)
            audio_sentence_lengths = torch.zeros_like(audio_sentence_lengths)
        if not current_modality_config.get('video', False):
            padded_video_features = torch.zeros_like(padded_video_features)
            video_sample_lengths = torch.zeros_like(video_sample_lengths)
            video_sentence_lengths = torch.zeros_like(video_sentence_lengths)
        if not current_modality_config.get('text', False):
            pad_token_id = tokenizer_for_padding.pad_token_id if tokenizer_for_padding.pad_token_id is not None else 0
            bert_input_ids = torch.full_like(bert_input_ids, pad_token_id)
            bert_attention_mask = torch.zeros_like(bert_attention_mask)

        # Move data to device
        padded_audio_features = padded_audio_features.to(device)
        audio_sample_lengths = audio_sample_lengths.to(device)
        audio_sentence_lengths = audio_sentence_lengths.to(device)
        padded_video_features = padded_video_features.to(device)
        video_sample_lengths = video_sample_lengths.to(device)
        video_sentence_lengths = video_sentence_lengths.to(device)
        bert_input_ids = bert_input_ids.to(device)
        bert_attention_mask = bert_attention_mask.to(device)
        label_data = label_data.to(device).long()

        optimizer.zero_grad()

        # 1. Get audio/video sample vectors through hierarchical LSTM
        # Only calculate if modality is active and there is actual content, otherwise use zero vectors
        audio_sample_vectors = torch.zeros(current_batch_size, SAMPLE_LSTM_HIDDEN_DIM_CONFIG, device=device) # Assume fixed dimension
        if current_modality_config.get('audio', False) and torch.any(audio_sample_lengths > 0):
            audio_sample_vectors = audio_hier_lstm(padded_audio_features, audio_sample_lengths, audio_sentence_lengths)

        video_sample_vectors = torch.zeros(current_batch_size, SAMPLE_LSTM_HIDDEN_DIM_CONFIG, device=device) # Assume fixed dimension
        if current_modality_config.get('video', False) and torch.any(video_sample_lengths > 0):
            video_sample_vectors = video_hier_lstm(padded_video_features, video_sample_lengths, video_sentence_lengths)

        # 2. Through projection layer
        projected_audio = audio_projector(audio_sample_vectors)
        projected_video = video_projector(video_sample_vectors)

        # 3. BERT Features
        # BERT parameters are frozen, so no gradients will flow back here if requires_grad=False for BERT
        bert_outputs = text_feature_extractor(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
        text_sequence_features = bert_outputs.last_hidden_state.to(torch.float32)

        # 4. Fusion head
        feat_audio, feat_vision, feat_text, prob_logits = clf_head(projected_audio, projected_video, text_sequence_features)

        bce_loss = bce_criterion(prob_logits, label_data)
        calculated_simclr_loss_sum = torch.tensor(0.0, device=device)
        audio_active = current_modality_config.get('audio', False) and feat_audio.nelement() > 0 and torch.any(audio_sample_lengths > 0)
        video_active = current_modality_config.get('video', False) and feat_vision.nelement() > 0 and torch.any(video_sample_lengths > 0)
        text_active = current_modality_config.get('text', False) and feat_text.nelement() > 0

        if current_batch_size > 1: # Contrastive loss requires at least 2 samples
            if audio_active and video_active:
                calculated_simclr_loss_sum += contrastive_loss_fn(feat_audio, feat_vision)
            if audio_active and text_active:
                calculated_simclr_loss_sum += contrastive_loss_fn(feat_audio, feat_text)
            if video_active and text_active:
                calculated_simclr_loss_sum += contrastive_loss_fn(feat_vision, feat_text)

        current_loss = bce_loss + contrastive_loss_weight * calculated_simclr_loss_sum
        current_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_bce_loss += bce_loss.item()
        total_simclr_loss += calculated_simclr_loss_sum.item()
        total_loss += current_loss.item()
        progress_bar.set_postfix(loss=f"{current_loss.item():.4f}", bce=f"{bce_loss.item():.4f}", simclr=f"{calculated_simclr_loss_sum.item():.4f}")

    if not progress_bar.iterable or len(progress_bar.iterable) == 0:
        print(f"Epoch {epoch+1} ({current_modality_config['name']}) Train: DataLoader is empty or was not processed.")
        return
    if len(data_loader) > 0:
        avg_loss = total_loss / len(data_loader)
        avg_bce_loss = total_bce_loss / len(data_loader)
        avg_simclr_loss = total_simclr_loss / len(data_loader)
        print(f"Epoch {epoch+1} ({current_modality_config['name']}) Train Avg Loss: {avg_loss:.4f}, BCE: {avg_bce_loss:.4f}, SimCLR: {avg_simclr_loss:.4f}")
    else:
        print(f"Epoch {epoch+1} ({current_modality_config['name']}) Train: DataLoader was empty. No average losses to report.")


from sklearn.metrics import accuracy_score, f1_score # Ensure this import is present at the top of your file
from tqdm import tqdm # Ensure this import is present
import torch # Ensure this import is present

# Constants used in the function, ensure they are defined in your script scope
# SAMPLE_LSTM_HIDDEN_DIM_CONFIG = 512 # Example value

def validate_or_test(audio_hier_lstm, video_hier_lstm,
                     audio_projector, video_projector, text_feature_extractor,
                     clf_head, data_loader, bce_criterion, device, epoch, num_epochs,
                     current_modality_config, tokenizer_for_padding, mode="Val"):

    audio_hier_lstm.eval()
    video_hier_lstm.eval()
    audio_projector.eval()
    video_projector.eval()
    text_feature_extractor.eval()
    clf_head.eval()

    total_bce_loss = 0
    all_preds = []
    all_labels = []
    desc = f"Epoch {epoch+1}/{num_epochs} [{mode} {current_modality_config['name']}]"
    if mode == "Test" and epoch is None : # For final test run not tied to an epoch
        desc = f"Final Test [{current_modality_config['name']}]"
    elif mode == "Test": # For test run after a specific epoch
        desc = f"Test after Epoch {epoch+1} [{current_modality_config['name']}]"


    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=desc, leave=False)):
            (padded_audio_features, audio_sample_lengths, audio_sentence_lengths,
             padded_video_features, video_sample_lengths, video_sentence_lengths,
             bert_input_ids, bert_attention_mask, label_data) = batch

            current_batch_size = padded_audio_features.shape[0]
            if current_batch_size == 0:
                continue

            if not current_modality_config.get('audio', False):
                padded_audio_features.zero_()
                audio_sample_lengths.zero_()
                audio_sentence_lengths.zero_()
            if not current_modality_config.get('video', False):
                padded_video_features.zero_()
                video_sample_lengths.zero_()
                video_sentence_lengths.zero_()
            if not current_modality_config.get('text', False):
                pad_token_id = tokenizer_for_padding.pad_token_id if tokenizer_for_padding.pad_token_id is not None else 0
                bert_input_ids.fill_(pad_token_id)
                bert_attention_mask.zero_()

            padded_audio_features = padded_audio_features.to(device)
            audio_sample_lengths = audio_sample_lengths.to(device)
            audio_sentence_lengths = audio_sentence_lengths.to(device)
            padded_video_features = padded_video_features.to(device)
            video_sample_lengths = video_sample_lengths.to(device)
            video_sentence_lengths = video_sentence_lengths.to(device)
            bert_input_ids = bert_input_ids.to(device)
            bert_attention_mask = bert_attention_mask.to(device)
            label_data = label_data.to(device).long()

            # Assuming SAMPLE_LSTM_HIDDEN_DIM_CONFIG is globally defined or passed correctly
            # And HierarchicalLSTMAggregator has sample_lstm_bidirectional attribute
            audio_sample_vec_dim = SAMPLE_LSTM_HIDDEN_DIM_CONFIG * (2 if hasattr(audio_hier_lstm, 'sample_lstm_bidirectional') and audio_hier_lstm.sample_lstm_bidirectional else 1)
            video_sample_vec_dim = SAMPLE_LSTM_HIDDEN_DIM_CONFIG * (2 if hasattr(video_hier_lstm, 'sample_lstm_bidirectional') and video_hier_lstm.sample_lstm_bidirectional else 1)


            audio_sample_vectors = torch.zeros(current_batch_size, audio_sample_vec_dim, device=device)
            if current_modality_config.get('audio', False) and torch.any(audio_sample_lengths > 0):
                audio_sample_vectors = audio_hier_lstm(padded_audio_features, audio_sample_lengths, audio_sentence_lengths)

            video_sample_vectors = torch.zeros(current_batch_size, video_sample_vec_dim, device=device)
            if current_modality_config.get('video', False) and torch.any(video_sample_lengths > 0):
                video_sample_vectors = video_hier_lstm(padded_video_features, video_sample_lengths, video_sentence_lengths)

            projected_audio = audio_projector(audio_sample_vectors)
            projected_video = video_projector(video_sample_vectors)
            bert_outputs = text_feature_extractor(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
            text_sequence_features = bert_outputs.last_hidden_state.to(torch.float32)
            _, _, _, prob_logits = clf_head(projected_audio, projected_video, text_sequence_features)

            if prob_logits.shape[0] == 0: # If clf_head returns empty logits for some reason
                continue

            bce_loss = bce_criterion(prob_logits, label_data)
            total_bce_loss += bce_loss.item()
            preds = torch.argmax(prob_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_data.cpu().numpy())

    epoch_display = epoch + 1 if epoch is not None else 'N/A'

    if len(data_loader) == 0 or not all_labels: # Check if all_labels is empty
        print(f"Epoch {epoch_display} ({current_modality_config['name']}) {mode}: DataLoader or collected labels are empty. Cannot compute metrics.")
        if mode == "Val":
            return 0.0  # Accuracy
        else: # Test mode
            return 0.0, 0.0, 0.0  # BCE Loss, Accuracy, F1 Score

    avg_bce_loss = total_bce_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    # Calculate F1 score, typically for binary classification, use average='binary'.
    # zero_division=0 handles cases where precision or recall is 0 for a class, resulting in F1=0.
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    print(f"Epoch {epoch_display} ({current_modality_config['name']}) {mode} Avg BCE: {avg_bce_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    if mode == "Val":
        return accuracy # For validation, return only accuracy as per original structure
    else: # Test mode
        return avg_bce_loss, accuracy, f1




data_folds = load_pickle(data_folds_path)
language_sdk = load_pickle(language_file)
covarep_sdk = load_pickle(covarep_file)
openface_sdk = load_pickle(openface_file)
humor_label_sdk = load_pickle(humor_label_file)
print("Raw data loading complete.")

train_ids = data_folds['train']
dev_ids = data_folds['dev']
test_ids = data_folds['test']

print("Starting to extract features and labels...")
(train_ps_orig, train_cs_orig, train_cvp_p_orig, train_cvp_c_orig,
  train_of_p_orig, train_of_c_orig, train_labels) = extract_features_and_labels(
      train_ids, language_sdk, covarep_sdk, openface_sdk, humor_label_sdk)
(dev_ps_orig, dev_cs_orig, dev_cvp_p_orig, dev_cvp_c_orig,
   dev_of_p_orig, dev_of_c_orig, dev_labels) = extract_features_and_labels(
     dev_ids, language_sdk, covarep_sdk, openface_sdk, humor_label_sdk)
(test_ps_orig, test_cs_orig, test_cvp_p_orig, test_cvp_c_orig,
 test_of_p_orig, test_of_c_orig, test_labels) = extract_features_and_labels(
      test_ids, language_sdk, covarep_sdk, openface_sdk, humor_label_sdk)
print("Feature and label extraction complete.")

print("Starting to concatenate multimodal data...")
concatenated_train_audio, concatenated_train_video, concatenated_train_text = concatenate_multimodal_data(
        train_cvp_c_orig, train_of_c_orig, train_cs_orig, train_cvp_p_orig, train_of_p_orig, train_ps_orig)
concatenated_dev_audio, concatenated_dev_video, concatenated_dev_text = concatenate_multimodal_data(
        dev_cvp_c_orig, dev_of_c_orig, dev_cs_orig, dev_cvp_p_orig, dev_of_p_orig, dev_ps_orig)
concatenated_test_audio, concatenated_test_video, concatenated_test_text = concatenate_multimodal_data(
        test_cvp_c_orig, test_of_c_orig, test_cs_orig, test_cvp_p_orig, test_of_p_orig, test_ps_orig)
print("Multimodal data concatenation complete.")

if __name__ == "__main__":
    BERT_MODEL_NAME_FOR_MAIN = "bert-base-uncased"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_BERT_LEN = 512
    # LSTM_HIDDEN_SIZE is now used for text LSTM, audio/video hierarchical LSTMs have their own config
    TEXT_LSTM_HIDDEN_SIZE = 256
    # Audio/video projection layer input dimension, now is the output dimension of HierarchicalLSTMAggregator
    HIER_LSTM_OUTPUT_DIM = SAMPLE_LSTM_HIDDEN_DIM_CONFIG # Use the constant defined above: 512
    PROJECTOR_OUTPUT_DIM = 1024 # Projection layer output / fusion head input dimension (unchanged)

    ATTENTION_TOKEN_DIM = 32
    NUM_ATTENTION_TOKENS_PER_MODAL = 16
    NUM_CLASSES = 2
    BATCH_SIZE = 16 # Note: if memory is insufficient, BATCH_SIZE can be reduced
    LEARNING_RATE = 5e-5
    BERT_LEARNING_RATE = 5e-5
    NUM_EPOCHS = 4
    TEMPERATURE_CONTRASTIVE = 0.5
    CONTRASTIVE_LOSS_WEIGHT = 0.03
    HIER_LSTM_DROPOUT = 0.3 # Dropout for hierarchical LSTM

    print(f"Using device: {DEVICE}")
    print(f"MAX_BERT_LEN set to: {MAX_BERT_LEN}")
    print(f"Text LSTM_HIDDEN_SIZE set to: {TEXT_LSTM_HIDDEN_SIZE}")
    print(f"Hierarchical LSTM output dim (projector input): {HIER_LSTM_OUTPUT_DIM}")
    print(f"Projector output dim (fusion head AV input): {PROJECTOR_OUTPUT_DIM}")
    print(f"ATTENTION_TOKEN_DIM set to: {ATTENTION_TOKEN_DIM}")
    print(f"NUM_ATTENTION_TOKENS_PER_MODAL set to: {NUM_ATTENTION_TOKENS_PER_MODAL} (processed modality feature dimension: {NUM_ATTENTION_TOKENS_PER_MODAL*ATTENTION_TOKEN_DIM})")


    modality_configurations = [
        {'name': 'AVT', 'audio': True,  'video': True,  'text': True},
        # Can add other configurations for testing
        # {'name': 'AV',  'audio': True,  'video': True,  'text': False},
        # {'name': 'T',   'audio': False, 'video': False, 'text': True},
    ]
    all_models_results = {}

    print("Initializing BERT model and tokenizer...")
    bert_tokenizer_global = AutoTokenizer.from_pretrained(BERT_MODEL_NAME_FOR_MAIN)
    bert_feature_extractor_global = AutoModel.from_pretrained(BERT_MODEL_NAME_FOR_MAIN).to(DEVICE)
    BERT_HIDDEN_SIZE_ACTUAL = bert_feature_extractor_global.config.hidden_size

    print("Freezing BERT parameters...") # BERT parameters will not be updated
    for param in bert_feature_extractor_global.parameters():
        param.requires_grad = False
    print("BERT parameters frozen.")

    print("Creating datasets...")
    train_dataset = CustomFeatureDataset(
        concatenated_train_audio, concatenated_train_video, concatenated_train_text, train_labels,
        bert_tokenizer=bert_tokenizer_global, max_bert_len=MAX_BERT_LEN
    )
    dev_dataset = CustomFeatureDataset(
        concatenated_dev_audio, concatenated_dev_video, concatenated_dev_text, dev_labels,
        bert_tokenizer=bert_tokenizer_global, max_bert_len=MAX_BERT_LEN
    )
    test_dataset = CustomFeatureDataset(
        concatenated_test_audio, concatenated_test_video, concatenated_test_text, test_labels,
        bert_tokenizer=bert_tokenizer_global, max_bert_len=MAX_BERT_LEN
    )
    print("Creating data loaders with custom collate_fn...")
    # DataLoader now uses custom_collate_fn
    # drop_last=True might be more stable for training contrastive loss, avoiding last batch being too small
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, drop_last=True if BATCH_SIZE > 1 else False)
    val_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    bce_criterion = nn.CrossEntropyLoss().to(DEVICE)
    contrastive_loss_fn = ContrastiveLossELI5(temperature=TEMPERATURE_CONTRASTIVE).to(DEVICE)

    for config_idx, model_config in enumerate(modality_configurations):
        config_name = model_config['name']
        active_modalities_tuple = tuple(m for m in ['audio', 'video', 'text'] if model_config[m])
        print(f"\n--- [{config_idx+1}/{len(modality_configurations)}] Starting processing for model configuration: {config_name} (Active: {active_modalities_tuple}) ---")

        print(f"Initializing model components for configuration {config_name}...")
        # 1. Initialize trainable hierarchical LSTM aggregators
        current_audio_hier_lstm = HierarchicalLSTMAggregator(
            word_dim=_AUDIO_WORD_DIM_CONST,
            sentence_lstm_hidden_dim=SENTENCE_LSTM_HIDDEN_DIM_CONFIG,
            sample_lstm_hidden_dim=HIER_LSTM_OUTPUT_DIM, # = SAMPLE_LSTM_HIDDEN_DIM_CONFIG
            dropout_rate=HIER_LSTM_DROPOUT
        ).to(DEVICE)
        current_video_hier_lstm = HierarchicalLSTMAggregator(
            word_dim=_VIDEO_WORD_DIM_CONST,
            sentence_lstm_hidden_dim=SENTENCE_LSTM_HIDDEN_DIM_CONFIG,
            sample_lstm_hidden_dim=HIER_LSTM_OUTPUT_DIM, # = SAMPLE_LSTM_HIDDEN_DIM_CONFIG
            dropout_rate=HIER_LSTM_DROPOUT
        ).to(DEVICE)

        # 2. Initialize projection layers (input dimension is the output dimension of hierarchical LSTM)
        current_audio_projector = nn.Linear(HIER_LSTM_OUTPUT_DIM, PROJECTOR_OUTPUT_DIM).to(DEVICE)
        current_video_projector = nn.Linear(HIER_LSTM_OUTPUT_DIM, PROJECTOR_OUTPUT_DIM).to(DEVICE)

        # 3. Initialize fusion head (input dimension is the output dimension of projection layers)
        current_clf_head = MultimodalFusionLSTMHead(
            projected_audio_video_dim=PROJECTOR_OUTPUT_DIM,
            bert_hidden_size=BERT_HIDDEN_SIZE_ACTUAL,
            max_bert_len=MAX_BERT_LEN,
            lstm_hidden_size=TEXT_LSTM_HIDDEN_SIZE, # Text LSTM
            attention_token_dim=ATTENTION_TOKEN_DIM,
            num_attention_tokens_per_modal=NUM_ATTENTION_TOKENS_PER_MODAL,
            num_classes=NUM_CLASSES,
            active_modalities=active_modalities_tuple
        ).to(DEVICE)
        print(f"Model components for {config_name} initialized.")

        # Parameters for BERT will have a specific learning rate, other parameters will use the main LEARNING_RATE
        # Since BERT parameters have requires_grad=False, they won't be updated by the optimizer,
        # but they are listed here for completeness or if fine-tuning BERT was an option.
        optimizer_grouped_parameters = [
            {'params': bert_feature_extractor_global.parameters(), 'lr': BERT_LEARNING_RATE}, # BERT params (frozen)
            # Group all other trainable parameters
            {'params': list(current_audio_hier_lstm.parameters()) + \
                       list(current_video_hier_lstm.parameters()) + \
                       list(current_audio_projector.parameters()) + \
                       list(current_video_projector.parameters()) + \
                       list(current_clf_head.parameters()), 'lr': LEARNING_RATE} # Main learning rate for these
        ]
        optimizer = AdamW(optimizer_grouped_parameters) # Default lr=LEARNING_RATE will be overridden for specific groups

        print(f"Starting training for {config_name}... Total {NUM_EPOCHS} epochs, Batch size {BATCH_SIZE}, Learning rate {LEARNING_RATE}")
        best_val_accuracy_for_config = 0.0
        best_model_state_for_config = {
            'epoch': 0,
            'audio_hier_lstm_state_dict': None, 'video_hier_lstm_state_dict': None,
            'audio_projector_state_dict': None, 'video_projector_state_dict': None,
            'bert_state_dict': None,  # Added this line
            'clf_head_state_dict': None, 'best_val_accuracy': 0.0
        }
        num_training_steps_per_epoch = len(train_loader)
        if num_training_steps_per_epoch == 0 and NUM_EPOCHS > 0 : # Handle case where train_loader is empty
            print(f"Warning: train_loader for {config_name} is empty. Scheduler will not be effective.")
            scheduler = None # Or do not create scheduler
        elif NUM_EPOCHS > 0:
            total_training_steps = num_training_steps_per_epoch * NUM_EPOCHS
            num_warmup_steps = int(total_training_steps * 0.1) # E.g., 10% of steps for warmup
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=total_training_steps)
        else:
            scheduler = None

        if len(train_loader) == 0:
            print(f"Training data loader is empty. Aborting training for {config_name}.")
        else:
            for epoch in range(NUM_EPOCHS):
                train(
                    current_audio_hier_lstm, current_video_hier_lstm,
                    current_audio_projector, current_video_projector, bert_feature_extractor_global, current_clf_head,
                    train_loader, optimizer, scheduler, bce_criterion, contrastive_loss_fn, DEVICE, epoch, NUM_EPOCHS, CONTRASTIVE_LOSS_WEIGHT,
                    model_config, bert_tokenizer_global
                )
                if len(val_loader) > 0:
                    val_accuracy = validate_or_test(
                        current_audio_hier_lstm, current_video_hier_lstm,
                        current_audio_projector, current_video_projector, bert_feature_extractor_global, current_clf_head,
                        val_loader, bce_criterion, DEVICE, epoch, NUM_EPOCHS,
                        model_config, bert_tokenizer_global, mode="Val"
                    )
                    if val_accuracy > best_val_accuracy_for_config:
                        best_val_accuracy_for_config = val_accuracy
                        print(f"Epoch {epoch+1} ({config_name}): New best validation accuracy: {best_val_accuracy_for_config:.4f}.")
                        best_model_state_for_config['epoch'] = epoch
                        best_model_state_for_config['audio_hier_lstm_state_dict'] = copy.deepcopy(current_audio_hier_lstm.state_dict())
                        best_model_state_for_config['video_hier_lstm_state_dict'] = copy.deepcopy(current_video_hier_lstm.state_dict())
                        best_model_state_for_config['audio_projector_state_dict'] = copy.deepcopy(current_audio_projector.state_dict())
                        best_model_state_for_config['video_projector_state_dict'] = copy.deepcopy(current_video_projector.state_dict())
                        best_model_state_for_config['clf_head_state_dict'] = copy.deepcopy(current_clf_head.state_dict())
                        best_model_state_for_config['best_val_accuracy'] = best_val_accuracy_for_config
                        best_model_state_for_config['bert_state_dict'] = copy.deepcopy(bert_feature_extractor_global.state_dict()) # Save BERT state too

                else:
                    print(f"Epoch {epoch+1} ({config_name}): Validation data loader is empty. Skipping validation.")
            print(f"Training for configuration {config_name} complete. Best validation accuracy for this config: {best_val_accuracy_for_config:.4f} (at epoch {best_model_state_for_config['epoch']+1})")

        print(f"\nStarting testing phase for configuration {config_name}...")
        if len(test_loader) == 0:
            print(f"Test data loader is empty. Skipping testing for {config_name}.")
            all_models_results[config_name] = {'val_acc': best_val_accuracy_for_config, 'test_acc': 0.0, 'test_loss': 0.0, 'best_epoch': best_model_state_for_config['epoch']+1 if best_val_accuracy_for_config > 0 else 'N/A'}
        else:
            # Re-initialize models for testing and load best weights
            test_audio_hier_lstm = HierarchicalLSTMAggregator(
                word_dim=_AUDIO_WORD_DIM_CONST, sentence_lstm_hidden_dim=SENTENCE_LSTM_HIDDEN_DIM_CONFIG,
                sample_lstm_hidden_dim=HIER_LSTM_OUTPUT_DIM, dropout_rate=HIER_LSTM_DROPOUT).to(DEVICE)
            test_video_hier_lstm = HierarchicalLSTMAggregator(
                word_dim=_VIDEO_WORD_DIM_CONST, sentence_lstm_hidden_dim=SENTENCE_LSTM_HIDDEN_DIM_CONFIG,
                sample_lstm_hidden_dim=HIER_LSTM_OUTPUT_DIM, dropout_rate=HIER_LSTM_DROPOUT).to(DEVICE)

            test_audio_projector = nn.Linear(HIER_LSTM_OUTPUT_DIM, PROJECTOR_OUTPUT_DIM).to(DEVICE)
            test_video_projector = nn.Linear(HIER_LSTM_OUTPUT_DIM, PROJECTOR_OUTPUT_DIM).to(DEVICE)

            test_clf_head = MultimodalFusionLSTMHead(
                projected_audio_video_dim=PROJECTOR_OUTPUT_DIM, bert_hidden_size=BERT_HIDDEN_SIZE_ACTUAL,
                max_bert_len=MAX_BERT_LEN, lstm_hidden_size=TEXT_LSTM_HIDDEN_SIZE,
                attention_token_dim=ATTENTION_TOKEN_DIM, num_attention_tokens_per_modal=NUM_ATTENTION_TOKENS_PER_MODAL,
                num_classes=NUM_CLASSES, active_modalities=active_modalities_tuple).to(DEVICE)

            # Initialize a separate BERT model for testing to load its specific state
            test_bert = AutoModel.from_pretrained(BERT_MODEL_NAME_FOR_MAIN).to(DEVICE)
            # BERT parameters for test_bert should also be frozen if they were during training
            for param in test_bert.parameters():
                param.requires_grad = False

            epoch_for_log = None
            if best_model_state_for_config['clf_head_state_dict'] is not None:
                test_audio_hier_lstm.load_state_dict(best_model_state_for_config['audio_hier_lstm_state_dict'])
                test_video_hier_lstm.load_state_dict(best_model_state_for_config['video_hier_lstm_state_dict'])
                test_audio_projector.load_state_dict(best_model_state_for_config['audio_projector_state_dict'])
                test_video_projector.load_state_dict(best_model_state_for_config['video_projector_state_dict'])
                test_clf_head.load_state_dict(best_model_state_for_config['clf_head_state_dict'])
                if best_model_state_for_config['bert_state_dict'] is not None:
                     test_bert.load_state_dict(best_model_state_for_config['bert_state_dict'])
                print(f"Loaded best model from Epoch {best_model_state_for_config['epoch']+1} ({config_name}) for testing.")
                epoch_for_log = best_model_state_for_config['epoch']
            else:
                print(f"No best validation model state found for {config_name}. Using final model from training for testing.")
                test_audio_hier_lstm.load_state_dict(current_audio_hier_lstm.state_dict())
                test_video_hier_lstm.load_state_dict(current_video_hier_lstm.state_dict())
                test_audio_projector.load_state_dict(current_audio_projector.state_dict())
                test_video_projector.load_state_dict(current_video_projector.state_dict())
                test_clf_head.load_state_dict(current_clf_head.state_dict())
                test_bert.load_state_dict(bert_feature_extractor_global.state_dict()) # Use the state of the global BERT
                epoch_for_log = NUM_EPOCHS - 1 if NUM_EPOCHS > 0 else None

            test_loss, test_accuracy, test_f1= validate_or_test(
                test_audio_hier_lstm,
                test_video_hier_lstm,
                test_audio_projector,
                test_video_projector,
                test_bert, # Pass the test_bert instance with loaded weights
                test_clf_head,
                test_loader,
                bce_criterion,
                DEVICE,
                epoch=epoch_for_log,
                num_epochs=NUM_EPOCHS, # or best_model_state_for_config['epoch']+1
                current_modality_config=model_config,
                tokenizer_for_padding=bert_tokenizer_global,
                mode="Test"
            )
            print(f"Final test results for configuration {config_name} -> Avg BCE Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
            all_models_results[config_name] = {
                'val_acc': best_val_accuracy_for_config,
                'test_acc': test_accuracy, 'test_loss': test_loss,
                'test_f1': test_f1,
                'best_epoch': best_model_state_for_config['epoch']+1 if best_model_state_for_config['clf_head_state_dict'] is not None else (NUM_EPOCHS if NUM_EPOCHS > 0 else 'N/A')
            }

    print("\n\n--- Final Results Summary for All Model Configurations ---")
    for config_name, results in all_models_results.items():
        print(f"Configuration: {config_name}")
        print(f"  Best Validation Accuracy (Epoch {results['best_epoch']}): {results['val_acc']:.4f}")
        print(f"  Test Set F1 Score: {results['test_f1']:.4f}")
        print(f"  Test Set Accuracy: {results['test_acc']:.4f}")
        print(f"  Test Set Loss: {results['test_loss']:.4f}")
        print("-" * 30)

    print("All operations complete.")