import torch
import torch.nn as nn
from transformers import BertModel
from config import BERT_TYPE, BERT_CACHE

class EncoderBERT(nn.Module):
    def __init__(self, device):
        super(EncoderBERT, self).__init__()
        self.model = BertModel.from_pretrained(BERT_TYPE, cache_dir=BERT_CACHE).to(device)
        self.device = device

    def forward(self, input_seq, input_lengths):
        input_seq = input_seq.T
        mask_ids = (input_seq != 0) * 1
        token_ids = torch.ones_like(input_seq).to(self.device)
        enc = self.model.forward(input_ids=input_seq, attention_mask=mask_ids, token_type_ids=token_ids)
        return enc[1]


class ContextEncoderRNN(nn.Module):
    """This module represents the context encoder component of CRAFT, responsible for creating an order-sensitive vector representation of conversation context"""

    def __init__(self, hidden_size, n_layers=1, dropout=0, device=None):
        super(ContextEncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # only unidirectional GRU for context encoding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False).to(device)
        self.device = device

    def forward(self, input_seq, input_lengths, hidden=None):
        # Pack padded batch of sequences for RNN module
        input_lengths = input_lengths.cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # return output and final hidden state
        return outputs, hidden


class SingleTargetClf(nn.Module):
    """This module represents the CRAFT classifier head, which takes the context encoding and uses it to make a forecast"""

    def __init__(self, hidden_size, num_classes, dropout=0.1, device=None):
        super(SingleTargetClf, self).__init__()

        self.hidden_size = hidden_size

        # initialize classifier
        self.layer1 = nn.Linear(hidden_size, hidden_size).to(device)
        self.layer1_act = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2).to(device)
        self.layer2_act = nn.LeakyReLU()
        self.clf = nn.Linear(hidden_size // 2, num_classes).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    def forward(self, encoder_outputs, encoder_input_lengths):
        # from stackoverflow (https://stackoverflow.com/questions/50856936/taking-the-last-state-from-bilstm-bigru-in-pytorch)
        # First we unsqueeze seqlengths two times so it has the same number of
        # of dimensions as output_forward
        # (batch_size) -> (1, batch_size, 1)
        lengths = encoder_input_lengths.unsqueeze(0).unsqueeze(2)
        # Then we expand it accordingly
        # (1, batch_size, 1) -> (1, batch_size, hidden_size)
        lengths = lengths.expand((1, -1, encoder_outputs.size(2)))
        lengths = lengths.to(self.device)
        # take only the last state of the encoder for each batch
        last_outputs = torch.gather(encoder_outputs, 0, lengths - 1).squeeze()
        # forward pass through hidden layers
        layer1_out = self.layer1_act(self.layer1(self.dropout(last_outputs)))
        layer2_out = self.layer2_act(self.layer2(self.dropout(layer1_out)))
        # compute and return logits
        logits = self.clf(self.dropout(layer2_out))
        return logits

def makeContextEncoderInput(utt_encoder_hidden, dialog_lengths, batch_size, batch_indices, dialog_indices):
    """The utterance encoder takes in utterances in combined batches, with no knowledge of which ones go where in which conversation.
       Its output is therefore also unordered. We correct this by using the information computed during tensor conversion to regroup
       the utterance vectors into their proper conversational order."""
    # first, sum the forward and backward encoder states
    utt_encoder_summed = utt_encoder_hidden
    # we now have hidden state of shape [utterance_batch_size, hidden_size]
    # split it into a list of [hidden_size,] x utterance_batch_size
    last_states = [t.squeeze() for t in utt_encoder_summed.split(1, dim=0)]

    # create a placeholder list of tensors to group the states by source dialog
    states_dialog_batched = [[None for _ in range(dialog_lengths[i])] for i in range(batch_size)]

    # group the states by source dialog
    for hidden_state, batch_idx, dialog_idx in zip(last_states, batch_indices, dialog_indices):
        states_dialog_batched[batch_idx][dialog_idx] = hidden_state

    # stack each dialog into a tensor of shape [dialog_length, hidden_size]
    states_dialog_batched = [torch.stack(d) for d in states_dialog_batched]

    # finally, condense all the dialog tensors into a single zero-padded tensor
    # of shape [max_dialog_length, batch_size, hidden_size]
    return torch.nn.utils.rnn.pad_sequence(states_dialog_batched)

class Predictor(nn.Module):
    """This helper module encapsulates the CRAFT pipeline, defining the logic of passing an input through each consecutive sub-module."""

    def __init__(self, encoder, context_encoder, classifier):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.classifier = classifier

    def forward(self, input_batch, dialog_lengths, utt_lengths, batch_indices, dialog_indices,
                batch_size, max_length):
        # Forward input through encoder model
        utt_encoder_hidden = self.encoder(input_batch, utt_lengths)

        # Convert utterance encoder final states to batched dialogs for use by context encoder
        context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths, batch_size,
                                                        batch_indices, dialog_indices)
        if self.context_encoder is not None:
            # Forward pass through context encoder
            context_encoder_outputs, context_encoder_hidden = self.context_encoder(context_encoder_input, dialog_lengths)
        else:
            context_encoder_outputs = context_encoder_input[-1,:,:].unsqueeze(0)

        # Forward pass through classifier to get prediction logits
        logits = self.classifier(context_encoder_outputs, dialog_lengths)

        # Apply sigmoid activation
        # predictions = F.sigmoid(logits)
        predictions = logits
        return predictions