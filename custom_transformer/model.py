# model.py
#BU KODLAR TRAİN.PY DOSYASINDA İMPORT EDİLİYOR. ÇALIŞTIRILMASINA GEREK YOK

import torch
import torch.nn as nn

class StoryTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        emotion_size,
        embed_size=256,
        hidden_size=512,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        max_len=500
    ):
        super(StoryTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.emotion_embedding = nn.Embedding(emotion_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embed_size))

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=1,  # dummy encoder
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
    
    def forward(self, input_seq, emotion_label, tgt_mask=None):
        """
        input_seq: (batch, seq_len)
        emotion_label: (batch)
        """
        batch_size, seq_len = input_seq.size()
        
        # Embedding
        input_emb = self.token_embedding(input_seq) + self.positional_encoding[:, :seq_len, :]
        emotion_emb = self.emotion_embedding(emotion_label).unsqueeze(1)  # (batch, 1, embed)

        # Başlangıca duygu embedding ekle
        combined_emb = torch.cat((emotion_emb, input_emb), dim=1)

        # Causal mask
        if tgt_mask is None:
            tgt_mask = self.transformer.generate_square_subsequent_mask(combined_emb.size(1)).to(combined_emb.device)

        # Encoder input (dummy) ve Decoder input
        encoder_input = combined_emb.transpose(0, 1)  # (seq_len, batch, embed)
        decoder_input = combined_emb.transpose(0, 1)

        output = self.transformer(
            encoder_input, decoder_input,
            tgt_mask=tgt_mask
        )
        output = self.fc_out(output.transpose(0, 1))  # (batch, seq_len, vocab_size)
        return output
