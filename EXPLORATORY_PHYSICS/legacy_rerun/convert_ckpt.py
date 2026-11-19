"""Wrap a v5 state_dict in the checkpoint format the THEORY scripts expect."""
import sys, torch
sd = torch.load(sys.argv[1], map_location='cpu')
torch.save({'latent_dim': sd['decoder.decoder.0.weight'].shape[1], 'mean': sd['mean'], 'std': sd['std'],
            'model_state_dict': sd}, sys.argv[2])
