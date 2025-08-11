# HRM Zero

HRM modified so as to not take T iterations when T is high. Original code is from https://github.com/sapientinc/HRM. Core modification featured below. Also AdamW optimizer was swapped in for Adam_Atan and Sparse Sign Gradient Descent optimizers.

```py

class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
...
    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        # with torch.no_grad():
        #     z_H, z_L = carry.z_H, carry.z_L

        #     for _H_step in range(self.config.H_cycles):
        #         for _L_step in range(self.config.L_cycles):
        #             if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
        #                 z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

        #         if not (_H_step == self.config.H_cycles - 1):
        #             z_H = self.H_level(z_H, z_L, **seq_info)

        self.i += 1

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        if self.i % self.config.H_cycles == 0:
            z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
```
