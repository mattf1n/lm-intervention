import torch
import torch.nn as nn
import torch.nn.functional as F

class TXLAttentionOverride(nn.Module):
    """ A copy of `modeling_transfo_xl.RelPartialLearnableMultiHeadAttn` class,
        but with overridden attention values """

    def __init__(self, module, attn_override, attn_override_mask):
        """
        Args:
            attention: instance of modeling_transfo_xl.RelPartialLearnableMultiHeadAttn
                from which variables will be copied
            attn_override: values to override the computed attention weights.
                Shape is [num_heads, seq_len, seq_len]
            attn_override_mask: indicates which attention weights to override.
                Shape is [num_heads, seq_len, seq_len]
        """
        super(TXLAttentionOverride, self).__init__()
        # Copy values from module
        self.output_attentions = module.output_attentions
        self.n_head = module.n_head
        self.d_model = module.d_model
        self.d_head = module.d_head
        self.dropout = module.dropout
        self.qkv_net = module.qkv_net
        self.drop = module.drop
        self.dropatt = module.dropatt
        self.o_net = module.o_net
        self.layer_norm = module.layer_norm
        self.scale = module.scale
        self.pre_lnorm = module.pre_lnorm
        self.r_r_bias = module.r_r_bias
        self.r_w_bias = module.r_w_bias
        self.r_net = module.r_net
        # Set attention override values
        self.attn_override = attn_override
        self.attn_override_mask = attn_override_mask

    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and torch.sum(attn_mask).item():
            attn_mask = attn_mask == 1  # Switch to bool
            if attn_mask.dim() == 2:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = (
                        attn_score.float().masked_fill(attn_mask[None, :, :, None], -65000).type_as(attn_score)
                    )
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], -1e30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -1e30).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # Intervention:
        # attn_override and attn_override_mask are of shape (bsz, n_heads, query_seq_len, key_seq_len)
        # attn_prob is of shape (query_seq_len, key_seq_len, bsz, n_heads)
        _, _, override_q_len, override_k_len = self.attn_override_mask.shape
        attn_prob[:override_q_len, :override_k_len, :, :] = torch.where(
            self.attn_override_mask.permute(2, 3, 0, 1),
            self.attn_override.permute(2, 3, 0, 1),
            attn_prob[:override_q_len, :override_k_len, :, :])


        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            outputs = [w + attn_out]
        else:
            # residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]

        if self.output_attentions:
            outputs.append(attn_prob)

        return outputs
