import haiku as hk

from mesh_transformer.layers import AdapterLayerShard
from mesh_transformer.transformer_shard import CausalTransformerShard

"""deprecated"""


class AdapterShard(hk.Module):
    def __init__(self, config, base_model: CausalTransformerShard):
        super().__init__()
        heads = config["n_heads"]
        shards = config["cores_per_replica"]
        layer_count = config["layers"]

        init_scale = 2. / layer_count

        self.base_model = base_model

        self.adapter_layers = []

        for i in range(layer_count):
            self.adapter_layers.append(AdapterLayerShard(config, name=f"adapter_layer_{i}", init_scale=init_scale))

    @property
    def embed(self):
        return self.base_model.embed

    @property
    def transformer_layers(self):
        return self.base_model.transformer_layers

    @property
    def proj(self):
        return self.base_model.proj

    @property
    def rpe(self):
        return self.base_model.rpe

    def eval(self, context, target, z_loss=0., mask=0.0):
        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.base_model.heads_per_shard, 32)
        else:
            attn_bias = 0

        attn_bias += mask

        x = hk.remat(self.embed)(context)

        for l, al in zip(self.transformer_layers, self.adapter_layers):
            x = x + hk.remat(l)(x, attn_bias) + hk.remat(al)(x)

        return hk.remat(self.proj.loss)(x, target, z_loss)

    def init_transform(self, x):
        """this computation isn't useful for anything, it just gets params for init"""
        for al in self.adapter_layers:
            x = x + hk.remat(al)(x)
        return x

    def loss(self, ctx, tgt, z_loss=False, mask=0.0):
        loss, correct = self.eval(ctx, tgt, float(z_loss), mask=mask)

        return {
            "loss": loss.mean(),
            "last_loss": loss[-1].mean(),
            "all_loss": loss,
            "correct": correct
        }

    # TODO: decode funcs
