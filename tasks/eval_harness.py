from functools import partial

import transformers
from lm_eval.base import LM
from tqdm import tqdm
import numpy as np

from tasks.util import sample_batch, shrink_seq
import multiprocessing
import ftfy

tokenizer = None


def process_init():
    global tokenizer
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.model_max_length = int(1e30)
    tokenizer.pad_token = "<|endoftext|>"

    assert tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]


def process_request(x, seq):
    global tokenizer

    ctx, cont = x

    ctx_tokens = tokenizer.encode("<|endoftext|>" + ftfy.fix_text(ctx, normalization="NFKC"))
    cont_tokens = tokenizer.encode(ftfy.fix_text(cont, normalization="NFKC"))

    all_tokens = ctx_tokens + cont_tokens
    all_tokens = np.array(all_tokens)[-seq:]  # truncate sequence at seq length

    provided_ctx = len(all_tokens) - 1
    pad_amount = seq - provided_ctx

    return {
        "obs": np.pad(all_tokens[:-1], ((0, pad_amount),), constant_values=50256),
        "target": np.pad(all_tokens[1:], ((0, pad_amount),), constant_values=50256),
        "ctx_length": seq,
        "eval_mask": np.logical_and(
            np.arange(0, seq) > len(all_tokens) - len(cont_tokens) - 2,
            np.arange(0, seq) < len(all_tokens) - 1
        ),
    }


class EvalHarnessAdaptor(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def loglikelihood_rolling(self, requests):
        raise Exception("unimplemented")

    def __init__(self, tpu_cluster, seq, batch, shrink):
        super().__init__()
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch
        self.shrink = shrink

        self.pool = multiprocessing.Pool(initializer=process_init)
        process_init()

    def convert_requests(self, requests):
        return self.pool.imap(partial(process_request, seq=self.seq), requests)

    def loglikelihood(self, requests):
        output = []

        r = self.convert_requests(requests)
        zero_example = process_request(requests[0], self.seq)

        for b in tqdm(sample_batch(r, self.batch, zero_example),
                      desc="LM eval harness",
                      total=len(requests) // self.batch,
                      mininterval=5):
            if self.shrink:
                b = shrink_seq(b)

            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output


class LocalTPUCluster:
    def __init__(self, model):
        self.nodes = [model]

    def eval(self, data):
        if isinstance(data, dict):
            data_chunked = [{} for _ in self.nodes]
            for k, v in data.items():
                v_chunks = np.array_split(v, len(self.nodes), axis=0)
                for idx, v_chunk in enumerate(v_chunks):
                    data_chunked[idx][k] = v_chunk

            res = []
            for n, d in zip(self.nodes, data_chunked):
                res.append(n.eval(d))

            total = 0
            correct = 0
            last_correct = 0

            total_last_loss = 0
            mask_loss = []
            each_correct = []

            for input, output in zip(data_chunked, res):
                correct_and_valid = np.logical_and(output["correct"], input["eval_mask"])

                correct_tokens_count = np.sum(correct_and_valid, -1)
                valid_tokens_count = np.sum(input["eval_mask"], -1)

                correct_example = np.logical_and(valid_tokens_count == correct_tokens_count, valid_tokens_count > 0)
                valid_example = valid_tokens_count > 0
                last_correct_example = correct_and_valid[:, -1]

                each_correct += correct_example.tolist()

                total += sum(valid_example)
                correct += sum(correct_example)
                last_correct += sum(last_correct_example)
                total_last_loss += sum(valid_example * output["last_loss"])

                valid_loss = np.sum(output["all_loss"] * input["eval_mask"], -1)
                mask_loss += valid_loss.tolist()

            return {
                "total": total,
                "correct": correct,
                "last_correct": last_correct,
                "last_loss": total_last_loss,
                "mask_loss": np.array(mask_loss),
                "each_correct": np.array(each_correct)
            }
        else:
            data_chunks = np.array_split(data, len(self.nodes), axis=0)

            res = []
            for n, d in zip(self.nodes, data_chunks):
                res.append(n.eval({
                    "obs": d[:, :-1],
                    "target": d[:, 1:],
                }))

            return np.array([i["loss"] for i in res]).mean()
