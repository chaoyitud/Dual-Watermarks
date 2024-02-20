# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import random
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import scipy
import torch
from scipy import stats
from transformers import LlamaForCausalLM, LlamaTokenizer
import numpy as np
import torch.nn.functional as F
from watermark_benchmark.watermark.templates.random import EmbeddedRandomness
from watermark_benchmark.utils.classes import VerifierOutput
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import special
from scipy.optimize import fminbound
from scipy.spatial.distance import cosine
from scipy.stats import norm
from watermark_benchmark.watermark.templates.verifier import Verifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from watermark_benchmark.watermark.templates.generator import Watermark

class ConSearchDistriGenerator(Watermark):
    def __init__(self,
                 rng,
                 verifiers,
                 tokenizer: LlamaTokenizer,
                 alpha: float = 0.0,
                 beta: float = 0.3,
                 window_size: int = 50,
                 temp: float = 0.8,
                 delta: float = 2.5,
                 gamma: float = 0.5,
                 beam_width: int = 4,
                 ):
        super().__init__(rng, verifiers, tokenizer, temp)
        # model config
        self.tokenizer = tokenizer
        self.verifiers = verifiers
        # watermark config
        self.rng =rng
        self.alpha = alpha
        self.device = device
        self.beta = beta
        self.window_size = window_size
        self.model = None
        self.pad_id = None
        self.eos_id = None
        self.max_seq_len = None
        self.temp = temp
        self.delta = delta
        self.gamma = gamma
        self.beam_width = beam_width
        self.p_hat = True

    def install_model(self, model: LlamaForCausalLM):
        self.model = model
        try:
            self.max_seq_len = model.config.max_sequence_length
        except Exception:
            self.max_seq_len = 2048

        self.pad_id = model.config.pad_token_id if model.config.pad_token_id else 0
        self.eos_id = model.config.eos_token_id

    def _process(self, logits, previous_tokens, ids):
        pass
    @torch.no_grad()
    def generate(
            self,
            batch,
            max_gen_len: int,
            temperature: float = 0.8,
            top_p: float = 0.95,
            ids: List[int] = None,
    ) -> List[str]:
        """
        Generate text from prompts.
        Adapted from https://github.com/facebookresearch/llama/
        """
        bsz = batch.input_ids.shape[0]
        max_prompt_size = batch.input_ids.shape[1]
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        # Prepare tokens tensor with padding
        tokens = torch.full((bsz, total_len), self.pad_id, dtype=torch.long).to(self.device)
        tokens[:, :max_prompt_size] = batch.input_ids
        input_text_mask = tokens != self.pad_id
        start_pos = max_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                outputs = self.model.forward(
                    tokens[:, prev_pos:cur_pos], attention_mask=input_text_mask[:, prev_pos:cur_pos],use_cache=True,
                    past_key_values=outputs.past_key_values if prev_pos > 0 else None, output_hidden_states=True
                )
                past_key_values = outputs.past_key_values
                last_hidden_states = outputs.hidden_states[-1]  # [B, S, E]
                logit_for_next_step = outputs.logits[:, -1, :]
                #next_id = self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p)
                bsz, seqlen, embed_dim = last_hidden_states.size()
                #last_hidden_states = last_hidden_states[:, 1:2, :]
            # Generate seeds and green_list for current position
            if True:
                seeds = self.rng.rand_index(self.rng.get_seed(tokens[:, start_pos:cur_pos], ids), 1)
            else:
                seeds = self.rng.rand_index(self.rng.get_seed(tokens[:, start_pos:cur_pos-1], ids), 1)
            green_list = self.rng.green_list(seeds, self.beta).to(self.device)  # [B, vocab_size]

            seeds_kgw = self.rng.rand_index(self.rng.get_seed(tokens[:, start_pos:cur_pos], ids), 0)
            green_list_kgw = self.rng.green_list(seeds_kgw, self.gamma).to(self.device)  # [B, vocab_size]

            N, _ = logit_for_next_step.shape

            logit_for_next_step[
                torch.arange(N).unsqueeze(1).expand(-1, green_list_kgw.size(1)),
                green_list_kgw,
            ] += self.delta

            logit_for_next_step = logit_for_next_step / self.temp

            current_tokens = tokens[:, cur_pos - 1].unsqueeze(1)  # [B, 1]

            # Initialize alpha tensor with zeros (default for normal decoding)
            alpha_tensor = torch.zeros(bsz, device=self.device)

            mask = (current_tokens == green_list).any(dim=1)  # [B]

            # Set alpha to self.alpha for those not in green_list
            alpha_tensor[mask] = self.alpha
            # Perform contrastive decoding with determined alpha values
            next_id, past_key_values, last_hidden_states, logit_for_next_step = self.ContrastiveDecodingOneStepFast(
                past_key_values,
                last_hidden_states,
                logit_for_next_step,
                alpha=alpha_tensor
            )
            next_id = next_id.squeeze(-1)

            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_id)
            prev_pos = cur_pos
        # tokens after eos_id are padded
        for ii in range(tokens.size(0)):
            # Find the first eos_id in each sequence
            eos_pos = (tokens[ii] == self.eos_id).nonzero(as_tuple=True)[0]
            if eos_pos.nelement() > 0:
                first_eos_pos = eos_pos[0]
                # Replace all tokens after the first eos_id with pad_id
                tokens[ii, first_eos_pos + 1:] = self.pad_id
        decoded_outputs = self.tokenizer.batch_decode(tokens[:, start_pos:], skip_special_tokens=True)

        return decoded_outputs

    def sample_next(
            self,
            logits: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
            ngram_tokens: torch.LongTensor,  # (bsz, ngram): tokens to consider when seeding
            temperature: float = 0.8,  # temperature for sampling
            top_p: float = 0.95,  # top p for sampling
    ) -> torch.LongTensor:
        """ Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort,
                                           num_samples=1)  # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token)  # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

    def ContrastiveDecodingOneStepFast(
            self,
            past_key_values,
            last_hidden_states,
            logit_for_next_step,
            beam_width=4,
            alpha=0.8,
    ):
        beam_width = self.beam_width
        bsz, seqlen, embed_dim = last_hidden_states.size()
        # keep the hidden_states length as 50
        if seqlen > self.window_size:
            last_hidden_states = last_hidden_states[:, -self.window_size:, :]
            seqlen = self.window_size
        p = random.uniform(0, 1)
        next_probs = F.softmax(logit_for_next_step, dim=-1)
        _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)  # [B, K]
        top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)  # [B, K]
        # compute new hidden
        past_key_values = enlarge_past_key_values(past_key_values, beam_width)
        output = self.model(
            input_ids=top_k_ids.view(-1, 1),
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]  # [B*K, V]
        next_hidden = output.hidden_states[-1]  # [B*K, 1, E]
        context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz * beam_width,
                                                                                                seqlen,
                                                                                                embed_dim)  # [B*K, S, E]

        selected_idx = ranking_fast(
            context_hidden,
            next_hidden,
            top_k_probs,  # [B, K]
            alpha,
            beam_width,
        )  # [B]
        # prepare for the next step
        next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)
        # TODO: check if the next_id is eos, if the next_id is eos, then set the next_id with the largest prob
        is_eos = next_id.squeeze(-1) == self.eos_id

        if is_eos.any():
            # For each element in the batch, if next_id is EOS, replace it with the ID having the largest probability
            for idx in range(bsz):
                if is_eos[idx]:
                    # Exclude the EOS token and get the token with the highest probability
                    _, top_one_ids = torch.topk(logit_for_next_step, dim=-1, k=1)
                    next_id[idx] = top_one_ids[idx]

        next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))  # [B, K, E]
        next_hidden = next_hidden[range(bsz), selected_idx, :]  # [B, E]
        last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)  # [B, S, E]
        past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
        logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]  # [B, V]
        # next_id: [B, 1]
        return next_id, past_key_values, last_hidden_states, logits


def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz * beam_width, num_head, seq_len,
                                                                                esz)  # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam // beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))  # [B, K, num_head, seq_len, esz]
            item = item[range(bsz), selected_idx, :, :, :]  # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def ranking_fast(context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
        alpha: bsz
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)  # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)  # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]

    # Reshape alpha to align with the scores and next_top_k_probs tensor
    alpha = alpha.repeat_interleave(beam_width)  # Repeat each alpha value 'beam_width' times

    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores
    scores = torch.stack(torch.split(scores, beam_width))  # [B, K]
    selected_idx = scores.max(dim=-1)[1]  # [B]
    return selected_idx

class ConSearchDistriEmpiricalVerifier(Verifier):
    def __init__(self, rng, pvalue, tokenizer, beta, window_size, model=None, gamma=0.5):
        super().__init__(rng, pvalue, tokenizer)
        self.beta = beta
        self.model = model
        self.window_size = window_size
        self.gamma = gamma
        self.runs = 1000
        self.half_length = None
        self.score_len = 0
        self.sum_of_parts = None

    def _verify(self, tokens, index=0):
        return_value_cs = self._verify_contrastive(tokens, index)
        return_value_dis = self._verify_distribution(tokens, index)
        return_value = VerifierOutput()
        for i in range(return_value_cs.sequence_token_count+1):
            return_value.update(i, self.combined_pvalue(return_value_cs.pvalues[i], return_value_dis.pvalues[i]))
        return return_value

    def combined_pvalue(self,pvalue1, pvalue2):
        # Calculate the chi-square statistic
        epsilon = 1e-10
        chi_square_statistic = -2 * (np.log(pvalue1+epsilon) + np.log(pvalue2+epsilon))

        # Degrees of freedom for two tests is 4 (2 times the number of tests)
        degrees_of_freedom = 4

        # Calculate the combined p-value
        combined_p = stats.chi2.sf(chi_square_statistic, degrees_of_freedom)

        return combined_p

    def get_pvalue(self, score: int, ntoks: int, eps=1e-10):
        """ from cdf of a normal distribution """
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)

    def _verify_contrastive(self, tokens, index=0):
        self.score_len = 0
        self.sum_of_parts = np.zeros((self.runs, 2))
        self.half_length = np.zeros((self.runs, 2))
        score_list = []
        score_chose_list = []
        score_none_chose_list = []
        ### get the hidden states of the text
        input_ids = torch.tensor(tokens).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0]
        hidden_states = hidden_states / torch.norm(hidden_states, dim=-1, keepdim=True)
        ### get the max cosine similarity between the a hiden state and previous hidden states
        tokens = torch.tensor(tokens).to(self.model.device)
        return_value = VerifierOutput()
        return_value.update(0, 1.0)
        return_value.update(1, 1.0)
        for i in range(1, len(tokens)):
            window_size = self.window_size if i > self.window_size else i
            prev_values = tokens[:i]
            current_token = tokens[i-1].item()
            seeds = self.rng.rand_index(
                self.rng.get_seed(prev_values, [index]), 1
            )
            greenlist = self.rng.green_list(seeds, self.beta).to(self.rng.device)
            ### calculate the score
            current_hidden_state = hidden_states[i]
            previous_hidden_states = hidden_states[i-window_size:i]
            cosine_matrix = torch.matmul(previous_hidden_states, current_hidden_state).squeeze(-1)  # [B, S]
            scores, _ = torch.max(cosine_matrix, dim=-1)
            score_list.append(scores.item())
            if current_token in set(greenlist.squeeze().cpu().numpy()):
                score_chose_list.append(scores.item())
            else:
                score_none_chose_list.append(scores.item())
            #if i>20:
            #if i == len(tokens)-1:
            if i>1:
                score_chose_list_mean = sum(score_chose_list) / len(score_chose_list) if len(score_chose_list) else 0
                score_none_chose_list_mean = sum(score_none_chose_list) / len(score_none_chose_list) if len(score_none_chose_list) else 0

                score = self.permution_test_pvalue(score_list,
                                                   score_none_chose_list_mean - score_chose_list_mean)
                # if score is not nan:
                if score != np.nan:
                    return_value.update(i, score)
                else:
                    return_value.update(i, 0.5)
            else:
                return_value.update(i, 0.5)
        return return_value

    def _verify_distribution(self, tokens, index=0):
        cumul = []
        tokens = torch.tensor(tokens).to(self.model.device)

        for i, _ in enumerate(tokens):
            prev_values = tokens[:i]
            current_token = tokens[i].item()

            seeds = self.rng.rand_index(
                self.rng.get_seed(prev_values, [index]), 0
            )
            greenlist = self.rng.green_list(seeds, self.gamma)



            if current_token in set(greenlist.squeeze().cpu().numpy()):
                cumul.append(1)
            else:
                cumul.append(0)

        if not len(cumul):
            return VerifierOutput()

        ctr = 0
        return_value = VerifierOutput()
        for i, val in enumerate(cumul):
            ctr += val
            cnt = i + 1
            #print("ctr: ", ctr, "cnt: ", cnt)
            nd = scipy.stats.binomtest(
                ctr, cnt, self.gamma, alternative="greater"
            ).pvalue
            #nd = self.get_pvalue(ctr, cnt)
            return_value.update(i, nd)

        return return_value

    def id(self):
        """
        Returns the ID of the verifier.

        Returns:
            tuple: The ID of the verifier.
        """
        return (self.pvalue, "theoretical", "standard")


    def permution_test_pvalue(self, max_cosine_similarity, similarity_difference):
        if len(max_cosine_similarity) == 1:
            raise ValueError("Need at least two values to calculate p-value")
        if len(max_cosine_similarity) == 2:
            mid_point = self.runs // 2
            self.sum_of_parts[:mid_point, 0] = max_cosine_similarity[0]
            self.sum_of_parts[mid_point:, 1] = max_cosine_similarity[1]
            self.half_length[:mid_point, 0] = 1
            self.half_length[mid_point:, 1] = 1
        else:
            new_similarity = max_cosine_similarity[self.score_len:]
            for new_value in new_similarity:
                self.calculate_mean_differences_for_step_np(new_value)
        difference = self.sum_of_parts[:, 0] / self.half_length[:, 0] - self.sum_of_parts[:, 1] / self.half_length[:, 1]
        self.score_len = len(max_cosine_similarity)
        std = np.std(difference)
        mean = np.mean(difference)
        z = (similarity_difference - mean) / std
        p = self.compute_p_value(z)
        #print(difference)
        #print(similarity_difference)
        # number of runs that simlarity_difference is greater than the max_cosine_similarity/len(max_cosine_similarity)
        #p = sum(difference > similarity_difference) / len(difference)
        #print(p)
        return p


    def calculate_mean_differences_for_step_np(self, new_value):
        indices = np.random.choice([0, 1], size=self.runs)
        self.half_length[np.arange(self.runs), indices] += 1
        self.sum_of_parts[np.arange(self.runs), indices] += new_value

    def compute_p_value(self, z_score):
        """Compute one-tailed p-value for a given z-score."""
        return 1 - norm.cdf(z_score)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder="offload",
    )
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    rng = EmbeddedRandomness([0], device=[0], vocab_size=tokenizer.vocab_size, hash_len=1, min_hash=False)
    verifier = ConSearchDistriEmpiricalVerifier(rng, 0.02, tokenizer, 0.5, model=model, window_size=50)

    generator = ConSearchDistriGenerator(rng,verifier, tokenizer, alpha=0.6, beta=0.5, window_size=50, temp=1)
    generator.install_model(model)
    prompt = ["Yu-Gi-Oh is"]
    batch = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                    )
    generations = generator.generate(batch, max_gen_len=1024, ids=[0])
    print(generations)
    score = verifier._verify_contrastive(tokenizer.encode(generations[0]), index=0)
    print(score)
    #score = verifier._verify_distribution(tokenizer.encode(generations[0]), index=0)
    #print(score)
    score = verifier._verify(tokenizer.encode(generations[0]), index=0)
    print(score)