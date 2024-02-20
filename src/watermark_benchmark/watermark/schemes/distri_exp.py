import scipy
import torch
from scipy import stats

from watermark_benchmark.utils.classes import VerifierOutput
from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.verifier import (
    EmpiricalVerifier,
    Verifier,
)
import torch.nn.functional as F
import math
import numpy as np

class DistriEXPGenerator(Watermark):

    def __init__(self, rng, verifier, tokenizer, temp, delta, gamma):
        super().__init__(rng, verifier, tokenizer, temp)
        self.delta = delta
        self.gamma = gamma
        self.temp = temp

    def _process(self, logits, previous_tokens, ids):
        """
        Applies the watermarking scheme to the input logits.

        Args:
            logits (torch.Tensor): The input logits.
            previous_tokens (torch.Tensor): The previous tokens in the sequence.
            ids (torch.Tensor): The IDs of the previous tokens.

        Returns:
            torch.Tensor: The logits with the watermark applied.
        """

        # Truncate unused logits
        logits = logits[:, : self.rng.vocab_size]

        N, _ = logits.shape

        # Get greenlist and update logits
        seeds = self.rng.rand_index(self.rng.get_seed(previous_tokens, ids), 0)
        greenlist = self.rng.green_list(seeds, self.gamma)
        logits[
            torch.arange(N).unsqueeze(1).expand(-1, greenlist.size(1)),
            greenlist,
        ] += self.delta

        local_logits = logits[:, : self.rng.vocab_size] / self.temp

        # Compute probabilities and get random values
        probs = F.softmax(local_logits, dim=-1)
        hash_values = self.rng.rand_range(
            self.rng.get_seed(previous_tokens, ids=ids),
            self.rng.vocab_size,
            device=probs.device,
        )
        hash_values = torch.div(-torch.log(hash_values), probs)

        # Get next token, and update logit
        next_token = hash_values.argmin(dim=-1)

        # print("Next token choice: {} | {} (P = {})".format(next_token.cpu(), hash_values.min(dim=-1).cpu(), probs[next_token.to(probs.device)].cpu()))

        local_logits[:] = -math.inf
        local_logits[torch.arange(local_logits.shape[0]), next_token] = 0

        # print("Next tokens: {}".format(next_token))

        return local_logits


class DistriEXPVerifier(Verifier):
    """
    A verifier that checks for distribution shift in a sequence of tokens.

    Args:
        rng (RandomNumberGenerator): A random number generator.
        pvalue (float): The p-value threshold for the binomial test.
        tokenizer (Tokenizer): A tokenizer for the sequence of tokens.
        gamma (float): The proportion of tokens that are allowed to be different.

    Attributes:
        gamma (float): The proportion of tokens that are allowed to be different.
    """

    def __init__(self, rng, pvalue, tokenizer, gamma):
        super().__init__(rng, pvalue, tokenizer)
        self.gamma = gamma
        self.log = True

    def _verify(self, tokens, index=0):
        return_value_cs = self._verify_exp(tokens, index)
        return_value_dis = self._verify_distribution(tokens, index)
        return_value = VerifierOutput()
        for i in range(return_value_cs.sequence_token_count + 1):
            return_value.update(i, self.combined_pvalue(return_value_cs.pvalues[i], return_value_dis.pvalues[i]))
        return return_value

    def combined_pvalue(self, pvalue1, pvalue2):
        # Calculate the chi-square statistic
        epsilon = 1e-10
        chi_square_statistic = -2 * (np.log(pvalue1 + epsilon) + np.log(pvalue2 + epsilon))

        # Degrees of freedom for two tests is 4 (2 times the number of tests)
        degrees_of_freedom = 4

        # Calculate the combined p-value
        combined_p = stats.chi2.sf(chi_square_statistic, degrees_of_freedom)

        return combined_p

    def _verify_distribution(self, tokens, index=0):
        cumul = []
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

    def _verify_exp(self, tokens, index=0):
        cumul = []
        for i, tok in enumerate(tokens):
            prev_values = tokens[:i]
            seed = self.rng.get_seed(prev_values, [index])
            hv = self.rng.rand_index(seed, tok).item()

            cumul.append((hv, i))

        return_value, ctr, ctn = VerifierOutput(), 0, 0
        for i, val in enumerate(cumul):
            ctn += 1
            ctr += val[0] if not self.log else -np.log(max(0.00001, 1 - val[0]))
            if not self.log:
                # pval = tfp.distributions.Bates(ctn).survival_function(ctr/ctn)
                pval = scipy.stats.norm.sf(
                    ctr / ctn, loc=0.5, scale=1 / math.sqrt(12 * ctn)
                )
            else:
                # pval = s(ctr, loc=0.5, scale=1/math.sqrt(12*(ctn))) if not self.log else scipy.stats.gamma.sf(ctr, ctn)
                pval = scipy.stats.gamma.sf(ctr, ctn)
            return_value.update(i, pval)
        return return_value

