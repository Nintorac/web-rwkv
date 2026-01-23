Flow chart

Figure 1 presents the overall architecture of RWKV-7. Please refer to Appendix F for more details.

<!-- image -->

Bar chart

Figure 2: A simple illustration of the update mechanism of a single head of RWKV-7's state. Note that the actual state size is 64 × 64 per head, not 4 × 4.

<!-- image -->

## 4 Method

In this section, we use D to denote the model dimension. Bold capital letters represent trainable matrices, and vectors without a subscript t are trainable parameters. The first subscript denotes sequence position and second subscript denotes layer index, where necessary. We use the convention that all vectors are row vectors unless explicitly transposed, so all matrices operate on the right side, therefore a $^{T}$b is an outer product and ab T is an inner one. We use the square subscript to denote a placeholder for variable names and use the ⊓ ⊙ sign for cumulative matrix multiplication. See Appendix G for a pseudocode implementation of these formulas.

## 4.1 Time Mixing

Weight Preparation Along the lines of (Peng et al., 2024b), we introduce the following notation templates for common operators in the model, using the square subscript to denote a variable:

$$\text {lerp} ( a , b , x ) = a + ( b - a ) \odot x ,$$

$$\text {lormlp} _ { \Box } ( f , x , \text {bias} ) = f ( x A _ { \Box } ) B _ { \Box } + ( \lambda _ { \Box } \text { if bias else 0} ) ,$$

Unless explicitly stated, all vectors appearing in this section are dimension D .

We extend the use of low-rank MLP (a 2-layer MLP with small hidden dimension compared to input and output), abbreviated as lorlamp, to implement data dependency using minimal parameters.

The replacement key ˜ k , value v , decay w , removal key κ , in-context learning rate a , receptance r , and rwkv gate parameters are computed as follows (outputs annotated with ):

| x □ t     | = lerp ( x$_{t}$ , x$_{t}$$_{-}$$_{1}$ , µ □ )           | □ ∈ { r , k , v , d , a , g } ,   | token shifted inputs   | (3)                      |
|-----------|----------------------------------------------------------|-----------------------------------|------------------------|--------------------------|
| a$_{t}$   | = sigmoid (loramlp$_{a}$ (Identity, x a t , bias=True)), | ▷in-context learning rate         | (4)                    | in-context learning rate |
| k$_{t}$   | = x k t W$_{k}$ ,                                        | key precursor                     | (5)                    | (6)                      |
| κ$_{t}$   | = k$_{t}$ ⊙ ξ ,                                          | ▷removal key                      | (6)                    | replacement key          |
| ˜ k$_{t}$ | = k$_{t}$ ⊙ lerp (1, a$_{t}$ , α ) ,                     | value residual gate               | (8)                    | (7)                      |
| v$_{t}$   | = sigmoid (loramlp$_{v}$ (Identity, x v t , bias=True)), | value precursor                   | (9)                    | value precursor          |
| v ' t , l | = x v t W$_{v}$ ,                                        | value l = 0                       | ▷value                 | (10)                     |
| v$_{t}$   | = { v ' t , 0 ,                                          | layer l = 0                       | ▷value                 | (10)                     |
| d$_{t}$   | = lerp ( v ' t , 0 , v ' t , l , v$_{t}$ ) ,             | layer l ≥ 1 ,                     | decay precursor        | (11)                     |
| d$_{t}$   | = lorlamp$_{d}$ (tanh, x d t , bias=True) ,              | decay precursor                   | (12)                   | (13)                     |
| w$_{t}$   | = exp ( - e - 0 . $^{5}$sigmoid ( d$_{t}$ )) ,           | ▷decay                            | (12)                   | (13)                     |
| r$_{t}$   | = x r t W$_{r}$ ,                                        | receptance                        | (13)                   | (14)                     |
| g$_{t}$   | = lorlamp$_{g}$ (sigmoid, x g t , bias=False)            | ▷rwkv gate                        | (14)                   | (15)                     |

ξ is a learned parameter representing the removal key multiplier, which transforms the original key into a version to be removed from the state. In practice, ξ lies in a range of approximately [ - 5.3, 9.4].

α is a learned parameter representing the replacement rate booster, which adjusts the amount added back to the state after the transition matrix is applied.

Unlike r , k and v which are the main carriers of information, g , d , ν and a act like gates which control the amount of information allowed to pass.

For comprehensive statistics of ξ , α and biases of d$\_{t}$ observed in the released RWKV-7 model, including extremum values, mean measurements, and distribution trends, see Appendix L.

For the computation of x □ t , we removed data dependency of linear interpolation from RWKV-6 to improve training speed.

We adapted the idea of Value Residual Learning Zhou et al. (2024) for the computation of v$\_{t}$ , which has shown to improve the final language modeling loss. ν$\_{t}$ represents the value residual mix, which interpolates between the layer zero and current layer value precursors: ν$\_{t}$$\_{,}$$\_{0}$ and v$\_{t}$$\_{,}$$\_{l}$ .

We also updated the formula for computation of w$\_{t}$ , restricting all entries in ( exp( - e - 0 . $^{5}$), 1) in favor of a smaller condition number for diag( w$\_{t}$ ), which maintains better training stability, and was beneficial to accuracy of the backward pass.

The ˜ k$\_{t}$ in the formula can be regarded as a "normalized key", a design to ensure that the state of w kv contains columns of O (1) size. Normally, we expect ˜ k$\_{t}$ = k$\_{t}$ ⊙ (1 - w$\_{t}$ ), as employed in RWKV-6c (see Appendix F), so that w kv$\_{t}$ rows are linear interpolations between w kv$\_{t}$$\_{-}$$\_{1}$ and v T t k$\_{t}$ controlled by w$\_{t}$ . However, to further enhance expressivity, we decide to decouple w$\_{t}$ and a$\_{t}$ . We further decouple a$\_{t}$ from the amount actually added to the state, allowing the replacement rate booster α to interpolate the amount added between the normal in-context learning rate and 1.0. Importantly, all of these modifications operate on a per-channel basis. The numerical range of RWKV-7's w kv entries are generally stable as in RWKV-6c, unlike RWKV-6, where entries of the states can accumulate to thousands (see Appendix J for a state visualization).

The Weighted Key Value State Evolution After weight preparation, we reshape ( r , w , ˜ k , v , κ , a )$\_{t}$, splitting them to h heads, with each head sized D / h . We always assume that h is a factor of D and heads are equally split. All operations in this section are shown per-head.

Before mixing in the time dimension, 𝜅$\_{𝑡}$ is normalized per head:

$$\hat { \kappa } _ { t } = \kappa _ { t } / \| \kappa _ { t } \| _ { 2 }$$

The wkv (Weighted Key Value) is a multi-headed matrix-valued state of fast weights that undergoes dynamic evolution. The evolution of wkv is crucial for encoding context information by learning at test time to map keys to values. We start by defining the WKV time mixing as the recurrence relation

$$\ w k v _ { 0 } = 0 ,$$

$$\ w k v _ { t } = \ w k v _ { t - 1 } \left ( \text {diag} ( w _ { t } ) - \hat { \kappa } _ { t } ^ { T } ( a _ { t } \odot \hat { \kappa } _ { t } ) \right ) + v _ { t } ^ { T } \cdot \tilde { k } _ { t }$$

Compared to RWKV-5 and RWKV-6, the wkv in this paper is transposed to ensure consistency with RWKV-7's code.

The wkv$\_{𝑡}$ attention calculation can alternatively be written in a parallel manner:

$$\ w k v _ { t } = \sum _ { i = 1 } ^ { t } \left ( v _ { i } ^ { T } \tilde { k } _ { i } \prod _ { j = i + 1 } ^ { t } \left ( \text {diag} ( w _ { j } ) - \hat { \kappa } _ { j } ^ { T } ( a _ { j } \odot \hat { \kappa } _ { j } ) \right ) \right ) \in \mathbb { R } ^ { ( D / h ) \times ( D / h ) }$$

The recurrent transition design has parallels with Schlag et al. (2021), but crucially the transition matrix

$$G _ { t } = \text {diag} ( w _ { t } ) - \hat { \kappa } _ { t } ^ { T } ( a _ { t } \odot \hat { \kappa } _ { t } ) = \left ( I - \hat { \kappa } _ { t } ^ { T } ( \frac { a _ { t } } { w _ { t } } \odot \hat { \kappa } _ { t } ) \right ) \text {diag} ( w _ { t } ) \approx \left ( I - 2 \hat { \kappa } _ { t } ^ { T } \hat { \kappa } _ { t } \right ) \text {diag} ( w _ { t } ) \quad ( 1 9 )$$

is no longer a Householder matrix but a scaled approximation of it, as ˆ 𝜅$\_{𝑡}$ ̸ = 𝑎$\_{𝑡}$$\_{𝐿}$$\_{𝑤}$$\_{𝑡}$$\_{𝑘}$ . This mimics a Householder matrix but with expanded dynamics, while still having all eigenvalues in a stable range of [ - 1 , 1] and allows the network to decay information in all subspaces if necessary. It contrasts with the case of a Householder-like matrix with learning rate ( 𝐼 - 𝑎 𝑉 $^{𝑇}$𝑣 ) , 𝑎 ∈ [ 0 , 1 ] , as used in Schlag et al. (2021); Yang et al. (2024c) where all eigenvalues are one except for the last one corresponding to 1 - 𝑎 . Given these properties, we refer to 𝑤$\_{𝑡}$ as "in-context weight decay" and to 𝑎$\_{𝑡}$ as "in-context learning rate" (ICLR). The RWKV-7 transition matrix, therefore, allows for both dynamic state evolution and approximation to a forget gate at the same time. See Appendix C for the details on the eigenvalue of the transition matrix, and when the transition matrix is guaranteed to be stable.

The original delta rule in Schlag et al. (2021) allows partial or full removal of pre-existing values from the state at each time-step, with the amount removed being equal to the scalar 𝑎 . Our formulation extends this ability by making 𝑎 a vector, allowing for different removal amount per state column.

WKV Bonus and Output All operations in this section are shown per-head unless otherwise specified.

Receptance, which acts like the query found in transformers, is applied to the WKV state, and the result is normalized. An added bonus, the amount of which is weighted by 𝜌 , allows the model to place extra attention on the current shifted input token without requiring it to store that token in the state.

$$u _ { t } = \left ( r _ { t } \cdot ( \rho \odot \tilde { k } _ { t } ) ^ { T } \right ) v _ { t }$$

$$p _ { t } = \text {LayerNorm} ( r _ { t } w k v _ { t } ^ { T } ) + u _ { t }$$

Finally, the heads are recombined via reshaping so that 𝜌$\_{𝑡}$ ∈ ℝ $^{𝐷}$, gated, and transformed into the output as follows:

$$o _ { t } = ( g _ { t } \odot p _ { t } ) W _ { o } \in \mathbb { R } ^ { D }$$

## 4.2 MLP

The MLP module of RWKV-7 is no longer identical to the Channel Mixing module of previous RWKV-4,5,6 architectures (Peng et al., 2024b). We remove the gating matrix 𝑊$\_{𝑟}$ , making it a two-layer MLP. In compensation for the removed gating parameters to satisfy the equi-parameter condition, we set the hidden dimension to be 4 times the size of model dimension.

$$k ^ { \prime } _ { t } = \text {lerp} ( x ^ { \prime } _ { t } , x ^ { \prime } _ { t - 1 } , \mu ^ { \prime } _ { k } ) W _ { k ^ { \prime } } \in \mathbb { R } ^ { 4 D }$$

$$o ^ { \prime } _ { t } = \text {ReLU} ( k ^ { \prime } _ { t } ) ^ { 2 } W _ { v ^ { \prime } } \in \mathbb { R } ^ { D }$$

## 5 RWKV World v3 Dataset

We train our models on the new RWKV World v3 Dataset , a new multilingual 3.119 trillion token dataset drawn from a wide variety of publicly available data sources. This dataset aims to help close the gap with the amount of data used to train modern LLMs, which may consume as many as 15 - 18 trillion tokens (Qwen et al., 2025; Grattafiori et al., 2024). We select the data to approximate the distribution of our previous World datasets, including English, multilingual, and code, while slightly enhancing Chinese novels.We describe the composition of our dataset in Appendix B.

## 6 Pre-Trained Models

We have pre-trained and publicly released seven Apache 2.0 licensed RWKV-7 models:

- 1. Trained on Pile: RWKV7-Pile of sizes 0.1B, 0.4B, and 1.4B
- 2. Trained on RWKV World V3: RWKV7-World-3 of sizes 0.1B, 0.4B, 1.5B, and 2.9B

See Appendix E for detailed configurations.

The RWKV-7 Pile models all use the GPT-NeoX-20B tokenizer (Black et al., 2022), and were all trained from scratch on the Pile dataset, which has 332 billion tokens.

All RWKV World dataset models use the RWKV World Tokenizer. Due to compute budget constraints, the Goose World 3 0.1B and 0.4B models were trained from pre-existing RWKV-5 World v1 and v2 checkpoints, and the Goose World 3 1.5B and 2.9B models were trained from pre-existing RWKV-6 World v2.1 checkpoints. These checkpoints' parameters were then converted to the RWKV-7 format via a process described below. Once in the new format, the models are trained on either the additional full 3.1 trillion tokens of the World v3 corpus, or an equally weighted sub-sampling of it. Under this methodology, some documents were seen two or even three times.

The World v1, v2, v2.1, and v3 corpora contain 0.6, 1.1, 1.4, and 3.1 trillion tokens, respectively. The amounts of training in each stage at with each successive model architecture and corpus are shown in Table 2.

Table 2: Total trillions of tokens trained for all RWKV-7 World 3 models

| Model             | World v1     | World v2     | World v2.1   | World v3     | Total        |
|-------------------|--------------|--------------|--------------|--------------|--------------|
| RWKV7-World3-0.1B | 0.6 (RWKV-5) |              |              | 1.0 (RWKV-7) | 1.6          |
| RWKV7-World3-0.4B |              | 1.1 (RWKV-5) |              | 2.0 (RWKV-7) | 3.1          |
| RWKV7-World3-1.5B |              | 1.1 (RWKV-6) |              | 1.4 (RWKV-6) | 5.6          |
| RWKV7-World3-2.9B |              | 1.1 (RWKV-6) |              | 1.4 (RWKV-6) | 3.1 (RWKV-7) |

Our model format conversion process involves removing the token-shift low-rank MLPs, rescaling by half the embeddings, wkv receptance, wkv output matrix weights, and Layernorm and Groupnorm bias values. Layernorm and Groupnorm weights are clamped above zero and square rooted. We widen the FFN MLP from 3.5x (in RWKV-6) to 4x and add new small (1 × 10 - $^{3}$) uniform initalizations in the new regions, removing the RWKV-6 FFN receptance weights. We widen the time decay Low-rank MLP and add new small (1 × 10 - $^{4}$) uniform initializations in the new regions. We replace the gate weights with a LoRA obtained through singular value decomposition and rescaling by half.

## 7 Language Modeling Experiments

## 7.1 LM Evaluation Harness Benchmarks

RWKV-7 models are evaluated on a series of common English-focused and multilingual benchmarks using LM Evaluation Harness (Gao et al., 2023) as shown in Tables 3 and 4. We benchmarked RWKV-7 along with several new open models which are state-of-the-art in their parameter count ranges. All numbers are evaluated under fp32 precision with lm-eval v0.4.8 using 0-shot, except for MMLU in which case 5-shot was used.

Figure 3: Model Comparisons across Multilingual Benchmarks

<!-- image -->

We find that RWKV-7 is generally able to match the English performance of Qwen2.5 (Qwen et al., 2025) with less than one third as many training tokens. Interestingly, we found that RWKV-7 models have shown giant leaps in MMLU performance compared to RWKV-6. We also find that RWKV-7-World models expand upon RWKV-6-World models already strong capabilities on multilingual benchmarks, outperforming SmolL2 (Alal et al., 2025), Llama 3.2 (Grattafiori et al., 2024), and Qwen-2.5 (Qwen et al., 2025) by a significant margin.

In Figures 3a and 4a we plot FLOPs used to train several open models versus average accuracy across the same sets of common english and multi-lingual benchmarks. The multilingual evals show a very dramatic Pareto improvement versus the transformer models. Also note the similar english-language eval scores, but dramatically lower total FLOPs usage of RWKV7-World models versus other highly trained open transformer models. We theorize that if we were less constrained by compute and were able to train these models from scratch with the same amount of total tokens instead of from pre-trained checkpoints of earlier RWKV versions, the difference would be even more dramatic. Note that we did not plot the Llama 3.2 series of models, as they have no corresponding FLOPs amounts due to having been created via pruning and distillation from larger models.

## 7.2 Recent Internet Data Evaluation

Modern large language models are trained on massive datasets. Despite careful data cleaning, benchmark data leakage remains a challenge, compromising the validity of these evaluations. To complement traditional benchmarks, we evaluated RWKV-7 Goose and other leading open-source models using temporally novel internet data, generated after the models' training periods; this data could not have appeared in the training sets, removing data leakage concerns.

Specifically, we collected new data created after January 2025, including: newly submitted computer science and physics papers on arXiv, newly created Python/C++ open-source repositories on GitHub, recently published Wikipedia entries, new fiction on Archive of Our Own (Various, 2025), and recent news articles. Inspired by Delétang et al. (2024); Li et al. (2024b), we used compression rate as our evaluation metric. See Table 5 for details.

Remarkably, despite being trained on significantly less data than other top models, RWKV-7 Goose showed competitive performance on this temporally novel data.