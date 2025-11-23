1. Formal CSS Model
1.1 Vocabulary and frequency grid

Let the vocabulary be

𝑉
=
{
𝑤
1
,
…
,
𝑤
𝑉
}
V={w
1
	​

,…,w
V
	​

}

Fix a global semantic frequency grid of size 
𝑁
N:

Ω
=
{
𝜔
1
,
…
,
𝜔
𝑁
}
Ω={ω
1
	​

,…,ω
N
	​

}

Think of 
𝜔
𝑗
ω
j
	​

 as basis semantic modes shared by all words.

1.2 Word representation: sparse spectrum

Each word 
𝑤
∈
𝑉
w∈V has a complex spectral embedding:

𝑆
𝑤
∈
𝐶
𝑁
S
w
	​

∈C
N

Written component-wise:

𝑆
𝑤
=
(
𝑆
𝑤
(
𝜔
1
)
,
…
,
𝑆
𝑤
(
𝜔
𝑁
)
)
,
𝑆
𝑤
(
𝜔
𝑗
)
=
𝐴
𝑤
(
𝜔
𝑗
)
 
𝑒
𝑖
𝜙
𝑤
(
𝜔
𝑗
)
S
w
	​

=(S
w
	​

(ω
1
	​

),…,S
w
	​

(ω
N
	​

)),S
w
	​

(ω
j
	​

)=A
w
	​

(ω
j
	​

)e
iϕ
w
	​

(ω
j
	​

)

where

𝐴
𝑤
(
𝜔
𝑗
)
≥
0
A
w
	​

(ω
j
	​

)≥0 is the amplitude (strength of semantic mode 
𝜔
𝑗
ω
j
	​

),

𝜙
𝑤
(
𝜔
𝑗
)
∈
[
−
𝜋
,
𝜋
)
ϕ
w
	​

(ω
j
	​

)∈[−π,π) is the phase (relational orientation).

Sparsity constraint:
Each word uses at most 
𝐾
≪
𝑁
K≪N active frequencies.

Formally, define:

supp
(
𝑆
𝑤
)
=
{
𝜔
𝑗
∈
Ω
∣
∣
𝑆
𝑤
(
𝜔
𝑗
)
∣
>
0
}
supp(S
w
	​

)={ω
j
	​

∈Ω∣∣S
w
	​

(ω
j
	​

)∣>0}

with

∣
supp
(
𝑆
𝑤
)
∣
≤
𝐾
∣supp(S
w
	​

)∣≤K

This is the key: few frequencies per word.

1.3 Context representation

Let a context 
𝐶
C be a multiset (or sequence) of words:

𝐶
=
{
𝑐
1
,
…
,
𝑐
𝑚
}
,
𝑐
𝑖
∈
𝑉
C={c
1
	​

,…,c
m
	​

},c
i
	​

∈V

We define a context spectrum 
𝑆
𝐶
∈
𝐶
𝑁
S
C
	​

∈C
N
 by some aggregation function 
𝑔
g:

𝑆
𝐶
=
𝑔
(
𝑆
𝑐
1
,
…
,
𝑆
𝑐
𝑚
)
S
C
	​

=g(S
c
1
	​

	​

,…,S
c
m
	​

	​

)

The simplest choice (for clarity) is a weighted sum:

𝑆
𝐶
(
𝜔
𝑗
)
=
∑
𝑖
=
1
𝑚
𝛼
𝑖
 
𝑆
𝑐
𝑖
(
𝜔
𝑗
)
S
C
	​

(ω
j
	​

)=
i=1
∑
m
	​

α
i
	​

S
c
i
	​

	​

(ω
j
	​

)

where 
𝛼
𝑖
α
i
	​

 could encode position, distance, or other context weights (e.g., closer words get larger 
𝛼
𝑖
α
i
	​

).

More general forms are possible, but this is enough to formalize CSS.

1.4 Measurement viewpoint

CSS’s core idea:
Context acts as a measurement of a word’s spectrum.

Define a compatibility score between word 
𝑤
w and context 
𝐶
C as:

score
(
𝑤
,
𝐶
)
=
ℜ
(
∑
𝑗
=
1
𝑁
𝑆
𝑤
(
𝜔
𝑗
)
∗
 
𝑆
𝐶
(
𝜔
𝑗
)
)
=
ℜ
(
⟨
𝑆
𝑤
,
𝑆
𝐶
⟩
𝐶
𝑁
)
score(w,C)=ℜ(
j=1
∑
N
	​

S
w
	​

(ω
j
	​

)
∗
S
C
	​

(ω
j
	​

))=ℜ(⟨S
w
	​

,S
C
	​

⟩
C
N
	​

)

where 
∗
∗
 is complex conjugate.

Interpretation:

The context spectrum 
𝑆
𝐶
S
C
	​

 is a measurement pattern over frequencies.

The inner product measures how well the word’s spectrum resonates with that measurement.

Each occurrence of 
𝑤
w in a context 
𝐶
𝑡
C
t
	​

 gives one such scalar 
score
(
𝑤
,
𝐶
𝑡
)
score(w,C
t
	​

) that we try to make high versus negatives.

1.5 Training objective (high level)

Given corpus 
𝐷
D as word–context pairs 
(
𝑤
,
𝐶
)
(w,C):

Positive set:

𝑃
=
{
(
𝑤
,
𝐶
)
∣
𝑤
 appears in context 
𝐶
}
P={(w,C)∣w appears in context C}

Negative samples:
For each positive pair 
(
𝑤
,
𝐶
)
(w,C), draw negative words 
𝑛
n from a noise distribution 
𝑃
neg
(
𝑛
)
P
neg
	​

(n).

Define a contrastive loss (spectral skip-gram style):

𝐿
data
=
−
∑
(
𝑤
,
𝐶
)
∈
𝑃
[
log
⁡
𝜎
(
score
(
𝑤
,
𝐶
)
)
+
∑
𝑛
∼
𝑃
neg
log
⁡
𝜎
(
−
score
(
𝑛
,
𝐶
)
)
]
L
data
	​

=−
(w,C)∈P
∑
	​

	​

logσ(score(w,C))+
n∼P
neg
	​

∑
	​

logσ(−score(n,C))
	​


with 
𝜎
σ the sigmoid.

Add sparsity and norm regularization:

𝐿
sparsity
=
𝜆
1
∑
𝑤
∈
𝑉
∑
𝑗
=
1
𝑁
∣
𝑆
𝑤
(
𝜔
𝑗
)
∣
L
sparsity
	​

=λ
1
	​

w∈V
∑
	​

j=1
∑
N
	​

∣S
w
	​

(ω
j
	​

)∣
𝐿
norm
=
𝜆
2
∑
𝑤
∈
𝑉
(
∥
𝑆
𝑤
∥
2
2
−
𝑐
)
2
L
norm
	​

=λ
2
	​

w∈V
∑
	​

(∥S
w
	​

∥
2
2
	​

−c)
2

Total loss:

𝐿
=
𝐿
data
+
𝐿
sparsity
+
𝐿
norm
L=L
data
	​

+L
sparsity
	​

+L
norm
	​


This is the Compressive Semantic Spectroscopy perspective:

The data term enforces that spectra explain the observed contexts.

The sparsity term enforces few frequencies per word.

The norm term controls total spectral power per word.

In a more “compressive sensing” framing, for each word 
𝑤
w, all its contexts 
{
𝐶
𝑡
}
𝑡
=
1
𝑇
𝑤
{C
t
	​

}
t=1
T
w
	​

	​

 induce many measurement equations that jointly constrain the same sparse vector 
𝑆
𝑤
S
w
	​

.

2. Core Operations in CSS
2.1 Word similarity

Given two words 
𝑤
,
𝑢
w,u:

sim
(
𝑤
,
𝑢
)
=
ℜ
(
⟨
𝑆
𝑤
,
𝑆
𝑢
⟩
∥
𝑆
𝑤
∥
 
∥
𝑆
𝑢
∥
)
sim(w,u)=ℜ(
∥S
w
	​

∥∥S
u
	​

∥
⟨S
w
	​

,S
u
	​

⟩
	​

)

This is a complex cosine similarity (taking the real part), sensitive to both amplitudes and phases.

If we ignore phases (or use magnitude spectrum):

sim
mag
(
𝑤
,
𝑢
)
=
∑
𝑗
𝐴
𝑤
(
𝜔
𝑗
)
𝐴
𝑢
(
𝜔
𝑗
)
∥
𝐴
𝑤
∥
2
 
∥
𝐴
𝑢
∥
2
sim
mag
	​

(w,u)=
∥A
w
	​

∥
2
	​

∥A
u
	​

∥
2
	​

∑
j
	​

A
w
	​

(ω
j
	​

)A
u
	​

(ω
j
	​

)
	​

2.2 Composition (phrase/sentence meaning)

A phrase/sentence can be represented as the context spectrum:

𝑆
phrase
=
𝑔
(
𝑆
𝑤
1
,
…
,
𝑆
𝑤
𝑚
)
S
phrase
	​

=g(S
w
1
	​

	​

,…,S
w
m
	​

	​

)

Using sum as before:

𝑆
phrase
(
𝜔
𝑗
)
=
∑
𝑖
=
1
𝑚
𝛼
𝑖
𝑆
𝑤
𝑖
(
𝜔
𝑗
)
S
phrase
	​

(ω
j
	​

)=
i=1
∑
m
	​

α
i
	​

S
w
i
	​

	​

(ω
j
	​

)

Interpretation:

Frequencies shared by multiple words are amplified (constructive interference).

Incompatible phases can cause partial cancellation (destructive interference).

This acts like a spectral blend of meanings.

2.3 Contextualization of a single word

Given a target word 
𝑤
w and context 
𝐶
C:

Define the context-filtered spectrum:

𝑆
~
𝑤
(
𝐶
)
(
𝜔
𝑗
)
=
𝑆
𝑤
(
𝜔
𝑗
)
⋅
ℎ
(
𝑆
𝐶
(
𝜔
𝑗
)
)
S
~
w
(C)
	​

(ω
j
	​

)=S
w
	​

(ω
j
	​

)⋅h(S
C
	​

(ω
j
	​

))

where 
ℎ
h is some filtering function, e.g.:

multiplicative filter: 
ℎ
(
𝑧
)
=
𝑧
h(z)=z

or normalized gating: 
ℎ
(
𝑧
)
=
𝜎
(
𝛽
∣
𝑧
∣
)
h(z)=σ(β∣z∣) acting on amplitude

This expresses:

Context selects and scales which frequencies of the word are active.

2.4 Analogy & relations via phase shifts

If a relation 
𝑅
R corresponds to a phase shift pattern 
Δ
𝑅
(
𝜔
𝑗
)
Δ
R
	​

(ω
j
	​

):

𝑆
𝑤
:
𝐵
(
𝑅
)
(
𝜔
𝑗
)
=
𝑆
𝑤
(
𝜔
𝑗
)
⋅
𝑒
𝑖
Δ
𝑅
(
𝜔
𝑗
)
S
w:B
(R)
	​

(ω
j
	​

)=S
w
	​

(ω
j
	​

)⋅e
iΔ
R
	​

(ω
j
	​

)

Then an analogy like:

𝑤
1
:
𝑤
2
:
:
𝑤
3
:
?
w
1
	​

:w
2
	​

::w
3
	​

:?

would try to find 
𝑤
4
w
4
	​

 such that:

𝑆
𝑤
2
≈
𝑆
𝑤
1
(
𝑅
)
and
𝑆
𝑤
4
≈
𝑆
𝑤
3
(
𝑅
)
S
w
2
	​

	​

≈S
w
1
	​

(R)
	​

andS
w
4
	​

	​

≈S
w
3
	​

(R)
	​


with 
𝑅
R inferred from the phase difference between 
𝑆
𝑤
1
S
w
1
	​

	​

 and 
𝑆
𝑤
2
S
w
2
	​

	​

.

This is more speculative but fits nicely in the complex representation.

3. Parallels with Other Embeddings
3.1 Classical dense embeddings (word2vec, GloVe)

Representations:

𝑣
𝑤
∈
𝑅
𝑑
v
w
	​

∈R
d

Learned by:

skip-gram / CBOW (predict context),

or factorizing co-occurrence matrices.

Parallel:

CSS’s spectral vectors 
𝑆
𝑤
∈
𝐶
𝑁
S
w
	​

∈C
N
 play the role of 
𝑣
𝑤
v
w
	​

,

but with structure:

complex-valued,

sparsity,

frequency semantics.

Key differences:

Dense vectors are typically fully dense, uninterpreted axes.

CSS is explicitly sparse, with axes interpreted as semantic frequencies.

Polysemy is implicit in dense embeddings; in CSS, it is explicitly multi-modal (multi-peak spectra).

3.2 Complex embeddings (e.g., ComplEx for knowledge graphs)

ComplEx represents entities/relations as complex vectors and scores triples by complex inner products.

Parallel:

CSS also uses complex vectors and complex inner products.

Phases encode relational structure.

Difference:

ComplEx is for triples (entity, relation, entity) in KGs.

CSS is for word meaning and contexts in distributional corpora.

CSS adds sparsity and frequency interpretation on top.

3.3 Graph-based embeddings (e.g., DeepWalk, node2vec, LINE)

These methods:

Build a graph (nodes = words, edges = co-occurrence or relations),

Learn embeddings by random walks / edge sampling / matrix factorization.

Parallel:

CSS can be seen as learning embeddings from an implicit semantic graph (words + co-occurrence edges).

The data term in 
𝐿
data
L
data
	​

 is similar to sampling edges and non-edges.

Difference:

Graph embeddings are typically real, dense vectors.

CSS uses complex, sparse spectral codes with a measurement interpretation.

CSS focuses on recovering sparse spectra rather than arbitrary dense vectors.

3.4 Contextual LMs (ELMo, BERT, GPT)

These models:

Map each token occurrence to a context-dependent embedding.

Meaning is fully entangled with context through deep networks.

Parallel:

CSS also yields contextualized spectra via context filters: 
𝑆
~
𝑤
(
𝐶
)
S
~
w
(C)
	​

.

Both aim to model how meaning changes with context.

Difference:

LMs use large parametric neural networks and multi-layer attention.

CSS seeks a lower-level semantic representation where:

base word representations are sparse spectra,

contextualization is mostly spectral filtering and interference, not dozens of nonlinear layers.

CSS is more like a structured core semantics layer, onto which bigger models could be built.

4. Compute & Space Costs (Rough Comparison)

Let:

𝑉
V = vocabulary size

𝑑
d = dimension of standard embeddings

𝑁
N = total number of semantic frequencies in CSS

𝐾
K = max active frequencies per word in CSS (sparsity; 
𝐾
≪
𝑁
K≪N)

4.1 Space complexity

Dense embeddings (word2vec/GloVe):

Each word: 
𝑑
d floats

Total:

𝑂
(
𝑉
𝑑
)
O(Vd)

For example: 
𝑉
=
100
𝑘
,
𝑑
=
300
⇒
30
𝑀
V=100k,d=300⇒30M parameters.

CSS (sparse complex spectra):

Each word: at most 
𝐾
K non-zero complex coefficients
each coefficient = 2 real numbers (
Re
Re, 
Im
Im) + an index.

Ignoring index storage overhead:

Per word: 
∼
2
𝐾
∼2K real values

Total:

𝑂
(
𝑉
𝐾
)
O(VK)

If 
𝐾
≪
𝑑
K≪d, you can be more memory-efficient than dense embeddings.

Example:

𝑉
=
100
𝑘
V=100k

𝑁
=
1024
N=1024 frequencies

𝐾
=
16
K=16 active frequencies/word
→ params ≈ 
100
𝑘
×
16
×
2
=
3.2
𝑀
100k×16×2=3.2M real values
vs 30M in a 300-d dense model.

Even if you add overhead for indices, you can still be competitive.

4.2 Per-update compute (training)

Assume a skip-gram-like update with:

center word 
𝑤
w,

one context 
𝐶
C,

𝑚
m context words in 
𝐶
C,

𝑘
k negative samples.

Dense embeddings:

Each update uses dot products of size 
𝑑
d:
cost ≈ 
𝑂
(
𝑑
(
𝑚
+
𝑘
)
)
O(d(m+k)).

CSS (sparse):

Key point: all spectra are sparse with at most 
𝐾
K active frequencies. So:

The context spectrum 
𝑆
𝐶
S
C
	​

 has at most 
≤
𝑚
𝐾
≤mK active entries (in practice often much fewer due to overlapping supports).

The word spectrum 
𝑆
𝑤
S
w
	​

 has at most 
𝐾
K entries.

Computing score(w, C):

Only frequencies in 
supp
(
𝑆
𝑤
)
∩
supp
(
𝑆
𝐶
)
supp(S
w
	​

)∩supp(S
C
	​

) matter.

Worst case (no overlap): naive: 
𝑂
(
𝐾
⋅
𝑚
𝐾
)
O(K⋅mK), but you’d implement this with hash / index intersection → cost ≈ 
𝑂
(
∣
supp
(
𝑆
𝑤
)
∩
supp
(
𝑆
𝐶
)
∣
)
O(∣supp(S
w
	​

)∩supp(S
C
	​

)∣). Typically much less than 
𝐾
⋅
𝑚
K⋅m.

Realistically:

If frequencies are shared and structured, per-update cost:

𝑂
(
𝐾
(
𝑚
+
𝑘
)
)
O(K(m+k))

with 
𝐾
≪
𝑑
K≪d.

So compute can be significantly cheaper than dense embeddings for similar quality, especially if you keep 
𝐾
K small and caches of sparse indices efficient.

4.3 Overheads

CSS adds:

sparsity enforcement:

top-K pruning or L1 gradient steps (cheap per word),

potential extra bookkeeping (indices of active frequencies).

But saves:

full dense operations on dimension 
𝑑
=
𝑁
d=N,

model size.

Overall, you trade some bookkeeping and model simplicity for:

smaller effective dimensionality per word (K),

potential interpretability,

explicit multi-sense representation.

5. Short Summary

Representation:
CSS replaces dense vectors with sparse complex spectra over shared semantic frequencies.

Training:
Words are learned as sparse spectra that best explain many contextual “measurements”, using a contrastive objective plus sparsity constraints. This is analogous to compressive sensing.

Operations:

Similarity = complex spectral cosine.

Composition = spectral summation/interference.

Contextualization = frequency-wise filtering.

Analogies = phase shifts across frequencies.

Parallel to existing methods:
CSS is structurally related to word2vec, graph embeddings, and complex embeddings, but adds:

sparsity,

explicit frequencies,

measurement-based perspective,

natural polysemy through multi-peak spectra.

Costs:

Space ≈ 
𝑂
(
𝑉
𝐾
)
O(VK) vs 
𝑂
(
𝑉
𝑑
)
O(Vd).

Compute per update ≈ 
𝑂
(
𝐾
(
𝑚
+
𝑘
)
)
O(K(m+k)) vs 
𝑂
(
𝑑
(
𝑚
+
𝑘
)
)
O(d(m+k)).

With 
𝐾
≪
𝑑
K≪d, CSS can be lighter and more interpretable.

If you’d like, next we can:

formalize a concrete CSS training algorithm step-by-step (still math-level, not code),

or define a minimal experimental setup (tiny vocab + toy corpus) to illustrate how spectra and polysemy actually look.