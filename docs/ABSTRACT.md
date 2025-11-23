**ABSTRACT — Compressive Semantic Spectroscopy (CSS):

A Sparse Spectral Framework for Learning Word Meaning**

We propose Compressive Semantic Spectroscopy (CSS), a new paradigm for representing and learning word meaning. Unlike traditional dense embeddings that encode each word as a single point in a high-dimensional vector space, CSS represents each word as a sparse spectrum of semantic frequencies, where each active frequency corresponds to a distinct semantic mode or sense. Amplitudes determine the degree of activation, while complex phases encode relational orientation, analogy structure, and contextual behavior.

In CSS, contexts are not prediction targets but measurement operators: each linguistic context provides a partial, noisy observation of the word’s latent spectrum. A word appearing across many contexts yields a set of diverse “spectral measurements,” each probing different subsets of its semantic frequencies. Learning becomes a global sparse inverse problem, analogous to compressive sensing or tomographic reconstruction:

recover the sparsest spectral signature that jointly explains all contextual measurements of a word.

This reframing produces several foundational advantages.
First, polysemy emerges naturally: words used in multiple sense-clusters must activate distinct frequency bands to satisfy incompatible measurement constraints. Second, sparsity leads to high interpretability: spectral peaks correspond to transparent semantic factors or roles. Third, modeling composition as spectral interference allows linear, physically meaningful compositionality, where context acts as a multiplicative filter that amplifies sense-specific frequencies and suppresses irrelevant ones. Fourth, CSS offers a principled mechanism for negative evidence—contexts that fail to activate certain frequencies—and accommodates both local and long-range linguistic signals.

We explore several training strategies within this paradigm:

Direct spectral skip-gram, where each (word, context) pair provides a spectral compatibility measurement.

Context-as-filter models, where aggregated context spectra selectively probe word frequencies.

Distillation-based initialization, where existing dense embeddings provide coarse semantic structure that CSS refines and sparsifies.

Graph-free, end-to-end sparse reconstruction, combining gradient descent with spectral sparsity constraints to recover interpretable, multi-modal meaning signatures.

CSS introduces a mathematically grounded, interpretable, and sparse alternative to traditional word embeddings. It redefines distributional semantics as a signal reconstruction problem, opening a new research direction for lightweight semantic models, explicit polysemy modeling, and a deeper theoretical understanding of how linguistic meaning emerges from context.