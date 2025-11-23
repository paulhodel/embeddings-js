🧠 Training Instructions for CSS (High-Level)
1. Initialize word spectra

Build your vocabulary (list of words).

For each word:

Assign a sparse random spectrum:

Choose K random frequency indices out of N.

Give each chosen frequency a small random amplitude + phase.

All other frequencies are implicitly 0 (inactive).

At this stage, every word just has a random “semantic spectrum” with a few random frequencies. No meaning yet.

2. Process text as sequences of words

You have a corpus, like:

... the river flows by the bank at night ...

You will slide over this text and, at each position, treat one word as the target and its neighbors as the context.

Pick a target position t in the sentence:

Target word = w_t

Choose a window size, e.g. 2–5 words on each side.

Context words = w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k}
(within sentence boundaries)

You now have one training example:
target word + adjacent context words.

3. Build a context signal (context spectrum)

You now turn the context words into a single context signal in frequency space.

For each context word c in the window:

Get its current spectrum S_c (the sparse object with its active frequencies).

Combine these spectra into a context spectrum S_context:

Simplest version: sum them up (optionally with weights for distance):

For each active frequency in each S_c, add it to S_context at that frequency.

The result is another sparse spectrum:

Frequencies that appear in several context words will have higher amplitudes.

Others might appear only once or not at all.

Intuition: the context spectrum is the combined semantic “signal” of the surrounding words.

4. Compare target word spectrum with context spectrum

Now you “shine” the context signal on the target word.

Take the target word’s spectrum S_target (for w_t).

Compute a compatibility score between S_target and S_context:

Conceptually: a dot product in frequency space.

Only overlapping frequencies contribute:

If both have energy at frequency f, that frequency contributes to the score.

If target has frequency f but context doesn’t, or vice versa, it contributes nothing.

Intuition:

If the target word’s frequencies align well with the context frequencies, the score is high.

If they don’t overlap or phases disagree, the score is low.

5. Add negative examples (contrast)

To avoid everything aligning with everything, you add negative words:

Randomly pick a few noise words from the vocabulary (words not actually in this context).

For each negative word n:

Take its spectrum S_n.

Compute its compatibility score with the same S_context.

You now have:

1 positive score: target vs context (should be high)

k negative scores: random word vs context (should be low)

6. Adjust spectra (the learning step)

Now comes the key part: update the spectra so the model behaves how we want.

For the target word:

If its score with context is too low:

Increase amplitudes on frequencies where it overlaps with the context.

Slightly adjust phases to better align with the context’s phases.

Over time, the target word’s spectrum gets pulled toward the kinds of frequencies that occur in its real contexts.

For negative words:

If a negative word scores too high with the context:

Decrease amplitudes on overlapping frequencies.

Adjust phases so they are less aligned.

This pushes unrelated words away from contexts they don’t belong to.

At the same time, apply sparsity pressure:

Penalize small amplitudes.

After some steps, drop tiny frequencies (set to zero).

Keep only the top K strongest frequencies per word.

Intuition:

Real contexts reinforce the correct semantic frequencies of a word.

Random negatives suppress frequencies that would wrongly associate it with unrelated contexts.

Sparsity keeps only the most stable and useful frequencies.

7. Repeat over the whole corpus (many times)

You do this again and again for many sentences / epochs:

Pick a target word.

Build its context spectrum.

Compare target vs context (and vs negatives).

Update spectra.

Enforce sparsity and norm constraints.

With time:

Frequencies that never help with real contexts → amplitudes shrink → get pruned (they “disappear”).

Frequencies that consistently align with the contexts of a word → amplitudes grow and become stable.

Polysemy emerges as multiple stable peaks for words that appear in very different context types.

Final view: each word’s spectrum is the compressed sum of all its semantic relationships, discovered through many small adjustments.

8. What you end up with

After training:

Each word has a sparse spectrum that:

concentrates energy on a few meaningful frequencies,

encodes its various senses,

lines up with the kinds of contexts it appears in.

Contexts act as filters:

When you compute word vs context again, only the relevant sense (frequencies) “light up”.