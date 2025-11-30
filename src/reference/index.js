/**
 * MNIST Neural Network - Training Script
 *
 * Architecture: 784 -> 16 -> 16 -> 10
 *
 * Features:
 *   ✓ Input normalization (z-score)
 *
 * TODO:
 *   - Bias terms
 *   - Batch normalization
 *   - Dropout regularization
 *   - Adam optimizer (currently using basic SGD)
 *   - Learning rate scheduling
 *   - Data augmentation
 *   - Proper training on full dataset
 */

import fs from 'fs';
import { readMNIST } from './images.js';
import {
    relu,
    softmax,
    crossEntropyLoss,
    forwardLayer,
    initializeWeights,
    createZeroMatrix
} from './utils.js';

// Hyperparameters
const learningRate = 0.01;

// Normalization statistics (calculated from MNIST training set)
// MNIST pixel values are in [0, 1] after division by 255
const PIXEL_MEAN = 0.1307; // Mean pixel value for MNIST
const PIXEL_STD = 0.3081;  // Standard deviation for MNIST

/**
 * Normalize pixel values using z-score normalization
 * Formula: (x - mean) / std
 * This centers the data around 0 and scales to unit variance
 */
function normalizePixels(pixels) {
    return pixels.map(pixel => (pixel - PIXEL_MEAN) / PIXEL_STD);
}

function loadWeights() {
    let w = {};
    let exists = fs.existsSync('./w.json');
    if (exists) {
        w = fs.readFileSync('./w.json');
        w = JSON.parse(w);
    } else {
        w = {
            // Input layer
            0: initializeWeights(784, 16),
            // First layer
            1: initializeWeights(16, 16),
            // Last layer
            2: initializeWeights(16, 10),
        }
    }
    return w;
}

function run(w, position) {
    // Load batch of images
    let images = readMNIST(position*10, position*10+10);

    // Accumulated weight gradients for the entire batch
    // Each matrix has shape [numNeurons][numInputs]
    let weightGradients = {
        0: createZeroMatrix(16, 784),  // Layer 0: 16 neurons, 784 inputs (pixels)
        1: createZeroMatrix(16, 16),   // Layer 1: 16 neurons, 16 inputs
        2: createZeroMatrix(10, 16)    // Layer 2: 10 neurons, 16 inputs
    };

    let totalLoss = 0;
    let correctPredictions = 0;

    // Process each image in the batch
    images.forEach(function (image) {
        // Create one-hot encoded target
        let target = new Array(10).fill(0);
        target[image.label] = 1;

        // ========== NORMALIZATION ==========
        // Normalize input pixels for better training stability
        const normalizedPixels = normalizePixels(image.pixels);

        // ========== FORWARD PASS - Store all activations ==========

        // Layer 0: Input pixels (784) -> Hidden layer 1 (16 neurons) with ReLU
        let z0 = forwardLayer(normalizedPixels, w[0]);  // Weighted sums before activation
        let a0 = z0.map(z => relu(z));  // Activations after ReLU

        // Layer 1: Hidden layer 1 (16) -> Hidden layer 2 (16 neurons) with ReLU
        let z1 = forwardLayer(a0, w[1]);
        let a1 = z1.map(z => relu(z));

        // Layer 2: Hidden layer 2 (16) -> Output layer (10 neurons) no activation
        let z2 = forwardLayer(a1, w[2]);
        let output = softmax(z2);

        // Calculate loss
        totalLoss += crossEntropyLoss(output, target);

        // Check if prediction is correct
        const predictedLabel = output.indexOf(Math.max(...output));
        if (predictedLabel === image.label) {
            correctPredictions++;
        }

        // ========== BACKWARD PASS - Calculate gradients ==========

        // Output layer gradient: dL/dz2 = output - target (derivative of softmax + cross-entropy)
        let dz2 = output.map((prob, i) => prob - target[i]);

        // Accumulate weight gradients for layer 2: dL/dw2 = dz2 * a1^T
        for (let i = 0; i < w[2].length; i++) {  // For each output neuron (10)
            for (let j = 0; j < w[2][i].length; j++) {  // For each input from previous layer (16)
                weightGradients[2][i][j] += dz2[i] * a1[j];
            }
        }

        // Backpropagate to hidden layer 2: dL/da1 = w[2]^T * dz2
        let da1 = new Array(a1.length).fill(0);
        for (let i = 0; i < a1.length; i++) {
            for (let j = 0; j < dz2.length; j++) {
                da1[i] += w[2][j][i] * dz2[j];
            }
        }

        // Apply ReLU derivative: dL/dz1 = da1 * relu'(z1)
        // relu'(z) = 1 if z > 0, else 0
        let dz1 = da1.map((grad, i) => z1[i] > 0 ? grad : 0);

        // Accumulate weight gradients for layer 1: dL/dw1 = dz1 * a0^T
        for (let i = 0; i < w[1].length; i++) {  // For each neuron (16)
            for (let j = 0; j < w[1][i].length; j++) {  // For each input (16)
                weightGradients[1][i][j] += dz1[i] * a0[j];
            }
        }

        // Backpropagate to hidden layer 1: dL/da0 = w[1]^T * dz1
        let da0 = new Array(a0.length).fill(0);
        for (let i = 0; i < a0.length; i++) {
            for (let j = 0; j < dz1.length; j++) {
                da0[i] += w[1][j][i] * dz1[j];
            }
        }

        // Apply ReLU derivative: dL/dz0 = da0 * relu'(z0)
        let dz0 = da0.map((grad, i) => z0[i] > 0 ? grad : 0);

        // Accumulate weight gradients for layer 0: dL/dw0 = dz0 * input^T
        for (let i = 0; i < w[0].length; i++) {  // For each neuron (16)
            for (let j = 0; j < w[0][i].length; j++) {  // For each input pixel (784)
                weightGradients[0][i][j] += dz0[i] * normalizedPixels[j];
            }
        }
    });

    // ========== UPDATE WEIGHTS ==========
    // Average gradients across batch and apply learning rate
    const batchSize = images.length;

    for (let layer = 0; layer <= 2; layer++) {
        for (let i = 0; i < w[layer].length; i++) {
            for (let j = 0; j < w[layer][i].length; j++) {
                const gradient = weightGradients[layer][i][j] / batchSize;
                w[layer][i][j] -= learningRate * gradient;
            }
        }
    }

    // Calculate accuracy for this batch
    const accuracy = (correctPredictions / batchSize) * 100;

    // Print batch results
    console.log(`Batch ${position} - Loss: ${(totalLoss / batchSize).toFixed(4)}, Accuracy: ${accuracy.toFixed(1)}% (${correctPredictions}/${batchSize})`);

    return { loss: totalLoss / batchSize, accuracy }; // Return metrics for tracking
}

function main() {
    // Load weights
    let w = loadWeights();

    const totalBatches = 1000;
    const reportInterval = 10; // Report progress every 10 batches
    let recentMetrics = [];

    console.log(`Starting training for ${totalBatches} batches...`);
    console.log('=' .repeat(60));

    for (let i = 0; i < totalBatches; i++) {
        const metrics = run(w, i);
        recentMetrics.push(metrics);

        // Print summary every reportInterval batches
        if ((i + 1) % reportInterval === 0) {
            const avgLoss = recentMetrics.reduce((a, b) => a + b.loss, 0) / recentMetrics.length;
            const avgAccuracy = recentMetrics.reduce((a, b) => a + b.accuracy, 0) / recentMetrics.length;
            const minLoss = Math.min(...recentMetrics.map(m => m.loss));
            const maxLoss = Math.max(...recentMetrics.map(m => m.loss));
            const progress = ((i + 1) / totalBatches * 100).toFixed(1);

            console.log('=' .repeat(60));
            console.log(`Progress: ${i + 1}/${totalBatches} (${progress}%)`);
            console.log(`Last ${reportInterval} batches - Avg Loss: ${avgLoss.toFixed(4)}, Avg Accuracy: ${avgAccuracy.toFixed(1)}%`);
            console.log(`Loss Range: ${minLoss.toFixed(4)} - ${maxLoss.toFixed(4)}`);
            console.log('=' .repeat(60));

            recentMetrics = []; // Reset for next interval
        }
    }

    console.log('\nTraining complete! Saving weights...');
    fs.writeFileSync('./w.json', JSON.stringify(w, null, 4));
    console.log('Weights saved to w.json');
    console.log('\nRun "node test.js" to test the trained network.');
}

main();
