/**
 * ReLU (Rectified Linear Unit) activation function
 * @param {number} x - Input value
 * @returns {number} max(0, x)
 */
export function relu(x) {
    return Math.max(0, x);
}

/**
 * Softmax activation function
 * Converts raw scores (logits) to probabilities that sum to 1
 * @param {Array} arr - Array of logits
 * @returns {Array} Array of probabilities
 */
export function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max)); // Subtract max for numerical stability
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(x => x / sum);
}

/**
 * Cross-entropy loss function
 * Measures the difference between predicted probabilities and true labels
 * @param {Array} output - Predicted probabilities (from softmax)
 * @param {Array} target - True labels (one-hot encoded)
 * @returns {number} Loss value
 */
export function crossEntropyLoss(output, target) {
    let loss = 0;
    const numberOfClasses = output.length;

    for (let i = 0; i < numberOfClasses; i++) {
        const targetProbability = target[i];
        const predictedProbability = output[i];

        // Avoiding log(0) which is undefined
        if (predictedProbability > 0) {
            loss -= targetProbability * Math.log(predictedProbability);
        }
    }

    return loss;
}

/**
 * Forward propagation through a neural network layer
 * Computes weighted sum: z = w * x (without activation)
 * @param {Array} inputActivations - Activations from the previous layer
 * @param {Array} layerWeights - Weight matrix [numNeurons][numInputs]
 * @returns {Array} Weighted sums for this layer
 */
export function forwardLayer(inputActivations, layerWeights) {
    const numInputFeatures = inputActivations.length;
    const numNeuronsInLayer = layerWeights.length;
    const outputActivations = [];

    for (let neuronIndex = 0; neuronIndex < numNeuronsInLayer; neuronIndex++) {
        let weightedSum = 0;

        for (let inputIndex = 0; inputIndex < numInputFeatures; inputIndex++) {
            const inputValue = inputActivations[inputIndex];
            const weight = layerWeights[neuronIndex][inputIndex];
            weightedSum += inputValue * weight;
        }

        outputActivations[neuronIndex] = weightedSum;
    }

    return outputActivations;
}

/**
 * Initialize weights using He initialization
 * Good for ReLU activation functions
 * @param {number} numInputs - Number of inputs to each neuron
 * @param {number} numNeurons - Number of neurons in the layer
 * @returns {Array} Weight matrix [numNeurons][numInputs]
 */
export function initializeWeights(numInputs, numNeurons) {
    const weights = [];
    // He initialization: std = sqrt(2 / numInputs)
    const std = Math.sqrt(2.0 / numInputs);

    for (let i = 0; i < numNeurons; i++) {
        weights[i] = [];
        for (let j = 0; j < numInputs; j++) {
            // Box-Muller transform for normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            weights[i][j] = z * std;
        }
    }
    return weights;
}

/**
 * Creates a matrix filled with zeros
 * @param {number} rows - Number of rows (neurons)
 * @param {number} cols - Number of columns (inputs to each neuron)
 * @returns {Array} A 2D array filled with zeros
 */
export function createZeroMatrix(rows, cols) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
        matrix[i] = new Array(cols).fill(0);
    }
    return matrix;
}
