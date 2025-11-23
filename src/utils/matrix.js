/**
 * Basic matrix and vector operations for embeddings
 * Pure JavaScript implementation for educational purposes
 */

export class Matrix {
  /**
   * Create a matrix with random values
   * @param {number} rows
   * @param {number} cols
   * @param {number} scale - scaling factor for random initialization
   */
  static random(rows, cols, scale = 0.1) {
    const data = [];
    for (let i = 0; i < rows; i++) {
      data[i] = [];
      for (let j = 0; j < cols; j++) {
        data[i][j] = (Math.random() - 0.5) * scale;
      }
    }
    return data;
  }

  /**
   * Create a zero matrix
   */
  static zeros(rows, cols) {
    return Array(rows).fill(0).map(() => Array(cols).fill(0));
  }

  /**
   * Dot product of two vectors
   */
  static dot(vec1, vec2) {
    if (vec1.length !== vec2.length) {
      throw new Error('Vectors must have same length');
    }
    return vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
  }

  /**
   * Calculate Euclidean norm (magnitude) of a vector
   */
  static norm(vec) {
    return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
  }

  /**
   * Cosine similarity between two vectors
   */
  static cosineSimilarity(vec1, vec2) {
    const dotProduct = Matrix.dot(vec1, vec2);
    const norm1 = Matrix.norm(vec1);
    const norm2 = Matrix.norm(vec2);

    if (norm1 === 0 || norm2 === 0) return 0;
    return dotProduct / (norm1 * norm2);
  }

  /**
   * Add two vectors
   */
  static add(vec1, vec2) {
    if (vec1.length !== vec2.length) {
      throw new Error('Vectors must have same length');
    }
    return vec1.map((val, i) => val + vec2[i]);
  }

  /**
   * Subtract two vectors (vec1 - vec2)
   */
  static subtract(vec1, vec2) {
    if (vec1.length !== vec2.length) {
      throw new Error('Vectors must have same length');
    }
    return vec1.map((val, i) => val - vec2[i]);
  }

  /**
   * Multiply vector by scalar
   */
  static scale(vec, scalar) {
    return vec.map(val => val * scalar);
  }

  /**
   * Normalize vector to unit length
   */
  static normalize(vec) {
    const magnitude = Matrix.norm(vec);
    if (magnitude === 0) return vec;
    return Matrix.scale(vec, 1 / magnitude);
  }
}
