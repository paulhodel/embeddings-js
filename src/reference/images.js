import fs from 'fs';
import { createCanvas } from 'canvas';

export function readMNIST(start, end) {
    try {
        const dataFileBuffer = fs.readFileSync('./train-images.idx3-ubyte');
        const labelFileBuffer = fs.readFileSync('./train-labels.idx1-ubyte');
        const pixelValues = [];

        for (let image = start; image < end && image < 60000; image++) {
            const pixels = [];
            for (let y = 0; y < 28; y++) {
                for (let x = 0; x < 28; x++) {
                    pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16] / 255);
                }
            }
            const imageData = {
                index: image,
                label: labelFileBuffer[image + 8],
                pixels: pixels
            };
            pixelValues.push(imageData);
        }
        return pixelValues;
    } catch (error) {
        console.error('Error reading MNIST files:', error.message);
        return [];
    }
}

export function saveMNIST(start, end) {
    try {
        if (!fs.existsSync('./images')) {
            fs.mkdirSync('./images');
        }
        const canvas = createCanvas(28, 28);
        const ctx = canvas.getContext('2d');
        const pixelValues = readMNIST(start, end);

        pixelValues.forEach(image => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (let y = 0; y < 28; y++) {
                for (let x = 0; x < 28; x++) {
                    const pixel = image.pixels[x + (y * 28)] * 255; // Scale back to [0, 255]
                    ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
                    ctx.fillRect(x, y, 1, 1);
                }
            }
            const buffer = canvas.toBuffer('image/png');
            fs.writeFileSync(`./images/image${image.index}-${image.label}.png`, buffer);
        });
    } catch (error) {
        console.error('Error saving MNIST images:', error.message);
    }
}