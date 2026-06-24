module ml

import math
import rand

// Sigmoid activation and its derivative
fn sigmoid(x f64) f64 {
    // s($x) -> 1.0 / (1.0 + :e(-$x)
    return 1.0 / (1.0 + math.exp(-x))
}

fn sigmoid_deriv(y f64) f64 {
    // d($y) -> $y * (1.0 - $y)
    return y * (1.0 - y)
}

// Simple Layer
pub struct Layer {
pub mut:
    weights [][]f64
    biases  []f64
    neurons int
}

// NN with one hidden layer
pub struct NN {
pub mut:
    inputs        int
    hidden        Layer
    output_layer  Layer
    learning_rate f64 = 0.5
}

// Create a new neural network
pub fn new_nn(inputs int, hidden_neurons int, outputs int) NN {
    mut nn := NN{
        inputs: inputs
    }

    nn.hidden = Layer{
        neurons: hidden_neurons
        weights: make_weights(hidden_neurons, inputs)
        biases:  make_biases(hidden_neurons)
    }

    nn.output_layer = Layer{
        neurons: outputs
        weights: make_weights(outputs, hidden_neurons)
        biases:  make_biases(outputs)
    }

    return nn
}

fn make_weights(rows int, cols int) [][]f64 {
    mut w := [][]f64{len: rows}
    for i in 0 .. rows {
        w[i] = []f64{len: cols}
        for j in 0 .. cols {
            // Xavier initialization: https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
            w[i][j] = (rand.f64() - 0.5) * 2.0 / math.sqrt(cols)
        }
    }
    return w
}

fn make_biases(size int) []f64 {
    mut b := []f64{len: size}
    for i in 0 .. size {
        b[i] = (rand.f64() - 0.5) * 0.1
    }
    return b
}

// Forward pass
fn (l Layer) forward(inputs []f64) ([]f64, []f64) {
    mut activations := []f64{len: l.neurons}
    mut z_values := []f64{len: l.neurons}

    for i in 0 .. l.neurons {
        mut sum := l.biases[i]
        for j in 0 .. inputs.len {
            sum += inputs[j] * l.weights[i][j]
        }
        z_values[i] = sum
        activations[i] = sigmoid(sum)
    }
    return activations, z_values
}

// Predict
pub fn (mut nn NN) predict(input []f64) []f64 {
    if input.len != nn.inputs {
        return []
    }

    hidden_act, _ := nn.hidden.forward(input)
    output_act, _ := nn.output_layer.forward(hidden_act)

    return output_act
}

// Train single example (online learning)
pub fn (mut nn NN) train(input []f64, target []f64) {
    if input.len != nn.inputs || target.len != nn.output_layer.neurons {
        return
    }

    // Forward
    hidden_act, _ := nn.hidden.forward(input)
    output_act, _ := nn.output_layer.forward(hidden_act)

    // Output deltas
    mut output_deltas := []f64{len: nn.output_layer.neurons}
    for i in 0 .. nn.output_layer.neurons {
        output_deltas[i] = (target[i] - output_act[i]) * sigmoid_deriv(output_act[i])
    }

    // Update output layer
    for i in 0 .. nn.output_layer.neurons {
        for j in 0 .. nn.hidden.neurons {
            nn.output_layer.weights[i][j] += nn.learning_rate * output_deltas[i] * hidden_act[j]
        }
        nn.output_layer.biases[i] += nn.learning_rate * output_deltas[i]
    }

    // Hidden deltas
    mut hidden_deltas := []f64{len: nn.hidden.neurons}
    for j in 0 .. nn.hidden.neurons {
        mut err := 0.0
        for i in 0 .. nn.output_layer.neurons {
            err += output_deltas[i] * nn.output_layer.weights[i][j]
        }
        hidden_deltas[j] = err * sigmoid_deriv(hidden_act[j])
    }

    // Update hidden layer
    for j in 0 .. nn.hidden.neurons {
        for i in 0 .. nn.inputs {
            nn.hidden.weights[j][i] += nn.learning_rate * hidden_deltas[j] * input[i]
        }
        nn.hidden.biases[j] += nn.learning_rate * hidden_deltas[j]
    }
}

// Train for multiple epochs
pub fn (mut nn NN) train_epochs(inputs [][]f64, targets [][]f64, epochs int) {
    for _ in 0 .. epochs {
        for i in 0 .. inputs.len {
            nn.train(inputs[i], targets[i])
        }
    }
}
