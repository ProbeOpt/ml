module main

import math
import rand

// ------------------------------------------------------------------
// Core Math & Activation Functions (Public)
// ------------------------------------------------------------------

pub fn sigmoid(value f64) f64 {
    if value < -500.0 { return 0.0 }
    return 1.0 / (1.0 + math.exp(-value))
}

pub fn sigmoid_derivative(value f64) f64 {
    s := sigmoid(value)
    return s * (1.0 - s)
}

pub fn relu(value f64) f64 {
    return math.max(0.0, value)
}

pub fn relu_derivative(value f64) f64 {
    if value > 0.0 { return 1.0 }
    return 0.0
}

// ------------------------------------------------------------------
// Data Normalizer
// ------------------------------------------------------------------

pub struct Normalizer {
pub mut:
    min_val f64
    max_val f64
}

pub fn new_normalizer(min_val f64, max_val f64) Normalizer {
    return Normalizer{
        min_val: min_val
        max_val: max_val
    }
}

pub fn (n Normalizer) normalize(value f64) f64 {
    range := n.max_val - n.min_val
    if range == 0.0 { return 0.5 }
    return (value - n.min_val) / range
}

pub fn (n Normalizer) denormalize(value f64) f64 {
    range := n.max_val - n.min_val
    return (value * range) + n.min_val
}

// ------------------------------------------------------------------
// Neural Network Layer
// ------------------------------------------------------------------

pub struct Layer {
pub mut:
    weights [][]f64
    biases  []f64
    input_count int
    output_count int
}

pub fn new_layer(input_count int, output_count int) Layer {
    mut weights := [][]f64{}
    for _ in 0..input_count {
        mut row := []f64{}
        for _ in 0..output_count {
            row << rand.f64() - 0.5
        }
        weights << row
    }

    mut biases := []f64{}
    for _ in 0..output_count {
        biases << rand.f64() - 0.5
    }

    return Layer{
        weights: weights
        biases: biases
        input_count: input_count
        output_count: output_count
    }
}

pub fn (l Layer) forward(inputs []f64, use_activation bool) ([]f64, []f64) {
    mut outputs := []f64{}
    mut pre_activations := []f64{}

    for j in 0..l.output_count {
        mut sum := l.biases[j]
        for i in 0..l.input_count {
            sum += inputs[i] * l.weights[i][j]
        }
        pre_activations << sum

        if use_activation {
            outputs << relu(sum)
        } else {
            outputs << sum
        }
    }
    return outputs, pre_activations
}

// ------------------------------------------------------------------
// The Neural Network Model
// ------------------------------------------------------------------

pub struct NeuralNetwork {
pub mut:
    layers []Layer
    learning_rate f64
}

pub fn new_neural_network(layer_sizes []int, learning_rate f64) NeuralNetwork {
    mut layers := []Layer{}
    for i in 0..layer_sizes.len - 1 {
        layers << new_layer(layer_sizes[i], layer_sizes[i+1])
    }

    return NeuralNetwork{
        layers: layers
        learning_rate: learning_rate
    }
}

pub fn (nn NeuralNetwork) predict(inputs []f64) f64 {
    mut current_input := inputs.clone()

    for i in 0..nn.layers.len - 1 {
        outputs, _ := nn.layers[i].forward(current_input, true)
        current_input = outputs.clone()
    }

    final_outputs, _ := nn.layers[nn.layers.len - 1].forward(current_input, false)
    return sigmoid(final_outputs[0])
}

pub fn (mut nn NeuralNetwork) train(inputs []f64, target f64) f64 {
    mut activations := [][]f64{}
    mut pre_activations := [][]f64{}
    mut current_input := inputs.clone()

    activations << inputs.clone()

    for i in 0..nn.layers.len - 1 {
        outs, pres := nn.layers[i].forward(current_input, true)
        activations << outs.clone()
        pre_activations << pres.clone()
        current_input = outs.clone()
    }

    final_out, final_pre := nn.layers[nn.layers.len - 1].forward(current_input, false)
    predicted := sigmoid(final_out[0])

    activations << [predicted]
    pre_activations << final_pre

    error := (target - predicted) * sigmoid_derivative(predicted)

    last_layer_idx := nn.layers.len - 1
    // FIX: Declare as mutable reference to modify fields
    mut last_layer := mut nn.layers[last_layer_idx]
    prev_activation := activations[last_layer_idx]

    for j in 0..last_layer.output_count {
        grad := error * prev_activation[j]
        last_layer.weights[j][0] += nn.learning_rate * grad
    }
    last_layer.biases[0] += nn.learning_rate * error

    // Backpropagation for hidden layers
    if nn.layers.len > 2 {
        hidden_layer_idx := nn.layers.len - 2
        // FIX: Declare as mutable reference
        mut current_layer := mut nn.layers[hidden_layer_idx]

        mut hidden_deltas := []f64{}
        for j in 0..current_layer.output_count {
            d := error * last_layer.weights[j][0] * relu_derivative(pre_activations[hidden_layer_idx][j])
            hidden_deltas << d
        }

        for j in 0..current_layer.output_count {
            delta_val := hidden_deltas[j]
            for k in 0..current_layer.input_count {
                current_layer.weights[k][j] += nn.learning_rate * delta_val * activations[hidden_layer_idx][k]
            }
            current_layer.biases[j] += nn.learning_rate * delta_val
        }
    }

    return math.abs(target - predicted)
}

// ------------------------------------------------------------------
// Main Execution
// ------------------------------------------------------------------

fn main() {
    hidden_neuron_count := 100
    min_val := 0.00000001
    max_val := 999999999.0

    normalizer := new_normalizer(min_val, max_val)
    mut nn := new_neural_network([2, hidden_neuron_count, 1], 0.05)

    println("Training Neural Network with ${hidden_neuron_count} neurons...")

    mut training_data := [][]f64{}
    mut targets := []f64{}

    for _ in 0..2000 {
        n1 := min_val + rand.f64() * (max_val - min_val)
        // FIX: n2 must be mutable to be modified below
        mut n2 := min_val + rand.f64() * (max_val - min_val)

        n1_norm := normalizer.normalize(n1)
        n2_norm := normalizer.normalize(n2)

        op := rand.int() % 4
        mut result := 0.0

        if op == 0 { result = n1 + n2 }
        else if op == 1 { result = math.abs(n1 - n2) }
        else if op == 2 { result = n1 * n2 }
        else {
            if n2 < 0.0001 { n2 = 0.0001 }
            result = n1 / n2
        }

        if result > max_val { result = max_val }
        result_norm := normalizer.normalize(result)

        training_data << [n1_norm, n2_norm]
        targets << result_norm
    }

    for epoch in 0..5000 {
        mut total_loss := 0.0
        for i in 0..training_data.len {
            loss := nn.train(training_data[i], targets[i])
            total_loss += loss
        }
        if epoch % 1000 == 0 {
            // FIX: Use f64() for conversion instead of float()
            //println("Epoch ${epoch}, Avg Loss: ${total_loss / f64(training_data.len):.6f}")
        }
    }

    println("\n--- Testing ---")
    test_cases := [
        [10.0, 5.0],
        [100.0, 20.0],
        [10.0, 10.0],
        [50.0, 2.0],
    ]

    for case in test_cases {
        n1 := case[0]
        n2 := case[1]

        n1_norm := normalizer.normalize(n1)
        n2_norm := normalizer.normalize(n2)

        pred_norm := nn.predict([n1_norm, n2_norm])
        pred_val := normalizer.denormalize(pred_norm)

        println("Input: ${n1}, ${n2} | Predicted: ${pred_val:.2f}")
    }
}