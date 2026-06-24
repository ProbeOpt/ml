module main

import ml


fn xor_model() {
    mut my_nn := ml.new_nn(2, 4, 1)  // 2 inputs, 4 hidden neurons, 1 output

    inputs := [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]

    targets := [
        [0.0],
        [1.0],
        [1.0],
        [1.0],
    ]

    println('Training XOR-like problem...')
    my_nn.train_epochs(inputs, targets, 20000)

    // Test predictions
    println('Results:')
    for i, inp in inputs {
        pred := my_nn.predict(inp)
        println('${inp} -> ${pred[0]:.4f}  (target: ${targets[i][0]})')
    }
}


fn and_model() {
    mut my_nn := ml.new_nn(2, 10, 1)  // 2 inputs, 4 hidden neurons, 1 output

    inputs := [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]

    targets := [
        [0.0],
        [1.0],
        [1.0],
        [1.0],
    ]

    println('Training AND problem...')
    my_nn.train_epochs(inputs, targets, 20000)

    // Test predictions
    println('Results:')
    for i, inp in inputs {
        pred := my_nn.predict(inp)
        println('${inp} -> ${pred[0]:.4f}  (target: ${targets[i][0]})')
    }
}

fn main() {
    and_model()
}