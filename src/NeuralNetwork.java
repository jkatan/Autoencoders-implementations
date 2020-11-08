import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class NeuralNetwork {

    private final List<List<Neuron>> neuralNetwork;
    private final Double alphaMomentum;
    private Integer consistentErrorIncrementTimes;
    private Integer consistentErrorDecrementTimes;
    private final Integer epochsRequiredToChangeEta;
    private final Double deltaEta;

    public NeuralNetwork(Double alphaMomentum, Integer epochsToChangeEta, Double deltaEta) {
        this.neuralNetwork = new ArrayList<>();
        this.alphaMomentum = alphaMomentum;
        this.epochsRequiredToChangeEta = epochsToChangeEta;
        this.deltaEta = deltaEta;
        this.consistentErrorDecrementTimes = 0;
        this.consistentErrorIncrementTimes = 0;
    }

    public void addLayer(int neuronsAmount, int previousLayerNeuronsAmount) {
        List<Neuron> newLayer = new ArrayList<>(neuronsAmount);
        for (int i = 0; i < neuronsAmount; i++) {
            Neuron neuron = new Neuron(generateRandomWeights(previousLayerNeuronsAmount+1));
            newLayer.add(neuron);
        }
        neuralNetwork.add(newLayer);
    }

    public List<Neuron> getLayer(int layerIndex) {
        return this.neuralNetwork.get(layerIndex);
    }

    public void train(List<List<Double>> trainingSamples, List<List<Double>> expectedOutputs, double eta, double minError) {
        int currentEpoch = 1;
        double currentMaxError = minError+1;
        Random random = new Random();
        double deltaMaxError = 0.0;
        Double previousMaxError = 0.0;
        while (currentMaxError > minError) {
            List<Double> errors = new ArrayList<>();
            for (int j = 0; j < trainingSamples.size(); j++) {
                int randomIndex = random.nextInt(trainingSamples.size());
                List<Double> trainingInput = trainingSamples.get(randomIndex);
                List<Double> expectedTrainingOutput = expectedOutputs.get(randomIndex);
                double error = 0.0;
                List<Double> outputs = forwardPropagate(trainingInput);
                for (int i = 0; i < outputs.size(); i++) {
                    error += Math.abs(outputs.get(i) - expectedTrainingOutput.get(i));
                }
                errors.add(error);
                backPropagate(expectedTrainingOutput);
                if (consistentErrorIncrementTimes > epochsRequiredToChangeEta) {
                    consistentErrorIncrementTimes = 0;
                    updateWeights(trainingInput, eta - deltaEta);
                } else if (consistentErrorDecrementTimes > epochsRequiredToChangeEta) {
                    consistentErrorDecrementTimes = 0;
                    updateWeights(trainingInput, eta + deltaEta);
                } else {
                    updateWeights(trainingInput, eta);
                }
            }

            previousMaxError = Collections.max(errors);

            deltaMaxError = currentMaxError - previousMaxError;
            if (deltaMaxError > 0) {
                consistentErrorDecrementTimes = 0;
                consistentErrorIncrementTimes += 1;
            } else {
                consistentErrorIncrementTimes = 0;
                consistentErrorDecrementTimes += 1;
            }

            currentMaxError = previousMaxError;
            currentEpoch += 1;

            System.out.println("Current epoch: " + currentEpoch);
            System.out.print("Current max error: " + currentMaxError);
            System.out.println();
        }

        System.out.println("Epochs: " + currentEpoch);
        System.out.println("Max error: " + currentMaxError);
    }

    public List<Double> forwardPropagate(List<Double> initialInputs) {
        List<Double> currentInputs = initialInputs;
        for (List<Neuron> layer : this.neuralNetwork) {
            List<Double> nextInputs = new ArrayList<>();
            for (Neuron neuron : layer) {
                nextInputs.add(neuron.activate(currentInputs));
            }
            currentInputs = nextInputs;
        }
        return currentInputs;
    }

    public List<Double> forwardPropagateAtStartingLayer(List<Double> initialInputs, int startingPropagationLayerIndex) {
        List<Double> currentInputs = initialInputs;
        int currentLayerIndex = 0;
        for (List<Neuron> layer : this.neuralNetwork) {
            if (currentLayerIndex >= startingPropagationLayerIndex) {
                List<Double> nextInputs = new ArrayList<>();
                for (Neuron neuron : layer) {
                    nextInputs.add(neuron.activate(currentInputs));
                }
                currentInputs = nextInputs;
            }
            currentLayerIndex += 1;
        }
        return currentInputs;
    }

    // This performs a forward propagation using the given inputs, and returns the outputs of the neurons on the latent space
    public List<Double> testAutoencoderLatentSpace(List<Double> inputs, int latentSpaceLayerIndex) {
        List<Double> currentInputs = inputs;
        int layerIndex = 0;
        for (List<Neuron> layer : this.neuralNetwork) {
            List<Double> nextInputs = new ArrayList<>();
            for (Neuron neuron : layer) {
                nextInputs.add(neuron.activate(currentInputs));
            }
            currentInputs = nextInputs;
            if (layerIndex == latentSpaceLayerIndex) {
                return currentInputs;
            }
            layerIndex += 1;
        }
        return currentInputs;
    }

    private void backPropagate(List<Double> expectedOutputs) {
        // We iterate the layers in reverse order
        for (int i = this.neuralNetwork.size()-1; i >= 0; i--) {
            List<Neuron> currentLayer = this.neuralNetwork.get(i);
            // In this case we calculate the delta of the exit layer
            if (i == this.neuralNetwork.size() - 1) {
                for (int j = 0; j < expectedOutputs.size(); j++) {
                    Neuron exitNeuron = currentLayer.get(j);
                    Double error = expectedOutputs.get(j) - exitNeuron.getOutput();
                    Double delta = exitNeuron.activationDerivative() * error;
                    exitNeuron.setDelta(delta);
                }
            } else {
                // Here we update the deltas of the hidden layer neurons
                List<Neuron> upperLayer = this.neuralNetwork.get(i+1);
                for (int j=0; j<currentLayer.size(); j++) {
                    double error = 0.0;
                    Neuron currentLayerNeuron = currentLayer.get(j);
                    for (Neuron upperNeuron : upperLayer) {
                        error += upperNeuron.getWeights().get(j) * upperNeuron.getDelta();
                    }
                    currentLayerNeuron.setDelta(error * currentLayerNeuron.activationDerivative());
                }
            }
        }
    }

    // Given a neural network that has passed through the forward and backpropagation phases, update it's weights
    private void updateWeights(List<Double> inputs, double eta) {
        List<Double> nextInputs = inputs;
        for (int i = 0; i < this.neuralNetwork.size(); i++) {
            if (i > 0) {
                nextInputs = this.neuralNetwork.get(i-1).stream().map(Neuron::getOutput).collect(Collectors.toList());
            }

            List<Neuron> currentLayer = this.neuralNetwork.get(i);
            for (Neuron neuron : currentLayer) {
                for (int j = 0; j < nextInputs.size(); j++) {
                    Double weight = neuron.getWeights().get(j);
                    Double lastDeltaWeight = neuron.getLastsDeltaWeights().get(j);
                    weight += eta * neuron.getDelta() * nextInputs.get(j) + alphaMomentum * lastDeltaWeight;
                    neuron.getWeights().set(j, weight);
                    neuron.getLastsDeltaWeights().set(j, eta * neuron.getDelta() * nextInputs.get(j));
                }

                int weightsAmount = neuron.getWeights().size();
                Double bias = neuron.getWeights().get(weightsAmount-1);
                bias += eta * neuron.getDelta() + alphaMomentum * neuron.getLastsDeltaWeights().get(weightsAmount-1);
                neuron.getWeights().set(weightsAmount-1, bias);
                neuron.getLastsDeltaWeights().set(weightsAmount-1, eta * neuron.getDelta());
            }
        }
    }

    private List<Double> generateRandomWeights(int amountToGenerate) {
        return new Random().doubles(amountToGenerate).boxed().collect(Collectors.toList());
    }
}
