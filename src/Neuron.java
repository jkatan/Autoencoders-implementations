import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Neuron {

    private final List<Double> weights;
    private final List<Double> lastsDeltaWeights;
    private Double output;
    private Double delta;

    public Neuron(List<Double> weights) {
        this.weights = weights;
        this.lastsDeltaWeights = new ArrayList<>(Arrays.asList(new Double[weights.size()]));
        Collections.fill(lastsDeltaWeights, 0.0);
        this.output = 0.0;
        this.delta = 0.0;
    }

    // Here we use the tanh activation function
    public Double activate(List<Double> inputs) {
        Double neuronExcitement = excite(inputs);
        output = Math.tanh(neuronExcitement);
        return output;
    }

    // This is the derivative of the tanh function
    public Double activationDerivative() {
        return (1 - Math.pow(Math.tanh(output), 2));
    }

    private Double excite(List<Double> inputs) {
        // We assume that the bias is the weight in the last position of the list
        Double neuronExcitement = this.weights.get(this.weights.size()-1);
        for (int i = 0; i < inputs.size(); i++) {
            neuronExcitement += this.weights.get(i) * inputs.get(i);
        }

        return neuronExcitement;
    }

    public List<Double> getWeights() {
        return weights;
    }

    public List<Double> getLastsDeltaWeights() {
        return lastsDeltaWeights;
    }

    public Double getOutput() {
        return output;
    }

    public Double getDelta() {
        return delta;
    }

    public void setDelta(Double delta) {
        this.delta = delta;
    }

    @Override
    public String toString() {
        return "Neuron{" +
                "weights=" + weights +
                ", output=" + output +
                ", delta=" + delta +
                '}';
    }
}
