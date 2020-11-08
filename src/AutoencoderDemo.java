import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class AutoencoderDemo {

    public static void main(String[] args) throws IOException {
        FontDatasetManager fontManager = new FontDatasetManager();
        List<List<Double>> fontsRepresentations = fontManager.getFontsRepresentations();

        List<List<Double>> trainRepresentations = fontsRepresentations.subList(0, 10);

        NeuralNetwork neuralNetwork = new NeuralNetwork(0.8, 20, 0.005, 1);
        neuralNetwork.addLayer(25, 35); // Encoder 1st layer
        neuralNetwork.addLayer(2, 25); // Latent code
        neuralNetwork.addLayer(25, 2); // Decoder 1st layer
        neuralNetwork.addLayer(35, 25); // Output layer

        neuralNetwork.train(trainRepresentations, trainRepresentations, 0.01, 5.0);

        FileWriter writer = new FileWriter("latentSpace.csv");
        int fontReprIndex = 0;
        for (List<Double> fontRepr : trainRepresentations) {
            System.out.println("____________________________________");
            System.out.println("Original output: ");
            System.out.println(fontRepr);
            fontManager.displayLetterRepresentation(fontRepr);

            List<Double> testOutput = neuralNetwork.forwardPropagate(fontRepr);
            System.out.println("Test output: ");
            System.out.println(testOutput);
            fontManager.displayLetterRepresentation(testOutput);

            List<Double> latentSpaceOutput = neuralNetwork.testAutoencoderLatentSpace(fontRepr);

            writer.write(String.valueOf(latentSpaceOutput.get(0)));
            writer.write(",");
            writer.write(String.valueOf(latentSpaceOutput.get(1)));
            writer.write(",");
            writer.write(String.valueOf(fontReprIndex));
            writer.write("\n");

            System.out.println("Latent space output: ");
            System.out.println(latentSpaceOutput);
            System.out.println("____________________________________");

            fontReprIndex += 1;
        }

        writer.close();
    }
}
