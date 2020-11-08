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

        // Training an autoencoder to represent each font in 2 dimensions
        NeuralNetwork neuralNetwork = new NeuralNetwork(0.8, 20, 0.005);
        neuralNetwork.addLayer(25, 35); // Encoder 1st layer
        neuralNetwork.addLayer(2, 25); // Latent code
        neuralNetwork.addLayer(25, 2); // Decoder 1st layer
        neuralNetwork.addLayer(35, 25); // Output layer

        neuralNetwork.train(trainRepresentations, trainRepresentations, 0.01, 5.0);

        // Write results to a file so we can plot the latent space in a 2D-plane
        List<List<Double>> latentSpacePoints = new ArrayList<>();
        FileWriter writer = new FileWriter("latentSpace.csv");
        for (List<Double> fontRepr : trainRepresentations) {
            System.out.println("____________________________________");
            System.out.println("Original output: ");
            System.out.println(fontRepr);
            fontManager.displayLetterRepresentation(fontRepr);

            List<Double> testOutput = neuralNetwork.forwardPropagate(fontRepr);
            System.out.println("Test output: ");
            System.out.println(testOutput);
            fontManager.displayLetterRepresentation(testOutput);

            List<Double> latentSpaceOutput = neuralNetwork.testAutoencoderLatentSpace(fontRepr, 1);
            latentSpacePoints.add(latentSpaceOutput);

            writer.write(String.valueOf(latentSpaceOutput.get(0)));
            writer.write(",");
            writer.write(String.valueOf(latentSpaceOutput.get(1)));
            writer.write("\n");

            System.out.println("Latent space output: ");
            System.out.println(latentSpaceOutput);
            System.out.println("____________________________________");
        }
        writer.close();

        // Now, we take two points in the latent space that are close to each other
        FileWriter closePointsWriter = new FileWriter("closeLettersIndexes.csv");
        boolean distanceFound = false;
        List<Double> closePointA = null;
        List<Double> closePointB = null;
        for (int i = 0; i < latentSpacePoints.size()-1 && !distanceFound; i++) {
            for (int j = i+1; j < latentSpacePoints.size() && !distanceFound; j++) {
                List<Double> pointA = latentSpacePoints.get(i);
                List<Double> pointB = latentSpacePoints.get(j);
                double distance = Math.sqrt((Math.pow(pointA.get(0) - pointB.get(0), 2)) + Math.pow((pointA.get(1) - pointB.get(1)), 2));
                if (distance < 0.75 && distance > 0.1) {
                    System.out.println("Point A: " + i);
                    System.out.println("Point B: " + j);
                    closePointsWriter.write(String.valueOf(i));
                    closePointsWriter.write(",");
                    closePointsWriter.write(String.valueOf(j));
                    closePointsWriter.write("\n");
                    distanceFound = true;
                    closePointA = pointA;
                    closePointB = pointB;
                }
            }
        }
        closePointsWriter.close();

        if (closePointA == null) {
            System.out.println("No properly distanced points found to interpolate, try again...");
            return;
        }

        // Now, we interpolate between these two points to generate a new one
        double t = 0.5;
        List<Double> interpolatedPoint = new ArrayList<>();
        double x = (1.0 - t) * closePointA.get(0) + t * closePointB.get(0);
        double y = (1.0 - t) * closePointA.get(1) + t * closePointB.get(1);
        interpolatedPoint.add(x);
        interpolatedPoint.add(y);
        FileWriter interpolatedPointWriter = new FileWriter("interpolatedPoint.csv");
        interpolatedPointWriter.write(String.valueOf(interpolatedPoint.get(0)) + "," + String.valueOf(interpolatedPoint.get(1)));
        interpolatedPointWriter.close();

        // Finally, we pass this new point to the decoder and see the new letter
        List<Double> interpolatedLetter = neuralNetwork.forwardPropagateAtStartingLayer(interpolatedPoint, 2);
        System.out.println("Interpolated letter: " + interpolatedLetter);
        fontManager.displayLetterRepresentation(interpolatedLetter);
    }
}
