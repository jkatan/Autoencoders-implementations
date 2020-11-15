import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class ej2Demo {

    public static void main(String[] args) throws FileNotFoundException {
        List<List<Double>> trainingData = getTrainingData();
        printDigits(trainingData);

        // Training an autoencoder to compress the data in the latent space of the network
        NeuralNetwork neuralNetwork = new NeuralNetwork(0.85, 5, 0.005);
        neuralNetwork.addLayer(30, 35); // Encoder 1st layer
        neuralNetwork.addLayer(3, 30); // Latent code
        neuralNetwork.addLayer(30, 3); // Decoder 1st layer
        neuralNetwork.addLayer(35, 30); // Output layer

        neuralNetwork.train(trainingData, trainingData, 0.01, 0.3);

        displayTestResults(neuralNetwork, trainingData);

        List<List<Double>> latentSpaceRepresentations = getLatentSpaceRepresentations(neuralNetwork, trainingData);
        System.out.println("Latent space representations: ");
        System.out.println(latentSpaceRepresentations);
        System.out.println();

        List<List<Double>> closePointsInLatentSpace = pickTwoClosePointsInLatentSpace(latentSpaceRepresentations);
        if (closePointsInLatentSpace == null) {
            System.out.println("No points found in latent space that are suitable for interpolation, try again");
            return;
        }

        System.out.println("Close points in latent space to interpolate: ");
        System.out.println(closePointsInLatentSpace);

        System.out.println("Point A digit: ");
        List<Double> pointADigitBits = neuralNetwork.forwardPropagateAtStartingLayer(closePointsInLatentSpace.get(0), 2);
        printDigit(pointADigitBits);

        System.out.println("Point B digit: ");
        List<Double> pointBDigitBits = neuralNetwork.forwardPropagateAtStartingLayer(closePointsInLatentSpace.get(1), 2);
        printDigit(pointBDigitBits);

        System.out.println();

        List<Double> interpolationFactors = new ArrayList<>();
        interpolationFactors.add(0.1);
        interpolationFactors.add(0.2);
        interpolationFactors.add(0.3);
        interpolationFactors.add(0.4);
        interpolationFactors.add(0.5);
        interpolationFactors.add(0.6);
        interpolationFactors.add(0.7);
        interpolationFactors.add(0.8);
        interpolationFactors.add(0.9);

        for (Double interpolationFactor : interpolationFactors) {
            List<Double> interpolatedDigit = interpolateTwoPoints(closePointsInLatentSpace.get(0), closePointsInLatentSpace.get(1), interpolationFactor);
            List<Double> interpolatedDigitNetworkOutput = neuralNetwork.forwardPropagateAtStartingLayer(interpolatedDigit, 2);
            System.out.println("New digit generated from two close points in the latent space of the network: ");
            System.out.println("Interpolation factor: " + interpolationFactor);
            printDigit(interpolatedDigitNetworkOutput);
        }
    }

    private static List<Double> interpolateTwoPoints(List<Double> pointA, List<Double> pointB, double interpolationFactor) {
        List<Double> interpolatedPoint = new ArrayList<>();

        double x = (1.0 - interpolationFactor) * pointA.get(0) + interpolationFactor * pointB.get(0);
        double y = (1.0 - interpolationFactor) * pointA.get(1) + interpolationFactor * pointB.get(1);
        double z = (1.0 - interpolationFactor) * pointA.get(2) + interpolationFactor * pointB.get(2);

        interpolatedPoint.add(x);
        interpolatedPoint.add(y);
        interpolatedPoint.add(z);

        return interpolatedPoint;
    }

    private static List<List<Double>> pickTwoClosePointsInLatentSpace(List<List<Double>> latentSpacePoints) {
        boolean distanceFound = false;
        List<Double> closePointA = null;
        List<Double> closePointB = null;
        for (int i = 0; i < latentSpacePoints.size()-1 && !distanceFound; i++) {
            for (int j = i+1; j < latentSpacePoints.size() && !distanceFound; j++) {
                List<Double> pointA = latentSpacePoints.get(i);
                List<Double> pointB = latentSpacePoints.get(j);
                double distance = Math.sqrt((Math.pow(pointA.get(0) - pointB.get(0), 2))
                                            + Math.pow((pointA.get(1) - pointB.get(1)), 2)
                                            + Math.pow(pointA.get(2) - pointB.get(2), 2));
                if (distance < 1.5 && distance > 1.0) {
                    System.out.println("Point A index: " + i);
                    System.out.println("Point B index: " + j);
                    distanceFound = true;
                    closePointA = pointA;
                    closePointB = pointB;
                }
            }
        }

        if (closePointA == null) {
            return null;
        }

        List<List<Double>> closePointsToReturn = new ArrayList<>();
        closePointsToReturn.add(closePointA);
        closePointsToReturn.add(closePointB);
        return closePointsToReturn;
    }

    private static List<List<Double>> getLatentSpaceRepresentations(NeuralNetwork network, List<List<Double>> samples) {
        List<List<Double>> latentSpaceRepresentations = new ArrayList<>();
        for (List<Double> sample : samples) {
            List<Double> latentSpaceOutput = network.testAutoencoderLatentSpace(sample, 1);
            latentSpaceRepresentations.add(latentSpaceOutput);
        }
        return latentSpaceRepresentations;
    }

    private static void displayTestResults(NeuralNetwork neuralNetwork, List<List<Double>> samplesToTest) {
        System.out.println("Autoencoder testing results:");
        for (List<Double> sample : samplesToTest) {
            System.out.println("Network input:");
            printDigit(sample);
            System.out.println(sample);

            System.out.println("Network output:");
            List<Double> output = neuralNetwork.forwardPropagate(sample);
            printDigit(output);
            System.out.println(output);
            System.out.println();
        }
    }

    private static void printDigits(List<List<Double>> trainingData) {
        int digit = 0;
        for (List<Double> trainingSample : trainingData) {
            printDigit(trainingSample);
            System.out.println("Digit: " + digit);
            System.out.println();
            digit += 1;
        }
    }

    private static void printDigit(List<Double> elementToPrint) {
        for (int j = 0; j < elementToPrint.size(); j++) {
            long bitToPrint = Math.round(elementToPrint.get(j));
            if (bitToPrint == 1.0)
                System.out.print("#");
            else if (bitToPrint == 0.0)
                System.out.print(" ");
            else
                System.out.print("-");

            if ((j + 1) % 5 == 0) {
                System.out.println();
            }
        }
    }

    private static List<List<Double>> getTrainingData() throws FileNotFoundException {
        File file = new File("mapaDeDigitos.txt");
        Scanner reader = new Scanner(file);
        int lineNumber = 1;
        List<List<Double>> trainingData = new ArrayList<>();
        List<Double> digitMap = new ArrayList<>();
        while (reader.hasNextLine()) {
            String data = reader.nextLine();
            for (int i = 0; i < data.length(); i++) {
                if (data.charAt(i) != '\n' && data.charAt(i) != ' ') {
                    digitMap.add(Double.parseDouble(String.valueOf(data.charAt(i))));
                }
            }
            if (lineNumber % 7 == 0) {
                trainingData.add(digitMap);
                digitMap = new ArrayList<>();
            }
            lineNumber += 1;
        }

        reader.close();
        return trainingData;
    }
}
