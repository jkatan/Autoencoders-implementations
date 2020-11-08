import java.util.List;

public class DenoisingAutoencoderDemo {

    public static void main(String[] args) {
        FontDatasetManager fontManager = new FontDatasetManager();
        List<List<Double>> originalFontsRepresentations = fontManager.getFontsRepresentations();
        List<List<Double>> fontsRepresentationsWithNoise = fontManager.getFontsRepresentationsWithRandomNoise(2);
        for (int i = 0; i < originalFontsRepresentations.size(); i++) {
            System.out.println("__________");
            System.out.println("Original font representation:");
            fontManager.displayLetterRepresentation(originalFontsRepresentations.get(i));
            System.out.println(originalFontsRepresentations.get(i));

            System.out.println("Noisy font representation: ");
            fontManager.displayLetterRepresentation(fontsRepresentationsWithNoise.get(i));
            System.out.println(fontsRepresentationsWithNoise.get(i));
        }

        // Training an autoencoder to "denoise" data
        NeuralNetwork neuralNetwork = new NeuralNetwork(0.85, 20, 0.05);
        neuralNetwork.addLayer(25, 35); // Encoder 1st layer
        neuralNetwork.addLayer(10, 25); // Latent code
        neuralNetwork.addLayer(25, 10); // Decoder 1st layer
        neuralNetwork.addLayer(35, 25); // Output layer
        neuralNetwork.train(fontsRepresentationsWithNoise, originalFontsRepresentations, 0.01, 2.0);

        // Testing the "denoising" capacity of the network
        for (List<Double> originalFont : originalFontsRepresentations) {
            System.out.println("Letter without noise: ");
            fontManager.displayLetterRepresentation(originalFont);

            List<Double> noisyLetter = fontManager.addNoiseToLetter(originalFont, 2);
            System.out.println("Letter with noise: ");
            fontManager.displayLetterRepresentation(noisyLetter);

            List<Double> denoisedOutput = neuralNetwork.forwardPropagate(noisyLetter);
            System.out.println("Network output after denoising letter: ");
            System.out.println(denoisedOutput);
            fontManager.displayLetterRepresentation(denoisedOutput);
        }
    }
}
