import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class FontDatasetManager {
    // The dimensions of each pattern are 7x5
    private final int[][] FONT_DATASET = new int[][] {
            {0x0e, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0f},   // 0x40, @
            {0x04, 0x0a, 0x11, 0x11, 0x1f, 0x11, 0x11},   // 0x41, A
            {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e},   // 0x42, B
            {0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e},   // 0x43, C
            {0x1e, 0x09, 0x09, 0x09, 0x09, 0x09, 0x1e},   // 0x44, D
            {0x1f, 0x10, 0x10, 0x1c, 0x10, 0x10, 0x1f},   // 0x45, E
            {0x1f, 0x10, 0x10, 0x1f, 0x10, 0x10, 0x10},   // 0x46, F
            {0x0e, 0x11, 0x10, 0x10, 0x13, 0x11, 0x0f},   // 0x37, G
            {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11},   // 0x48, H
            {0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e},   // 0x49, I
            {0x1f, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c},   // 0x4a, J
            {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},   // 0x4b, K
            {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f},   // 0x4c, L
            {0x11, 0x1b, 0x15, 0x11, 0x11, 0x11, 0x11},   // 0x4d, M
            {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11},   // 0x4e, N
            {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e},   // 0x4f, O
            {0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10},   // 0x50, P
            {0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d},   // 0x51, Q
            {0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11},   // 0x52, R
            {0x0e, 0x11, 0x10, 0x0e, 0x01, 0x11, 0x0e},   // 0x53, S
            {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},   // 0x54, T
            {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e},   // 0x55, U
            {0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04},   // 0x56, V
            {0x11, 0x11, 0x11, 0x15, 0x15, 0x1b, 0x11},   // 0x57, W
            {0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11},   // 0x58, X
            {0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04},   // 0x59, Y
            {0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f},   // 0x5a, Z
            {0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e},   // 0x5b, [
            {0x10, 0x10, 0x08, 0x04, 0x02, 0x01, 0x01},   // 0x5c, \\
            {0x0e, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0e},   // 0x5d, ]
            {0x04, 0x0a, 0x11, 0x00, 0x00, 0x00, 0x00},   // 0x5e, ^
            {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f}   // 0x5f, _
    };

    public List<List<Double>> getFontsRepresentations() {
        List<List<Double>> fontsRepresentations = new ArrayList<>();
        for (int[] letter : FONT_DATASET) {
            fontsRepresentations.add(getLetterRepresentation(letter));
        }

        return fontsRepresentations;
    }

    public List<List<Double>> getFontsRepresentationsWithRandomNoise(int amountOfRandomPixelsToFlip) {
        List<List<Double>> fontsRepresentationsWithNoise = new ArrayList<>();
        Random random =  new Random();
        for (int[] letter : FONT_DATASET) {
           List<Double> letterRepresentation = getLetterRepresentation(letter);
            List<Double> noisyLetter = addNoiseToLetter(letterRepresentation, amountOfRandomPixelsToFlip);
           fontsRepresentationsWithNoise.add(noisyLetter);
        }

        return fontsRepresentationsWithNoise;
    }

    public List<Double> addNoiseToLetter(List<Double> letterRepresentation, int amountOfRandomPixelsToFlip) {
        List<Double> noisyLetter = new ArrayList<>(letterRepresentation);
        Random random = new Random();
        random.ints(amountOfRandomPixelsToFlip, 0, noisyLetter.size()).distinct().
            forEach((randIndex) -> {
                flipLetterPixel(noisyLetter, randIndex);
            });
        return noisyLetter;
    }

    private void flipLetterPixel(List<Double> letterToModify, int pixelIndexToFlip) {
        Double letterPixel = letterToModify.get(pixelIndexToFlip);
        if (letterPixel == 1.0) {
            letterToModify.set(pixelIndexToFlip, 0.0);
        } else {
            letterToModify.set(pixelIndexToFlip, 1.0);
        }
    }

    public List<Double> getLetterRepresentation(int[] letterBitmap) {
        List<Double> letterRepresentation = new ArrayList<>();
        for (int value : letterBitmap) {
            StringBuilder row = new StringBuilder(Integer.toBinaryString(value));
            int paddingZeros = 0;
            if (row.length() < 5) {
                paddingZeros = 5 - row.length();
                for (int paddIndex = 0; paddIndex < paddingZeros; paddIndex++) {
                    row.insert(0, "0");
                    letterRepresentation.add(0.0);
                }
            }

            for (int j = paddingZeros; j < row.length(); j++) {
                if (row.charAt(j) == '1') {
                    letterRepresentation.add(1.0);
                } else {
                    letterRepresentation.add(0.0);
                }
            }
        }

        return letterRepresentation;
    }

    public void displayLettersRepresentations(List<List<Double>> lettersRepresentations) {
        for (List<Double> letterRepresentation : lettersRepresentations) {
            displayLetterRepresentation(letterRepresentation);
        }
    }

    public void displayLetterRepresentation(List<Double> letterRepresentation) {
        int count = 1;
        long roundedValue;
        for (Double bit : letterRepresentation) {
            roundedValue = Math.round(bit);
            if (roundedValue == 1.0) {
                System.out.print("#");
            } else if (roundedValue == 0.0) {
                System.out.print(" ");
            } else {
                System.out.print("-");
            }
            if (count % 5 == 0) {
                System.out.println();
            }
            count++;
        }
        System.out.println();
    }
}
