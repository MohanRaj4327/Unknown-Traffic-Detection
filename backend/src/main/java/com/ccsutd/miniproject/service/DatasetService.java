package com.ccsutd.miniproject.service;

import com.ccsutd.miniproject.dto.DatasetSplit;
import org.springframework.stereotype.Service;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

@Service
public class DatasetService {

    // Default configuration for open-set simulation on Scenario A2
    private static final Set<String> KNOWN_CLASSES = new HashSet<>(Arrays.asList(
            "BROWSING", "CHAT", "STREAMING", "MAIL", "VOIP"
    ));
    // New Classes: "P2P", "FT"

    /**
     * Loads an ARFF file using Weka's ArffLoader.
     * 
     * @param filePath the path to the ARFF dataset file
     * @return Instances object representing the loaded dataset
     * @throws IOException if the file cannot be found or read
     */
    public Instances loadDataset(String filePath) throws IOException {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(filePath));
        Instances data = loader.getDataSet();
        
        // Ensure the last attribute is set as the class attribute
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        
        return data;
    }

    /**
     * Creates a reproducible split of the dataset for the open-set experiment.
     * Known Classes go into Training (60%), Unlabelled (20%), and Testing (20%).
     * New Classes go into Unlabelled (50%) and Testing (50%) only.
     * 
     * @param data The complete loaded dataset
     * @return DatasetSplit containing the three subsets
     */
    public DatasetSplit prepareOpenSetExperiment(Instances data) {
        // Create empty datasets with the same header structure
        Instances knownTrainingData = new Instances(data, 0);
        Instances unlabelledData = new Instances(data, 0);
        Instances testingData = new Instances(data, 0);

        // Make a copy to shuffle
        Instances shuffledData = new Instances(data);
        // Fixed seed for reproducible experiments
        shuffledData.randomize(new Random(42));

        for (int i = 0; i < shuffledData.numInstances(); i++) {
            Instance inst = shuffledData.instance(i);
            String className = inst.stringValue(inst.classIndex());

            if (KNOWN_CLASSES.contains(className)) {
                // Determine split for KnownC: 60% Train, 20% Unlabelled, 20% Test
                // We can use modulo for a deterministic split after shuffle
                if (i % 5 < 3) {
                    knownTrainingData.add(inst); // 3/5 = 60%
                } else if (i % 5 == 3) {
                    unlabelledData.add(inst);    // 1/5 = 20%
                } else {
                    testingData.add(inst);       // 1/5 = 20%
                }
            } else {
                // Determine split for NewC (e.g., P2P, FT): 50% Unlabelled, 50% Test
                // NewC is NEVER added to knownTrainingData
                if (i % 2 == 0) {
                    unlabelledData.add(inst);
                } else {
                    testingData.add(inst);
                }
            }
        }

        return new DatasetSplit(knownTrainingData, unlabelledData, testingData);
    }
}
