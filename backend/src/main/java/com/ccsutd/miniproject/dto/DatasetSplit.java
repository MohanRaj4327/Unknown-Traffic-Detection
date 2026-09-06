package com.ccsutd.miniproject.dto;

import weka.core.Instances;

/**
 * Data Transfer Object to hold the datasets for the open-set experiment.
 */
public class DatasetSplit {
    
    // Data used to train H2 and the OC-SVMs
    private Instances knownTrainingData;
    
    // Data without labels (conceptually) used to find pseudo-negatives for H1
    private Instances unlabelledData;
    
    // Data used for final cascade evaluation
    private Instances testingData;

    public DatasetSplit(Instances knownTrainingData, Instances unlabelledData, Instances testingData) {
        this.knownTrainingData = knownTrainingData;
        this.unlabelledData = unlabelledData;
        this.testingData = testingData;
    }

    public Instances getKnownTrainingData() {
        return knownTrainingData;
    }

    public Instances getUnlabelledData() {
        return unlabelledData;
    }

    public Instances getTestingData() {
        return testingData;
    }
}
