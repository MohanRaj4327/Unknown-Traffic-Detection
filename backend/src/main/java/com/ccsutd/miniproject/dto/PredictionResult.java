package com.ccsutd.miniproject.dto;

/**
 * Data Transfer Object representing the final prediction of the cascade structure.
 */
public class PredictionResult {

    private String predictedClass;
    private String classType; // "KNOWN" or "NEW"
    private ConfidenceResult confidenceMetrics;
    private double thresholdApplied;
    private String h1Result; // To be used in later phases (e.g., "PASS_TO_H2", "NEW_CLASS_DETECTED")

    public PredictionResult(String predictedClass, String classType, ConfidenceResult confidenceMetrics, double thresholdApplied, String h1Result) {
        this.predictedClass = predictedClass;
        this.classType = classType;
        this.confidenceMetrics = confidenceMetrics;
        this.thresholdApplied = thresholdApplied;
        this.h1Result = h1Result;
    }

    public String getPredictedClass() {
        return predictedClass;
    }

    public String getClassType() {
        return classType;
    }

    public ConfidenceResult getConfidenceMetrics() {
        return confidenceMetrics;
    }

    public double getThresholdApplied() {
        return thresholdApplied;
    }

    public String getH1Result() {
        return h1Result;
    }
}
