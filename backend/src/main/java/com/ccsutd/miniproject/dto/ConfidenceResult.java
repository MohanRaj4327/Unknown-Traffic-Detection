package com.ccsutd.miniproject.dto;

import java.util.Map;

/**
 * Data Transfer Object to hold the confidence metrics produced by H2 (Random Forest).
 */
public class ConfidenceResult {
    
    private String predictedClass;
    private Map<String, Double> classConfidences;
    private double cfMax;
    private double cfMin;
    private double cfDMax;

    public ConfidenceResult(String predictedClass, Map<String, Double> classConfidences, double cfMax, double cfMin, double cfDMax) {
        this.predictedClass = predictedClass;
        this.classConfidences = classConfidences;
        this.cfMax = cfMax;
        this.cfMin = cfMin;
        this.cfDMax = cfDMax;
    }

    public String getPredictedClass() {
        return predictedClass;
    }

    public Map<String, Double> getClassConfidences() {
        return classConfidences;
    }

    public double getCfMax() {
        return cfMax;
    }

    public double getCfMin() {
        return cfMin;
    }

    public double getCfDMax() {
        return cfDMax;
    }

    @Override
    public String toString() {
        return "ConfidenceResult{" +
                "predictedClass='" + predictedClass + '\'' +
                ", cfMax=" + cfMax +
                ", cfMin=" + cfMin +
                ", cfDMax=" + cfDMax +
                '}';
    }
}
