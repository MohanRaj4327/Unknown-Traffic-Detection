package com.ccsutd.miniproject.controller;

import com.ccsutd.miniproject.dto.PredictionResult;
import com.ccsutd.miniproject.dto.DatasetSplit;
import com.ccsutd.miniproject.service.DatasetService;
import com.ccsutd.miniproject.service.MachineLearningService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.Arrays;
import java.util.HashSet;

@RestController
@RequestMapping("/api/ml")
public class MLController {

    @Autowired
    private DatasetService datasetService;

    @Autowired
    private MachineLearningService mlService;

    // Hardcoded known classes for evaluation matching DatasetService
    private static final Set<String> KNOWN_CLASSES = new HashSet<>(Arrays.asList(
            "BROWSING", "CHAT", "STREAMING", "MAIL", "VOIP"
    ));

    @GetMapping("/test-baseline")
    public ResponseEntity<Map<String, Object>> testFixedThresholdBaseline(
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        
        try {
            // Load and split dataset
            Instances data = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(data);

            // Train H2 on KnownC
            long startTrain = System.currentTimeMillis();
            mlService.trainH2Classifier(split.getKnownTrainingData());
            long trainTime = System.currentTimeMillis() - startTrain;

            // Evaluate on testing dataset
            int totalTested = 0;
            int correctKnown = 0; // True Positive (Known classified as exact Known)
            int correctNew = 0;   // True Negative (New classified as NEW)
            int falseKnown = 0;   // False Positive (New classified as Known)
            int falseNew = 0;     // False Negative (Known classified as NEW)
            
            Instances testingData = split.getTestingData();
            
            for (int i = 0; i < testingData.numInstances(); i++) {
                Instance testInst = testingData.instance(i);
                String actualClass = data.classAttribute().value((int) testInst.classValue());
                boolean isActuallyKnown = KNOWN_CLASSES.contains(actualClass);

                PredictionResult prediction = mlService.predictFixedThreshold(testInst, data);

                if (prediction.getClassType().equals("KNOWN")) {
                    if (isActuallyKnown && prediction.getPredictedClass().equals(actualClass)) {
                        correctKnown++;
                    } else {
                        falseKnown++; // Could be a New class classified as Known, or wrong Known class
                    }
                } else { // Predicted as "NEW"
                    if (!isActuallyKnown) {
                        correctNew++;
                    } else {
                        falseNew++;
                    }
                }
                totalTested++;
            }

            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            response.put("trainTimeMs", trainTime);
            response.put("totalTested", totalTested);
            response.put("correctKnown", correctKnown);
            response.put("correctNew", correctNew);
            response.put("falseKnown", falseKnown);
            response.put("falseNew", falseNew);

            // Calculate preliminary metrics
            int totalActuallyKnown = correctKnown + falseNew;
            int totalActuallyNew = correctNew + falseKnown;
            
            double knownAccuracy = totalActuallyKnown > 0 ? (double) correctKnown / totalActuallyKnown : 0;
            double unknownAccuracy = totalActuallyNew > 0 ? (double) correctNew / totalActuallyNew : 0;
            
            // Normalized Accuracy (NA) with lambda = 0.5
            double normalizedAccuracy = 0.5 * knownAccuracy + 0.5 * unknownAccuracy;

            Map<String, Double> metrics = new HashMap<>();
            metrics.put("knownAccuracy", knownAccuracy);
            metrics.put("unknownAccuracy", unknownAccuracy);
            metrics.put("normalizedAccuracy", normalizedAccuracy);
            response.put("metrics", metrics);

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
}
