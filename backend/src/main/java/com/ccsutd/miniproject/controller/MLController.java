package com.ccsutd.miniproject.controller;

import com.ccsutd.miniproject.dto.PredictionResult;
import com.ccsutd.miniproject.dto.DatasetSplit;
import com.ccsutd.miniproject.service.DatasetService;
import com.ccsutd.miniproject.service.FeatureSelectionService;
import com.ccsutd.miniproject.service.MachineLearningService;
import com.ccsutd.miniproject.service.OcSvmService;
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
    
    @Autowired
    private FeatureSelectionService featureSelectionService;

    @Autowired
    private OcSvmService ocSvmService;

    // Hardcoded known classes for evaluation matching DatasetService
    private static final Set<String> KNOWN_CLASSES = new HashSet<>(Arrays.asList(
            "BROWSING", "CHAT", "STREAMING", "MAIL", "VOIP"
    ));

    @GetMapping("/test-baseline")
    public ResponseEntity<Map<String, Object>> testFixedThresholdBaseline(
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        
        try {
            Instances rawData = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(rawData);

            Instances reducedKnownTraining = featureSelectionService.fitAndTransform(split.getKnownTrainingData());
            Instances reducedTesting = featureSelectionService.transform(split.getTestingData());

            long startTrain = System.currentTimeMillis();
            mlService.trainH2Classifier(reducedKnownTraining);
            long trainTime = System.currentTimeMillis() - startTrain;

            int totalTested = 0;
            int correctKnown = 0; 
            int correctNew = 0;   
            int falseKnown = 0;   
            int falseNew = 0;     
            
            for (int i = 0; i < reducedTesting.numInstances(); i++) {
                Instance testInst = reducedTesting.instance(i);
                String actualClass = reducedTesting.classAttribute().value((int) testInst.classValue());
                boolean isActuallyKnown = KNOWN_CLASSES.contains(actualClass);

                PredictionResult prediction = mlService.predictFixedThreshold(testInst, reducedTesting);

                if (prediction.getClassType().equals("KNOWN")) {
                    if (isActuallyKnown && prediction.getPredictedClass().equals(actualClass)) {
                        correctKnown++;
                    } else {
                        falseKnown++; 
                    }
                } else { 
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
            response.put("originalFeatureCount", rawData.numAttributes() - 1);
            response.put("selectedFeatureCount", featureSelectionService.getSelectedFeatureNames().size());
            response.put("trainTimeMs", trainTime);
            response.put("totalTested", totalTested);
            response.put("correctKnown", correctKnown);
            response.put("correctNew", correctNew);
            response.put("falseKnown", falseKnown);
            response.put("falseNew", falseNew);

            int totalActuallyKnown = correctKnown + falseNew;
            int totalActuallyNew = correctNew + falseKnown;
            
            double knownAccuracy = totalActuallyKnown > 0 ? (double) correctKnown / totalActuallyKnown : 0;
            double unknownAccuracy = totalActuallyNew > 0 ? (double) correctNew / totalActuallyNew : 0;
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

    @GetMapping("/extract-pseudo-negatives")
    public ResponseEntity<Map<String, Object>> extractPseudoNegatives(
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        try {
            Instances rawData = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(rawData);
            
            Instances reducedKnownTraining = featureSelectionService.fitAndTransform(split.getKnownTrainingData());
            Instances reducedUnlabelled = featureSelectionService.transform(split.getUnlabelledData());

            mlService.trainH2Classifier(reducedKnownTraining);

            long startExtract = System.currentTimeMillis();
            Instances pseudoNegatives = ocSvmService.selectPseudoNegatives(reducedKnownTraining, reducedUnlabelled);
            long extractTime = System.currentTimeMillis() - startExtract;

            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            response.put("extractTimeMs", extractTime);
            response.put("totalUnlabelledSamples", reducedUnlabelled.numInstances());
            response.put("pseudoNegativesFound", pseudoNegatives.numInstances());
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }

    @GetMapping("/train-h1")
    public ResponseEntity<Map<String, Object>> trainH1(
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        try {
            Instances rawData = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(rawData);
            
            Instances reducedKnownTraining = featureSelectionService.fitAndTransform(split.getKnownTrainingData());
            Instances reducedUnlabelled = featureSelectionService.transform(split.getUnlabelledData());

            mlService.trainH2Classifier(reducedKnownTraining);
            Instances pseudoNegatives = ocSvmService.selectPseudoNegatives(reducedKnownTraining, reducedUnlabelled);

            long startTrainH1 = System.currentTimeMillis();
            mlService.trainH1Classifier(reducedKnownTraining, pseudoNegatives);
            long trainH1Time = System.currentTimeMillis() - startTrainH1;

            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            response.put("h1TrainTimeMs", trainH1Time);
            response.put("knownSamplesUsed", reducedKnownTraining.numInstances());
            response.put("pseudoNegativesUsed", pseudoNegatives.numInstances());
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
}
