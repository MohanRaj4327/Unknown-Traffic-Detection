package com.ccsutd.miniproject.controller;

import com.ccsutd.miniproject.dto.PredictionResult;
import com.ccsutd.miniproject.dto.DatasetSplit;
import com.ccsutd.miniproject.dto.OcSvmResult;
import com.ccsutd.miniproject.service.AtsService;
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
    
    @Autowired
    private AtsService atsService;

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

            mlService.trainH2Classifier(reducedKnownTraining);

            int totalTested = 0, correctKnown = 0, correctNew = 0, falseKnown = 0, falseNew = 0;     
            
            for (int i = 0; i < reducedTesting.numInstances(); i++) {
                Instance testInst = reducedTesting.instance(i);
                String actualClass = reducedTesting.classAttribute().value((int) testInst.classValue());
                boolean isActuallyKnown = KNOWN_CLASSES.contains(actualClass);

                PredictionResult prediction = mlService.predictFixedThreshold(testInst, reducedTesting);

                if (prediction.getClassType().equals("KNOWN")) {
                    if (isActuallyKnown && prediction.getPredictedClass().equals(actualClass)) {
                        correctKnown++;
                    } else { falseKnown++; }
                } else { 
                    if (!isActuallyKnown) { correctNew++; } else { falseNew++; }
                }
                totalTested++;
            }

            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            
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

    @GetMapping("/train-full-cascade")
    public ResponseEntity<Map<String, Object>> trainFullCascade(
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        try {
            Instances rawData = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(rawData);
            
            Instances reducedKnownTraining = featureSelectionService.fitAndTransform(split.getKnownTrainingData());
            Instances reducedUnlabelled = featureSelectionService.transform(split.getUnlabelledData());
            Instances reducedTesting = featureSelectionService.transform(split.getTestingData());

            // Train H2 baseline needed for CfDmax extractions
            mlService.trainH2Classifier(reducedKnownTraining);

            // Phase 7: Extract pseudo-negatives & likely-knowns via OC-SVM
            OcSvmResult ocSvmResult = ocSvmService.selectPseudoNegatives(reducedKnownTraining, reducedUnlabelled);

            // Phase 10: ATS Calculation
            double atsAlpha = atsService.calculateAdaptiveThreshold(
                    reducedKnownTraining, 
                    ocSvmResult.getPseudoNegatives(), 
                    ocSvmResult.getLikelyKnowns()
            );
            
            // Set beta to ATS alpha as recommended by the paper
            mlService.setBetaThreshold(atsAlpha);

            // Phase 8: Train H1 Classifier
            long startTrainH1 = System.currentTimeMillis();
            mlService.trainH1Classifier(reducedKnownTraining, ocSvmResult.getPseudoNegatives());
            long trainH1Time = System.currentTimeMillis() - startTrainH1;

            // Phase 11: Final Cascade Testing
            int totalTested = 0, correctKnown = 0, correctNew = 0, falseKnown = 0, falseNew = 0;
            int h1EarlyBlocks = 0;
            
            for (int i = 0; i < reducedTesting.numInstances(); i++) {
                Instance testInst = reducedTesting.instance(i);
                String actualClass = reducedTesting.classAttribute().value((int) testInst.classValue());
                boolean isActuallyKnown = KNOWN_CLASSES.contains(actualClass);

                PredictionResult prediction = mlService.predictCascade(testInst, reducedTesting);

                if ("NEW_CLASS_DETECTED".equals(prediction.getH1Result())) {
                    h1EarlyBlocks++;
                }

                if (prediction.getClassType().equals("KNOWN")) {
                    if (isActuallyKnown && prediction.getPredictedClass().equals(actualClass)) {
                        correctKnown++;
                    } else { falseKnown++; }
                } else { 
                    if (!isActuallyKnown) { correctNew++; } else { falseNew++; }
                }
                totalTested++;
            }

            int totalActuallyKnown = correctKnown + falseNew;
            int totalActuallyNew = correctNew + falseKnown;
            
            double knownAccuracy = totalActuallyKnown > 0 ? (double) correctKnown / totalActuallyKnown : 0;
            double unknownAccuracy = totalActuallyNew > 0 ? (double) correctNew / totalActuallyNew : 0;
            double normalizedAccuracy = 0.5 * knownAccuracy + 0.5 * unknownAccuracy;

            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            response.put("atsCalculatedAlpha", atsAlpha);
            response.put("pseudoNegativesUsed", ocSvmResult.getPseudoNegatives().numInstances());
            response.put("likelyKnownsUsed", ocSvmResult.getLikelyKnowns().numInstances());
            response.put("h1TrainTimeMs", trainH1Time);
            
            Map<String, Object> cascadeMetrics = new HashMap<>();
            cascadeMetrics.put("totalTested", totalTested);
            cascadeMetrics.put("correctKnown", correctKnown);
            cascadeMetrics.put("correctNew", correctNew);
            cascadeMetrics.put("falseKnown", falseKnown);
            cascadeMetrics.put("falseNew", falseNew);
            cascadeMetrics.put("h1EarlyBlocks", h1EarlyBlocks);
            cascadeMetrics.put("knownAccuracy", knownAccuracy);
            cascadeMetrics.put("unknownAccuracy", unknownAccuracy);
            cascadeMetrics.put("normalizedAccuracy", normalizedAccuracy);
            response.put("cascadeMetrics", cascadeMetrics);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
}
