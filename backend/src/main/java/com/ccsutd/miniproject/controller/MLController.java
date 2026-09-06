package com.ccsutd.miniproject.controller;

import com.ccsutd.miniproject.dto.ConfidenceResult;
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

@RestController
@RequestMapping("/api/ml")
public class MLController {

    @Autowired
    private DatasetService datasetService;

    @Autowired
    private MachineLearningService mlService;

    @GetMapping("/test-h2")
    public ResponseEntity<Map<String, Object>> testH2Classifier(
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        
        try {
            // 1. Load and split dataset
            Instances data = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(data);

            // 2. Train H2 on KnownC
            long startTrain = System.currentTimeMillis();
            mlService.trainH2Classifier(split.getKnownTrainingData());
            long trainTime = System.currentTimeMillis() - startTrain;

            // 3. Test on a single KnownC instance
            Instance testKnown = split.getTestingData().instance(0); // Pick first testing sample
            ConfidenceResult resultKnown = mlService.getH2Confidence(testKnown, data);

            // 4. Return response to verify it works
            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            response.put("trainTimeMs", trainTime);
            
            Map<String, Object> sampleResult = new HashMap<>();
            sampleResult.put("actualClass", data.classAttribute().value((int) testKnown.classValue()));
            sampleResult.put("confidenceMetrics", resultKnown);
            
            response.put("samplePrediction", sampleResult);

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
}
