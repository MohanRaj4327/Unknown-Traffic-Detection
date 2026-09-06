package com.ccsutd.miniproject.controller;

import com.ccsutd.miniproject.dto.DatasetSplit;
import com.ccsutd.miniproject.service.DatasetService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/dataset")
public class DatasetController {

    @Autowired
    private DatasetService datasetService;

    @GetMapping("/info")
    public ResponseEntity<Map<String, Object>> getDatasetInfo(
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        
        try {
            Instances data = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(data);

            Map<String, Object> response = new HashMap<>();
            response.put("totalInstances", data.numInstances());
            response.put("numAttributes", data.numAttributes());
            
            Map<String, Integer> splitCounts = new HashMap<>();
            splitCounts.put("knownTraining", split.getKnownTrainingData().numInstances());
            splitCounts.put("unlabelled", split.getUnlabelledData().numInstances());
            splitCounts.put("testing", split.getTestingData().numInstances());
            
            response.put("splitSizes", splitCounts);
            response.put("status", "success");

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
}
